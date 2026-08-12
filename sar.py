import os
import gc
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import rasterio
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT
from scipy.ndimage import median_filter
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statistics import median

# ================================================================
# CONFIG — EDIT THESE PATHS TO MATCH YOUR KAGGLE DATASET LAYOUT
# ================================================================
DATA_ROOT = "/kaggle/input/competitions/anrf-aise-hack-2-0-round-2-sar-crop-health-yield-estimation"

VILLAGE_SHP_PATH = f"{DATA_ROOT}/Village_Shp/Village_Shp/Sokhda_Village.shp"
FARMS_SHP_PATH   = f"{DATA_ROOT}/Farm_boundaries_shp/Farm_boundaries_shp/Sokhda_Farms.shp"

RASTER_PATHS_BY_DATE = {
    "2025-06-06": f"{DATA_ROOT}/CAPELLA_C14_SM_SLC_HH_20250606072501_20250606072506/CAPELLA_C14_SM_SLC_HH_20250606072501_20250606072506.tif",
    "2025-06-19": f"{DATA_ROOT}/CAPELLA_C14_SM_SLC_HH_20250619021410_20250619021415/CAPELLA_C14_SM_SLC_HH_20250619021410_20250619021415.tif",
    "2025-08-14": f"{DATA_ROOT}/CAPELLA_C14_SM_SLC_HH_20250814031124_20250814031129/CAPELLA_C14_SM_SLC_HH_20250814031124_20250814031129.tif",
    "2025-10-13": f"{DATA_ROOT}/CAPELLA_C14_SM_SLC_HH_20251013022643_20251013022648/CAPELLA_C14_SM_SLC_HH_20251013022643_20251013022648.tif",
}
DATE_ORDER = ["2025-06-06", "2025-06-19", "2025-08-14", "2025-10-13"]
TARGET_CRS = "EPSG:32643"
VILLAGE_ID = 22

OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CROPS = 5
CROP_LABEL_MAP = {0: "Rice", 1: "Cotton", 2: "Maize", 3: "Bajra", 4: "Groundnut"}
CROP_INDEX_MAP = {v: k for k, v in CROP_LABEL_MAP.items()}
MARGIN_ZDIFF_THRESHOLD = 0.2
RESCUE_CONF_THRESHOLD = 0.75

# NOTE: Base yield coefficients derived from the dummy submission schema
CROP_YIELD_COEFF_T_PER_HA = {
    "Maize": 12.4, "Rice": 10.5, "Groundnut": 4.4, "Cotton": 4.4, "Bajra": 3.5,
}

# Will be populated dynamically in main() to calibrate health scores accurately
CROP_EXPECTED_PEAK_DB = {}

# 4-component crop-condition weights
HEALTH_FORMULA_WEIGHTS = {"w1_peak": 0.4, "w2_uniformity": 0.2, "w3_growth": 0.2, "w4_phenology": 0.2}

CROP_COLORS = {
    "Rice": "#2CA02C",       
    "Cotton": "#F5D300",     
    "Maize": "#FF8C00",      
    "Bajra": "#7B2D8E",      
    "Groundnut": "#8B4513",  
}

# Weighted feature+direction rules per crop
CROP_RULES = {
    "Rice":       [("dyn_range", +1, 1.0), ("temporal_var", +1, 0.8), ("peak_date_idx", +1, 0.6), ("rank_jun06", -1, 1.0)],
    "Cotton":     [("growth_ratio", +1, 1.0), ("time_to_peak_ratio", +1, 0.8)],
    "Maize":      [("peak_sharpness", +1, 1.0), ("dyn_range", +1, 0.7)],
    "Bajra":      [("temporal_mean", -1, 1.0), ("dyn_range", -1, 1.2)],
    "Groundnut":  [("var_range_ratio", +1, 1.0), ("curvature", -1, 0.9)],
}

DENSE_FEATURE_NAMES = [
    "slope_1", "slope_2", "slope_3",
    "dyn_range", "temporal_var", "temporal_mean",
    "rank_jun06", "rank_jun19", "rank_aug14", "rank_oct13",
    "curvature_1", "curvature_2",
    "abs_curvature_1", "abs_curvature_2",
    "peak_date_idx", "time_to_peak_ratio", "peak_mid_proximity", "peak_sharpness",
    "growth_ratio", "range_slope_interaction", "var_range_ratio",
    "seq_minmax_jun06", "seq_minmax_jun19", "seq_minmax_aug14", "seq_minmax_oct13",
    "seq_z_jun06", "seq_z_jun19", "seq_z_aug14", "seq_z_oct13",
    "disp_mean", "disp_iqr_mean",
]
DENSE_IDX = {name: i for i, name in enumerate(DENSE_FEATURE_NAMES)}


def resolve_target_crs(raster_paths_by_date, fallback_crs=TARGET_CRS):
    first_path = raster_paths_by_date[DATE_ORDER[0]]
    with rasterio.open(first_path) as src:
        if src.crs is not None:
            print(f"Raster CRS detected directly: {src.crs}")
            return src.crs
        if src.gcps and src.gcps[0]:
            gcp_list, gcp_crs = src.gcps
            if gcp_crs is not None:
                print(f"Raster has no direct CRS but GCPs are tagged with: {gcp_crs}")
                return gcp_crs
    return fallback_crs


def load_and_align_vectors(village_path, farms_path, target_crs):
    village_gdf = gpd.read_file(village_path)
    farms_gdf = gpd.read_file(farms_path).reset_index(drop=True)
    farms_gdf["id"] = range(len(farms_gdf))
    village_gdf = village_gdf.to_crs(target_crs)
    farms_gdf = farms_gdf.to_crs(target_crs)
    return village_gdf, farms_gdf


def extract_farm_stack(farm_geom, raster_paths_by_date, target_crs=TARGET_CRS):
    date_median, date_std, date_p10, date_p90 = [], [], [], []
    for date_key in DATE_ORDER:
        with rasterio.open(raster_paths_by_date[date_key]) as src:
            needs_vrt = (src.crs is None) or (src.crs.to_string() != target_crs)
            ctx = WarpedVRT(src, crs=target_crs) if needs_vrt else src
            try:
                out_image, _ = mask(ctx, [farm_geom], crop=True, nodata=0, filled=True)
            finally:
                if needs_vrt:
                    ctx.close()
        mag = np.abs(out_image[0])
        mag_valid = mag[mag > 0]
        if mag_valid.size < 4:
            date_median.append(np.nan); date_std.append(np.nan)
            date_p10.append(np.nan); date_p90.append(np.nan)
            continue
        mag_smooth = median_filter(mag_valid, size=3)
        db = 10 * np.log10(mag_smooth ** 2 + 1e-10)
        date_median.append(np.median(db))
        date_std.append(np.std(db))
        date_p10.append(np.percentile(db, 10))
        date_p90.append(np.percentile(db, 90))

    return {
        "seq_median": np.array(date_median, dtype=np.float32),
        "seq_std":    np.array(date_std, dtype=np.float32),
        "seq_iqr":    np.array(date_p90, dtype=np.float32) - np.array(date_p10, dtype=np.float32),
    }


def extract_all_farms(farms_gdf, raster_paths_by_date, village_id, target_crs, log_every=100):
    records, dropped = {}, []
    for i, row in farms_gdf.iterrows():
        stats = extract_farm_stack(row.geometry, raster_paths_by_date, target_crs=target_crs)
        farm_key = row.get("id", i)
        if np.isnan(stats["seq_median"]).any():
            dropped.append(farm_key)
            continue
        records[farm_key] = stats
        if i % log_every == 0:
            gc.collect()
    print(f"[{village_id}] extracted {len(records)}/{len(farms_gdf)} farms, {len(dropped)} dropped (no-data)")
    return records, dropped


def engineer_farm_features(stats):
    seq = stats["seq_median"]
    slopes = np.diff(seq)
    dyn_range = seq.max() - seq.min()
    temporal_var = seq.var()
    temporal_mean = seq.mean()
    ranks = np.argsort(np.argsort(seq)).astype(np.float32)
    curvature = slopes[1:] - slopes[:-1]
    abs_curvature = np.abs(curvature)
    peak_idx = np.argmax(seq)
    time_to_peak_ratio = peak_idx / (len(seq) - 1)
    peak_mid_proximity = -abs(peak_idx - 1.5)
    peak_sharpness = seq[peak_idx] - np.mean(np.delete(seq, peak_idx))
    growth_ratio = (seq[-1] + 1e-6) / (seq[0] + 1e-6)
    range_slope_interaction = dyn_range * slopes.mean()
    var_range_ratio = temporal_var / (dyn_range + 1e-6)
    seq_minmax = (seq - seq.min()) / (dyn_range + 1e-6)
    seq_z = (seq - temporal_mean) / (seq.std() + 1e-6)
    disp_mean = stats["seq_std"].mean()
    disp_iqr_mean = stats["seq_iqr"].mean()

    return np.concatenate([
        slopes, [dyn_range, temporal_var, temporal_mean], ranks, curvature, abs_curvature,
        [peak_idx, time_to_peak_ratio, peak_mid_proximity, peak_sharpness],
        [growth_ratio, range_slope_interaction, var_range_ratio], seq_minmax, seq_z,
        [disp_mean, disp_iqr_mean]
    ]).astype(np.float32)


def _feature_columns(dense_arr, feat_name):
    exact = [i for n, i in DENSE_IDX.items() if n == feat_name]
    if exact: return dense_arr[:, exact[0]]
    prefixed = [i for n, i in DENSE_IDX.items() if n.startswith(feat_name)]
    if prefixed: return dense_arr[:, prefixed].mean(axis=1)
    raise KeyError(f"Feature '{feat_name}' not found")


def score_crop_candidates(dense_arr):
    n = dense_arr.shape[0]
    raw_scores = np.zeros((n, N_CROPS), dtype=np.float32)
    for c, crop in enumerate(CROP_LABEL_MAP.values()):
        for feat_name, direction, weight in CROP_RULES[crop]:
            col = _feature_columns(dense_arr, feat_name)
            rank = col.argsort().argsort() / max(len(col) - 1, 1)
            raw_scores[:, c] += weight * (rank if direction == 1 else (1 - rank))

    z_scores = np.zeros_like(raw_scores)
    for c in range(N_CROPS):
        col = raw_scores[:, c]
        z_scores[:, c] = (col - col.mean()) / (col.std() + 1e-6)

    sorted_z = np.sort(z_scores, axis=1)
    margin = sorted_z[:, -1] - sorted_z[:, -2]
    pred = z_scores.argmax(axis=1)
    return np.where(margin >= MARGIN_ZDIFF_THRESHOLD, pred, -1), z_scores, margin


class AttentionCropModel(nn.Module):
    def __init__(self, dense_dim, n_crops=N_CROPS, seq_len=4, embed_dim=32, n_heads=4):
        super().__init__()
        self.embed = nn.Linear(1, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dense_path = nn.Sequential(
            nn.BatchNorm1d(dense_dim), nn.Linear(dense_dim, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
        )
        self.trunk = nn.Sequential(nn.Linear(embed_dim + 64, 64), nn.GELU(), nn.Dropout(0.1))
        self.crop_head = nn.Linear(64, n_crops)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.pool_query, std=0.02)

    def forward(self, x_seq, x_dense, return_attention=False):
        B = x_seq.size(0)
        x = self.embed(x_seq.unsqueeze(-1)) + self.pos_embed
        q = self.pool_query.expand(B, -1, -1)
        pooled, attn_w = self.attn(q, x, x)
        h = self.trunk(torch.cat([self.attn_norm(pooled.squeeze(1)), self.dense_path(x_dense)], dim=1))
        if return_attention: return self.crop_head(h), attn_w.squeeze(1)
        return self.crop_head(h)


class FarmDataset(Dataset):
    def __init__(self, X_seq, X_dense, y):
        self.X_seq, self.X_dense, self.y = torch.from_numpy(X_seq).float(), torch.from_numpy(X_dense).float(), torch.from_numpy(y).long()
    def __len__(self): return self.X_seq.shape[0]
    def __getitem__(self, idx): return self.X_seq[idx], self.X_dense[idx], self.y[idx]


def train_phase(model, train_loader, val_loader, epochs, lr, class_weights, tag, noise_std=0.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_acc, best_state = 0.0, None

    for epoch in range(epochs):
        model.train()
        tr_correct = tr_n = 0
        for seq_x, dense_x, y in train_loader:
            seq_x, dense_x, y = seq_x.to(DEVICE), dense_x.to(DEVICE), y.to(DEVICE)
            if noise_std > 0:
                seq_x, dense_x = seq_x + torch.randn_like(seq_x)*noise_std, dense_x + torch.randn_like(dense_x)*noise_std
            logits = model(seq_x, dense_x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tr_correct += (logits.argmax(1) == y).sum().item(); tr_n += y.size(0)

        model.eval()
        val_correct = val_n = 0
        with torch.no_grad():
            for seq_x, dense_x, y in val_loader:
                logits = model(seq_x.to(DEVICE), dense_x.to(DEVICE))
                val_correct += (logits.argmax(1) == y.to(DEVICE)).sum().item(); val_n += y.size(0)

        scheduler.step()
        val_acc = val_correct / max(val_n, 1)
        print(f"[{tag}] Epoch {epoch + 1}/{epochs} | train_acc={tr_correct / max(tr_n,1):.4f} | val_acc={val_acc:.4f}")
        if val_acc > best_acc: best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())

    if best_state is not None: model.load_state_dict(best_state)
    return model, best_acc


def compute_health_and_yield(seq, dense_feats, crop_type):
    w1, w2, w3, w4 = HEALTH_FORMULA_WEIGHTS["w1_peak"], HEALTH_FORMULA_WEIGHTS["w2_uniformity"], HEALTH_FORMULA_WEIGHTS["w3_growth"], HEALTH_FORMULA_WEIGHTS["w4_phenology"]

    peak_db = seq.max()
    expected_peak = CROP_EXPECTED_PEAK_DB.get(crop_type, 30.0)
    P_c = np.exp(-((peak_db - expected_peak) ** 2) / (2 * 3.0 ** 2)) 
    U_c = np.clip(1 - (dense_feats[DENSE_IDX["disp_iqr_mean"]] / 20.0), 0, 1)
    mean_slope = np.mean([dense_feats[DENSE_IDX[k]] for k in DENSE_FEATURE_NAMES if k.startswith("slope_")])
    G_c = 1.0 / (1.0 + np.exp(-mean_slope / 2.0))
    abs_curv = np.mean([dense_feats[DENSE_IDX[k]] for k in DENSE_FEATURE_NAMES if k.startswith("abs_curvature")])
    S_c = np.clip(1 - (abs_curv / 6.7), 0, 1)

    health = float(np.clip(100 * (w1 * P_c + w2 * U_c + w3 * G_c + w4 * S_c), 0, 100))
    yield_est = round(CROP_YIELD_COEFF_T_PER_HA[crop_type] * (health / 100.0), 2)
    return health, yield_est


def main():
    resolved_crs = resolve_target_crs(RASTER_PATHS_BY_DATE)
    village_gdf, farms_gdf = load_and_align_vectors(VILLAGE_SHP_PATH, FARMS_SHP_PATH, resolved_crs)
    
    raw_stats_by_farm, dropped_farms = extract_all_farms(farms_gdf, RASTER_PATHS_BY_DATE, VILLAGE_ID, target_crs=resolved_crs)
    farms_gdf_valid = farms_gdf[farms_gdf["id"].isin(raw_stats_by_farm.keys())].reset_index(drop=True)

    farm_features = {}
    for farm_id, stats in raw_stats_by_farm.items():
        farm_features[farm_id] = {"seq_median": stats["seq_median"], "dense": engineer_farm_features(stats)}

    farm_ids_all = list(farm_features.keys())
    X_seq_all = np.stack([farm_features[fid]["seq_median"] for fid in farm_ids_all])
    X_dense_all_raw = np.stack([farm_features[fid]["dense"] for fid in farm_ids_all])

    pseudo_labels, z_scores, margins = score_crop_candidates(X_dense_all_raw)
    labeled_mask = pseudo_labels != -1

    idx_labeled = np.where(labeled_mask)[0]
    idx_tr, idx_val = train_test_split(idx_labeled, test_size=0.15, random_state=42, stratify=pseudo_labels[idx_labeled])

    clip_lo, clip_hi = np.percentile(X_dense_all_raw[idx_tr], 1, axis=0), np.percentile(X_dense_all_raw[idx_tr], 99, axis=0)
    scaler = RobustScaler().fit(np.clip(X_dense_all_raw[idx_tr], clip_lo, clip_hi))
    X_dense_scaled_all = scaler.transform(np.clip(X_dense_all_raw, clip_lo, clip_hi)).astype(np.float32)

    seq_mu, seq_sigma = X_seq_all[idx_tr].mean(axis=0), X_seq_all[idx_tr].std(axis=0) + 1e-6
    X_seq_norm_all = ((X_seq_all - seq_mu) / seq_sigma).astype(np.float32)

    train_loader = DataLoader(FarmDataset(X_seq_norm_all[idx_tr], X_dense_scaled_all[idx_tr], pseudo_labels[idx_tr]), batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(FarmDataset(X_seq_norm_all[idx_val], X_dense_scaled_all[idx_val], pseudo_labels[idx_val]), batch_size=32, shuffle=False)

    class_counts = np.bincount(pseudo_labels[idx_tr], minlength=N_CROPS)
    class_weights = torch.tensor(class_counts.sum() / (N_CROPS * np.clip(class_counts, 1, None)), dtype=torch.float32).to(DEVICE)

    model = AttentionCropModel(dense_dim=X_dense_all_raw.shape[1]).to(DEVICE)
    print("=" * 60); print("PHASE 1: Seed training"); print("=" * 60)
    model, phase1_val_acc = train_phase(model, train_loader, val_loader, epochs=20, lr=1e-3, class_weights=class_weights, tag="Phase1")
    
    print("\n" + "=" * 60); print("PHASE 2: AUP Rescue"); print("=" * 60)
    model.eval()
    unlabeled_idx = np.where(pseudo_labels == -1)[0]
    rescued_labels = pseudo_labels.copy()

    if unlabeled_idx.size > 0:
        with torch.no_grad():
            probs = torch.softmax(model(torch.from_numpy(X_seq_norm_all[unlabeled_idx]).float().to(DEVICE), torch.from_numpy(X_dense_scaled_all[unlabeled_idx]).float().to(DEVICE)), dim=1).cpu().numpy()
        max_probs, preds = probs.max(axis=1), probs.argmax(axis=1)
        confident = max_probs >= RESCUE_CONF_THRESHOLD
        rescued_labels[unlabeled_idx[confident]] = preds[confident]
        print(f"Rescued {confident.sum()}/{unlabeled_idx.size} farms.")

    print("\n" + "=" * 60); print("PHASE 3: Consistency-regularized retrain"); print("=" * 60)
    idx_labeled_p3 = np.where(rescued_labels != -1)[0]
    idx_tr3, idx_val3 = train_test_split(idx_labeled_p3, test_size=0.15, random_state=42, stratify=rescued_labels[idx_labeled_p3])

    train_loader_p3 = DataLoader(FarmDataset(X_seq_norm_all[idx_tr3], X_dense_scaled_all[idx_tr3], rescued_labels[idx_tr3]), batch_size=32, shuffle=True, drop_last=True)
    val_loader_p3 = DataLoader(FarmDataset(X_seq_norm_all[idx_val3], X_dense_scaled_all[idx_val3], rescued_labels[idx_val3]), batch_size=32, shuffle=False)

    class_counts_p3 = np.bincount(rescued_labels[idx_tr3], minlength=N_CROPS)
    class_weights_p3 = torch.tensor(class_counts_p3.sum() / (N_CROPS * np.clip(class_counts_p3, 1, None)), dtype=torch.float32).to(DEVICE)
    model, phase3_val_acc = train_phase(model, train_loader_p3, val_loader_p3, epochs=15, lr=5e-4, class_weights=class_weights_p3, tag="Phase3", noise_std=0.05)

    model.eval()
    with torch.no_grad():
        logits_all, attn_all = model(torch.from_numpy(X_seq_norm_all).float().to(DEVICE), torch.from_numpy(X_dense_scaled_all).float().to(DEVICE), return_attention=True)
        probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()
        attn_all = attn_all.cpu().numpy()

    final_crop_pred = probs_all.argmax(axis=1)
    final_crop_conf = probs_all.max(axis=1)

    # -------------------------------------------------------------------------
    # DYNAMIC HEALTH CALIBRATION FIX
    # Calculates the empirical 95th percentile peak for each crop dynamically 
    # so health scores can reach ~100 instead of capping at ~50.
    # -------------------------------------------------------------------------
    global CROP_EXPECTED_PEAK_DB
    for c_idx, crop_name in CROP_LABEL_MAP.items():
        crop_mask = (final_crop_pred == c_idx) & (rescued_labels != -1)
        if crop_mask.sum() > 0:
            CROP_EXPECTED_PEAK_DB[crop_name] = np.percentile(X_seq_all[crop_mask].max(axis=1), 95)
        else:
            CROP_EXPECTED_PEAK_DB[crop_name] = 35.0  

    # ---- Step 8: health / yield + submission assembly + IMPUTATION ----
    rows = []
    covered_mask = rescued_labels != -1
    
    valid_crops = []
    crop_health_map = {crop: [] for crop in CROP_LABEL_MAP.values()}
    crop_yield_map = {crop: [] for crop in CROP_LABEL_MAP.values()}
    crop_attn_map = {crop: {"jun06": [], "jun19": [], "aug14": [], "oct13": [], "conf": []} for crop in CROP_LABEL_MAP.values()}

    for i, fid in enumerate(farm_ids_all):
        if not covered_mask[i]:
            continue
        crop_type = CROP_LABEL_MAP[final_crop_pred[i]]
        health, yield_est = compute_health_and_yield(X_seq_all[i], X_dense_all_raw[i], crop_type)
        
        valid_crops.append(crop_type)
        crop_health_map[crop_type].append(health)
        crop_yield_map[crop_type].append(yield_est)
        
        attn_0, attn_1, attn_2, attn_3 = float(attn_all[i, 0]), float(attn_all[i, 1]), float(attn_all[i, 2]), float(attn_all[i, 3])
        conf = float(final_crop_conf[i])
        
        crop_attn_map[crop_type]["jun06"].append(attn_0); crop_attn_map[crop_type]["jun19"].append(attn_1)
        crop_attn_map[crop_type]["aug14"].append(attn_2); crop_attn_map[crop_type]["oct13"].append(attn_3)
        crop_attn_map[crop_type]["conf"].append(conf)

        rows.append({
            "village_id": VILLAGE_ID, "farm_id": fid + 1, "crop_type": crop_type,
            "health_index": round(health, 1), "yield_estimate_to_date": yield_est,
            "_attn_jun06": round(attn_0, 3), "_attn_jun19": round(attn_1, 3),
            "_attn_aug14": round(attn_2, 3), "_attn_oct13": round(attn_3, 3),
            "_confidence": round(conf, 3), "_imputed": False
        })
        
    mode_crop = max(set(valid_crops), key=valid_crops.count) if valid_crops else "Bajra"
    median_health = {crop: median(vals) if vals else 50.0 for crop, vals in crop_health_map.items()}
    median_yield = {crop: median(vals) if vals else CROP_YIELD_COEFF_T_PER_HA[crop]*0.5 for crop, vals in crop_yield_map.items()}
    
    if crop_attn_map[mode_crop]["conf"]:
        med_attn_0 = median(crop_attn_map[mode_crop]["jun06"])
        med_attn_1 = median(crop_attn_map[mode_crop]["jun19"])
        med_attn_2 = median(crop_attn_map[mode_crop]["aug14"])
        med_attn_3 = median(crop_attn_map[mode_crop]["oct13"])
        med_conf   = median(crop_attn_map[mode_crop]["conf"])
    else:
        med_attn_0 = med_attn_1 = med_attn_2 = med_attn_3 = 0.25; med_conf = 0.50
    
    for i, fid in enumerate(farm_ids_all):
        if covered_mask[i]: continue
        rows.append({
            "village_id": VILLAGE_ID, "farm_id": fid + 1, "crop_type": mode_crop,
            "health_index": round(median_health[mode_crop], 1), "yield_estimate_to_date": round(median_yield[mode_crop], 2),
            "_attn_jun06": round(med_attn_0, 3), "_attn_jun19": round(med_attn_1, 3),
            "_attn_aug14": round(med_attn_2, 3), "_attn_oct13": round(med_attn_3, 3), "_confidence": round(med_conf, 3),
            "_imputed": True
        })
        
    for fid in dropped_farms:
         rows.append({
            "village_id": VILLAGE_ID, "farm_id": fid + 1, "crop_type": mode_crop,
            "health_index": round(median_health[mode_crop], 1), "yield_estimate_to_date": round(median_yield[mode_crop], 2),
            "_attn_jun06": round(med_attn_0, 3), "_attn_jun19": round(med_attn_1, 3),
            "_attn_aug14": round(med_attn_2, 3), "_attn_oct13": round(med_attn_3, 3), "_confidence": round(med_conf, 3),
            "_imputed": True
        })

    submission_full = pd.DataFrame(rows).sort_values("farm_id").reset_index(drop=True)
    submission_csv = submission_full[["village_id", "farm_id", "crop_type", "health_index", "yield_estimate_to_date"]]
    submission_csv.to_csv(f"{OUT_DIR}/submission.csv", index=False)
    submission_full.to_csv(f"{OUT_DIR}/submission_with_attention.csv", index=False)

    farms_gdf["crop_type"] = submission_full["crop_type"].values
    farms_gdf["health_index"] = submission_full["health_index"].values
    farms_gdf["yield_estimate_to_date"] = submission_full["yield_estimate_to_date"].values

    valid_mask = ~submission_full["_imputed"].values
    plot_seq, plot_attn, plot_crop, plot_health, plot_yield = [], [], [], [], []
    for _, row in submission_full[valid_mask].iterrows():
        orig_idx = farm_ids_all.index(row["farm_id"] - 1) 
        plot_seq.append(X_seq_all[orig_idx])
        plot_attn.append([row["_attn_jun06"], row["_attn_jun19"], row["_attn_aug14"], row["_attn_oct13"]])
        plot_crop.append(row["crop_type"])
        plot_health.append(row["health_index"])
        plot_yield.append(row["yield_estimate_to_date"])
        
    covered_seq, covered_attn, covered_crop, covered_health, covered_yield = np.array(plot_seq), np.array(plot_attn), np.array(plot_crop), np.array(plot_health), np.array(plot_yield)

    # ================================================================
    # STEP 9 — SEVEN SEPARATE FIGURES FOR THE WRITEUP
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 9))
    village_gdf.boundary.plot(ax=ax, color="red", linewidth=1.5, label="Village Boundary")
    farms_gdf.boundary.plot(ax=ax, color="#5FA25F", linewidth=0.6, label="Farm Boundaries")
    ax.set_title("Sokhda Village and Farm Boundaries", fontsize=13, fontweight="bold")
    ax.set_xlabel("Easting"); ax.set_ylabel("Northing")
    ax.legend(handles=[mpatches.Patch(color="red", label="Village Boundary"), mpatches.Patch(color="#5FA25F", label="Farm Boundaries")], loc="upper right")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig1_village_farm_boundaries.png", dpi=150); plt.close(fig)

    village_gdf_wgs84, farms_gdf_wgs84 = village_gdf.to_crs("EPSG:4326"), farms_gdf.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 9))
    village_gdf_wgs84.boundary.plot(ax=ax, color="black", linewidth=1.2)
    for crop, color in CROP_COLORS.items():
        subset = farms_gdf_wgs84[farms_gdf_wgs84["crop_type"] == crop]
        if len(subset): subset.plot(ax=ax, color=color, edgecolor="black", linewidth=0.15, label=crop)
    ax.set_title("Crop Distribution Map - Sokhda Village", fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(handles=[mpatches.Patch(color=c, label=n) for n, c in CROP_COLORS.items()], title="Crop Type", loc="upper right")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig2_crop_distribution_map.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 9))
    village_gdf_wgs84.boundary.plot(ax=ax, color="black", linewidth=1.2)
    farms_gdf_wgs84.plot(ax=ax, column="health_index", cmap="RdYlGn", vmin=0, vmax=100, legend=True, edgecolor="black", linewidth=0.15, legend_kwds={"label": "Crop Health (%)", "shrink": 0.7})
    ax.set_title("Crop Health Map - Sokhda Village", fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig3_crop_health_map.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Crop-wise Yield Maps - Sokhda Village", fontsize=15, fontweight="bold")
    flat_axes = axes.flatten()
    for i, crop in enumerate(["Bajra", "Groundnut", "Maize", "Cotton", "Rice"]):
        ax = flat_axes[i]
        village_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0)
        subset = farms_gdf[farms_gdf["crop_type"] == crop]
        if len(subset): subset.plot(ax=ax, column="yield_estimate_to_date", cmap="Greens", legend=True, edgecolor="black", linewidth=0.15, legend_kwds={"label": "Yield (t/ha)", "shrink": 0.75})
        ax.set_title(f"{crop} Yield Map", fontsize=11, fontweight="bold")
        ax.set_xlabel("Easting", fontsize=8); ax.set_ylabel("Northing", fontsize=8); ax.tick_params(labelsize=7)
    flat_axes[-1].axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig4_crop_yield_maps.png", dpi=150); plt.close(fig)

    x_pos, date_labels = np.arange(4), ["Jun 06", "Jun 19", "Aug 14", "Oct 13"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for crop, color in CROP_COLORS.items():
        crop_mask = covered_crop == crop
        n = int(crop_mask.sum())
        if n > 0: ax.plot(x_pos, covered_seq[crop_mask].mean(axis=0), marker="o", markersize=7, linewidth=2.2, color=color, label=f"{crop} (n={n})")
    ax.set_xticks(x_pos); ax.set_xticklabels(date_labels)
    ax.set_xlabel("Acquisition Date"); ax.set_ylabel("Mean HH Backscatter (dB)")
    ax.set_title("Temporal Backscatter Trajectories by Crop Type", fontsize=13, fontweight="bold")
    ax.legend(loc="best", title="Crop (farm count)"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig5_phenology_trajectories.png", dpi=150); plt.close(fig)

    crop_names = list(CROP_LABEL_MAP.values())
    attn_matrix, crop_counts = np.full((5, 4), np.nan), []
    for i, crop in enumerate(crop_names):
        crop_mask = covered_crop == crop
        crop_counts.append(int(crop_mask.sum()))
        if crop_mask.sum() > 0: attn_matrix[i] = covered_attn[crop_mask].mean(axis=0)
    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = np.nanmax(attn_matrix) if np.isfinite(np.nanmax(attn_matrix)) else 1.0
    cmap = plt.cm.viridis.copy(); cmap.set_bad(color="lightgray")
    im = ax.imshow(np.ma.masked_invalid(attn_matrix), cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(x_pos); ax.set_xticklabels(date_labels)
    ax.set_yticks(np.arange(5)); ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(crop_names, crop_counts)])
    for i in range(5):
        for j in range(4):
            if not np.isnan(attn_matrix[i, j]):
                ax.text(j, i, f"{attn_matrix[i, j]:.2f}", ha="center", va="center", color="white" if attn_matrix[i, j] < vmax * 0.6 else "black", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Mean Attention Weight", shrink=0.8)
    ax.set_title("Model Attention Weights by Crop and Acquisition Date", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig6_attention_heatmap.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    for crop, color in CROP_COLORS.items():
        crop_mask = covered_crop == crop
        if crop_mask.sum() > 0: ax.scatter(covered_health[crop_mask], covered_yield[crop_mask], color=color, label=crop, alpha=0.75, edgecolor="black", linewidth=0.4, s=55)
    ax.set_xlabel("Health Index (0-100)"); ax.set_ylabel("Yield Estimate (t/ha)")
    ax.set_title("Deterministic Yield-Scaling Formula by Crop Type", fontsize=13, fontweight="bold")
    ax.legend(loc="best", title="Crop Type"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig7_health_vs_yield_scatter.png", dpi=150); plt.close(fig)

    # ================================================================
    # STEP 10 — TERMINAL SUMMARY (mirrors what the figures show)
    # ================================================================
    print("\n" + "=" * 70)
    print("FINAL SUBMISSION SUMMARY")
    print("=" * 70)

    print(f"\nTotal farms: {len(submission_full)}  "
          f"(model-labeled: {(~submission_full['_imputed']).sum()}, "
          f"imputed: {submission_full['_imputed'].sum()})")

    print("\nCrop distribution:")
    print(submission_full["crop_type"].value_counts().to_string())

    print("\nConfidence (model-labeled farms only):")
    print(submission_full.loc[~submission_full["_imputed"], "_confidence"].describe().to_string())
    low_conf = ((submission_full["_confidence"] < 0.6) & (~submission_full["_imputed"])).sum()
    print(f"Model-labeled farms with confidence < 0.6: {low_conf} "
          f"({100*low_conf/max((~submission_full['_imputed']).sum(),1):.1f}%)")

    print("\nHealth index by crop:")
    print(submission_full.groupby("crop_type")["health_index"]
          .agg(["count", "mean", "std", "min", "max"]).round(2).to_string())

    print("\nYield estimate (t/ha) by crop:")
    print(submission_full.groupby("crop_type")["yield_estimate_to_date"]
          .agg(["count", "mean", "std", "min", "max"]).round(2).to_string())

    print("\nAttention weight by crop and date (mean):")
    attn_summary = pd.DataFrame(attn_matrix, index=crop_names, columns=date_labels).round(3)
    print(attn_summary.to_string())

    print("\nPhase accuracies — Phase1 val_acc: {:.4f} | Phase3 val_acc: {:.4f}".format(
        phase1_val_acc, phase3_val_acc))
    print("=" * 70 + "\n")

    return submission_csv, submission_full

if __name__ == "__main__":
    submission_csv, submission_full = main()