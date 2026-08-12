import os, gc, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling
from rasterio.io import MemoryFile
import geopandas as gpd
from shapely.geometry import mapping, MultiPolygon, Polygon
from scipy.ndimage import median_filter, sobel
from skimage.feature import graycomatrix, graycoprops
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
warnings.filterwarnings('ignore')

# ── 0. CONFIGURATION ──────────────────────────────────────────────────────────
def get_kaggle_dataset_path():
    base_path = Path('/kaggle/input')
    if not base_path.exists():
        return Path.cwd()
    for child in base_path.iterdir():
        if child.is_dir() and ('anrf' in child.name.lower() or 'sar' in child.name.lower() or 'sokhda' in child.name.lower()):
            return child
    return base_path

DATA_PATH = get_kaggle_dataset_path()
print(f"Dataset root: {DATA_PATH}")

GEO_FILES = [
    "CAPELLA_C14_SM_GEO_HH_20250606072501_20250606072506_preview.tif",
    "CAPELLA_C14_SM_GEO_HH_20250619021410_20250619021415_preview.tif",
    "CAPELLA_C14_SM_GEO_HH_20250814031124_20250814031129_preview.tif",
    "CAPELLA_C14_SM_GEO_HH_20251013022643_20251013022648_preview.tif",
]
DATE_LABELS = ["Jun06", "Jun19", "Aug14", "Oct13"]
DEM_FILENAME_HINT = "dem"
CROP_CLASSES = ["Rice", "Cotton", "Maize", "Bajra", "Groundnut"]
N_CROPS = len(CROP_CLASSES)
APPLY_DEM_CORRECTION = False
GLCM_LEVELS = 32
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
MIN_PIXELS_FOR_FULL_CNN_TRUST = 9
RANDOM_STATE = 42
PATCH_SIZE = 16
CROP_COLORS = {'Rice': '#1f77b4', 'Cotton': '#ff7f0e', 'Maize': '#2ca02c',
               'Bajra': '#d62728', 'Groundnut': '#9467bd'}
CROP_YIELD_BENCHMARKS_T_HA = {
    "Rice":      (2.2, 4.8),
    "Cotton":    (1.0, 2.6),
    "Maize":     (2.5, 6.0),
    "Bajra":     (1.0, 2.6),
    "Groundnut": (1.0, 2.4),
}

# ── 1. SAR PREPROCESSING ─────────────────────────────────────────────────────
def amp_to_db(amp, eps=1e-6):
    return 20.0 * np.log10(np.maximum(amp.astype(np.float32), eps))

def find_file(base, hint):
    matches = [p for p in Path(base).rglob("*") if hint.lower() in p.name.lower()
               and p.suffix.lower() in ('.tif', '.tiff')]
    return matches[0] if matches else None

def compute_local_incidence_correction(dem_path, ref_src):
    if dem_path is None:
        print("  [DEM] No DEM found — skipping terrain correction.")
        return None
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1).astype(np.float32)
        dem_resampled = np.empty((ref_src.height, ref_src.width), dtype=np.float32)
        reproject(source=dem, destination=dem_resampled,
                  src_transform=dem_src.transform, src_crs=dem_src.crs,
                  dst_transform=ref_src.transform, dst_crs=ref_src.crs,
                  resampling=Resampling.bilinear)
    px_size = abs(ref_src.transform.a)
    gy, gx = sobel(dem_resampled, axis=0) / (8 * px_size), sobel(dem_resampled, axis=1) / (8 * px_size)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    nominal_incidence = np.radians(35.0)
    local_incidence = np.clip(nominal_incidence - slope, np.radians(5), np.radians(80))
    correction_db = 10.0 * np.log10(np.maximum(np.cos(local_incidence), 1e-3))
    correction_db -= 10.0 * np.log10(np.cos(nominal_incidence))
    return correction_db.astype(np.float32)

def load_sar_stack(base_dir):
    sar_paths = []
    for f in GEO_FILES:
        matches = list(Path(base_dir).rglob(f"*{f}"))
        if not matches:
            raise FileNotFoundError(f"Could not find {f} under {base_dir}")
        sar_paths.append(str(matches[0]))
    dem_path = find_file(base_dir, DEM_FILENAME_HINT)
    print(f"  DEM located: {dem_path}" if dem_path else "  DEM not found.")
    return sar_paths, dem_path

# ── 2. FARM <-> VILLAGE SPATIAL JOIN ─────────────────────────────────────────
def spatial_join_farms_to_villages(farms_gdf, village_gdf, farm_id_col, village_id_col):
    farms_gdf = farms_gdf.copy()
    if village_gdf.crs != farms_gdf.crs:
        village_gdf = village_gdf.to_crs(farms_gdf.crs)
    centroids = farms_gdf.geometry.centroid
    centroid_gdf = gpd.GeoDataFrame({farm_id_col: farms_gdf[farm_id_col]},
                                     geometry=centroids, crs=farms_gdf.crs)
    joined = gpd.sjoin(centroid_gdf, village_gdf[[village_id_col, 'geometry']],
                       how='left', predicate='within')
    dup_counts = joined.groupby(farm_id_col).size()
    n_ambiguous = int((dup_counts > 1).sum())
    if n_ambiguous > 0:
        print(f"  [village join] {n_ambiguous} farms matched >1 village — keeping first.")
    joined = joined.drop_duplicates(subset=farm_id_col)
    n_missed = int(joined[village_id_col].isnull().sum())
    if n_missed > 0:
        print(f"  [village join] {n_missed} centroids outside any village polygon — running nearest fallback.")
        missed_ids = joined.loc[joined[village_id_col].isnull(), farm_id_col]
        missed_centroids = centroid_gdf[centroid_gdf[farm_id_col].isin(missed_ids)]
        try:
            nearest = gpd.sjoin_nearest(missed_centroids, village_gdf[[village_id_col, 'geometry']], how='left')
            nearest = nearest.drop_duplicates(subset=farm_id_col)
            fill_map = dict(zip(nearest[farm_id_col], nearest[village_id_col]))
            joined.loc[joined[village_id_col].isnull(), village_id_col] = \
                joined.loc[joined[village_id_col].isnull(), farm_id_col].map(fill_map)
        except Exception as e:
            print(f"  [village join] Nearest fallback failed ({e}) — remaining farms stay UNKNOWN.")
    farms_gdf = farms_gdf.merge(joined[[farm_id_col, village_id_col]], on=farm_id_col, how='left')
    farms_gdf[village_id_col] = farms_gdf[village_id_col].fillna('UNKNOWN')
    print(f"  [village join] Distribution: {farms_gdf[village_id_col].value_counts().to_dict()}")
    return farms_gdf

def _ensure_clean_village_id(farms_gdf, village_id_col):
    raw = farms_gdf[village_id_col]
    non_unknown = raw[raw != 'UNKNOWN']
    coerced_numeric = pd.to_numeric(non_unknown, errors='coerce')
    frac_numeric = coerced_numeric.notnull().mean() if len(non_unknown) > 0 else 0.0
    if frac_numeric > 0.95:
        def to_clean(v):
            if v == 'UNKNOWN': return 'UNKNOWN'
            n = pd.to_numeric(v, errors='coerce')
            return str(int(n)) if pd.notnull(n) else 'UNKNOWN'
        farms_gdf[village_id_col] = raw.apply(to_clean)
    else:
        farms_gdf[village_id_col] = raw.astype(str).str.strip()
    n_unknown = int((farms_gdf[village_id_col] == 'UNKNOWN').sum())
    n_villages = farms_gdf.loc[farms_gdf[village_id_col] != 'UNKNOWN', village_id_col].nunique()
    print(f"  [village_id] Cleaned: {n_villages} distinct village(s), {n_unknown}/{len(farms_gdf)} UNKNOWN.")
    return farms_gdf

# ── 3. FARM-LEVEL PIXEL EXTRACTION ───────────────────────────────────────────
def get_valid_geometries(geom):
    if isinstance(geom, MultiPolygon): return [mapping(p) for p in geom.geoms]
    elif isinstance(geom, Polygon): return [mapping(geom)]
    return [mapping(geom)]

def _stack_correction_as_band(src, correction_surface):
    band1 = src.read(1)
    memfile = MemoryFile()
    dst = memfile.open(driver='GTiff', height=src.height, width=src.width, count=2,
                       dtype='float32', crs=src.crs, transform=src.transform)
    dst.write(band1.astype(np.float32), 1)
    dst.write(correction_surface.astype(np.float32), 2)
    return memfile, dst

def extract_farm_arrays(geom, sar_srcs, correction_surfaces, error_counter=None):
    geom_list = get_valid_geometries(geom)
    arrays = []
    pixel_area_m2 = None
    for i, src in enumerate(sar_srcs):
        corr_surface = correction_surfaces[i] if correction_surfaces is not None else None
        try:
            if corr_surface is not None:
                memfile, dst = _stack_correction_as_band(src, corr_surface)
                try:
                    out_image, out_transform = rio_mask(dst, geom_list, crop=True, nodata=0, filled=True)
                    amp_2d = out_image[0].astype(np.float32)
                    corr_2d = out_image[1].astype(np.float32)
                finally:
                    dst.close(); memfile.close()
                db_2d = median_filter(amp_to_db(amp_2d), size=3).astype(np.float32) + corr_2d
            else:
                out_image, out_transform = rio_mask(src, geom_list, crop=True, nodata=0, filled=True)
                amp_2d = out_image[0].astype(np.float32)
                db_2d = median_filter(amp_to_db(amp_2d), size=3).astype(np.float32)
            if pixel_area_m2 is None:
                pixel_area_m2 = abs(out_transform.a) * abs(out_transform.e)
            arrays.append(db_2d)
        except Exception as e:
            arrays.append(None)
            if error_counter is not None:
                key = f"{DATE_LABELS[i]}:{type(e).__name__}"
                error_counter[key] = error_counter.get(key, 0) + 1
    return arrays, pixel_area_m2

def stack_valid_dates(arrays):
    valid = [a for a in arrays if a is not None]
    if len(valid) < 4: return None
    min_h = min(a.shape[0] for a in valid)
    min_w = min(a.shape[1] for a in valid)
    if min_h < 2 or min_w < 2: return None
    return np.stack([a[:min_h, :min_w] for a in valid], axis=-1)

def resize_patch_fixed(stack_3d, size=16):
    H, W, C = stack_3d.shape
    if H == size and W == size: return stack_3d.astype(np.float32)
    row_idx = np.linspace(0, H - 1, size)
    col_idx = np.linspace(0, W - 1, size)
    r0 = np.clip(np.floor(row_idx).astype(int), 0, H - 1)
    c0 = np.clip(np.floor(col_idx).astype(int), 0, W - 1)
    return stack_3d[r0][:, c0].astype(np.float32)

# ── 4. TEMPORAL + GLCM FEATURE ENGINEERING ───────────────────────────────────
def glcm_features(band_2d, levels=GLCM_LEVELS):
    empty = {'contrast': 0.0, 'homogeneity': 0.0, 'entropy': 0.0, 'correlation': 0.0}
    if band_2d.size < 9 or np.all(band_2d == 0): return empty
    valid = band_2d[band_2d != 0]
    if valid.size < 9: return empty
    vmin, vmax = np.percentile(valid, 1), np.percentile(valid, 99)
    if vmax <= vmin: return empty
    clipped = np.clip(band_2d, vmin, vmax)
    quantized = ((clipped - vmin) / (vmax - vmin) * (levels - 1)).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                        levels=levels, symmetric=True, normed=True)
    n_angles = glcm.shape[3]
    contrast_raw = float(np.mean(graycoprops(glcm, 'contrast')))
    range_sq = float((vmax - vmin) ** 2) + 1e-6
    contrast = contrast_raw / range_sq * (levels - 1) ** 2
    homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    correlation = float(np.mean(graycoprops(glcm, 'correlation')))
    entropies = []
    for a in range(n_angles):
        p = glcm[:, :, 0, a]
        p_nonzero = p[p > 0]
        entropies.append(float(-np.sum(p_nonzero * np.log2(p_nonzero))) if p_nonzero.size > 0 else 0.0)
    entropy = float(np.mean(entropies))
    return {'contrast': contrast, 'homogeneity': homogeneity, 'entropy': entropy, 'correlation': correlation}

def build_farm_features(stack_3d):
    feats = {}
    valid_mask = np.all(stack_3d > 0, axis=-1)
    n_valid = valid_mask.sum()
    per_date_mean = []
    for i, label in enumerate(DATE_LABELS):
        band = stack_3d[..., i]
        vals = band[valid_mask] if n_valid >= 4 else band[band > 0]
        if vals.size == 0: vals = np.array([0.0], dtype=np.float32)
        mean_v, med_v, var_v = float(np.mean(vals)), float(np.median(vals)), float(np.var(vals))
        std_v = float(np.std(vals))
        cv_v = std_v / (abs(mean_v) + 1e-3)
        feats[f'{label}_mean'] = mean_v; feats[f'{label}_median'] = med_v
        feats[f'{label}_var'] = var_v; feats[f'{label}_cv'] = cv_v
        per_date_mean.append(mean_v)
        tex = glcm_features(band)
        for k, v in tex.items(): feats[f'{label}_glcm_{k}'] = v
    t1, t2, t3, t4 = per_date_mean
    feats['growth_jun_aug'] = t3 - t2; feats['harvest_decline'] = t4 - t3
    feats['d_jun06_jun19'] = t2 - t1; feats['d_full_season'] = t4 - t1
    denom = abs(np.mean(per_date_mean)) + 1e-3
    feats['ratio_growth'] = feats['growth_jun_aug'] / denom
    feats['ratio_harvest'] = feats['harvest_decline'] / denom
    feats['ratio_early_late'] = (t1 + 1e-3) / (t4 + 1e-3)
    feats['ratio_peak_trough'] = (max(per_date_mean) + 1e-3) / (min(per_date_mean) + 1e-3)
    all_vals = np.concatenate([stack_3d[..., i][stack_3d[..., i] > 0].ravel() for i in range(4)]) \
        if n_valid > 0 else np.array([0.0])
    feats['overall_mean'] = float(np.mean(all_vals))
    feats['overall_std'] = float(np.std(all_vals))
    feats['overall_cv'] = feats['overall_std'] / (abs(feats['overall_mean']) + 1e-3)
    feats['n_valid_pixels'] = int(n_valid)
    return feats

# ── 4b. CNN + CBAM (2D) FARM PATCH CLASSIFIER ─────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Using Device: {DEVICE}")

class ChannelAttention2D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        att = self.sigmoid(self.mlp(nn.functional.adaptive_avg_pool2d(x, 1)) +
                           self.mlp(nn.functional.adaptive_max_pool2d(x, 1)))
        return x * att

class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        att = self.sigmoid(self.conv(torch.cat([torch.mean(x, dim=1, keepdim=True),
                                                torch.max(x, dim=1, keepdim=True)[0]], dim=1)))
        return x * att

class CBAM2D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channel_att = ChannelAttention2D(channels, reduction)
        self.spatial_att = SpatialAttention2D()
    def forward(self, x): return self.spatial_att(self.channel_att(x))

class FarmCNNCBAM(nn.Module):
    def __init__(self, in_channels=4, num_classes=N_CROPS, embed_dim=64):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.cbam1 = CBAM2D(32); self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.cbam2 = CBAM2D(64); self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Sequential(nn.Conv2d(64, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU())
        self.cbam3 = CBAM2D(embed_dim); self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4); self.head = nn.Linear(embed_dim, num_classes)
    def forward(self, x, return_embedding=False):
        x = self.pool1(self.cbam1(self.conv1(x)))
        x = self.pool2(self.cbam2(self.conv2(x)))
        x = self.cbam3(self.conv3(x))
        embed = self.gap(x).flatten(1)
        logits = self.head(self.dropout(embed))
        return (logits, embed) if return_embedding else logits

def train_cnn_cbam_classifier(patches, labels, n_epochs=40):
    N = len(labels)
    probs = np.full((N, N_CROPS), 1.0 / N_CROPS, dtype=np.float32)
    embeddings = np.zeros((N, 64), dtype=np.float32)
    train_mask = labels >= 0
    if train_mask.sum() < 30:
        print("  [CNN+CBAM] Too few labeled farms — skipping.")
        return probs, embeddings
    X_all = patches.copy()
    mu = X_all[train_mask].mean(axis=(0, 1, 2), keepdims=True)
    sigma = X_all[train_mask].std(axis=(0, 1, 2), keepdims=True) + 1e-3
    X_norm = (X_all - mu) / sigma
    X_tr = torch.tensor(X_norm[train_mask].transpose(0, 3, 1, 2), dtype=torch.float32)
    y_tr = torch.tensor(labels[train_mask], dtype=torch.long)
    class_counts = torch.bincount(y_tr, minlength=N_CROPS).float()
    weights = torch.where(class_counts > 0, class_counts.sum() / (class_counts * N_CROPS),
                          torch.tensor(1.0)).to(DEVICE)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True,
                        pin_memory=(DEVICE.type == 'cuda'))
    model = FarmCNNCBAM(in_channels=4, num_classes=N_CROPS).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs * max(len(loader), 1), eta_min=1e-5)
    print(f"  [CNN+CBAM] Training on {train_mask.sum()} labeled farm patches ({PATCH_SIZE}x{PATCH_SIZE}x4)...")
    model.train()
    for ep in range(n_epochs):
        for bX, by in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(bX.to(DEVICE)), by.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    model.eval()
    softmax = nn.Softmax(dim=1)
    X_full = torch.tensor(X_norm.transpose(0, 3, 1, 2), dtype=torch.float32)
    all_loader = DataLoader(TensorDataset(X_full), batch_size=256, pin_memory=(DEVICE.type == 'cuda'))
    ptr = 0
    with torch.no_grad():
        for (bX,) in all_loader:
            logits, embed = model(bX.to(DEVICE), return_embedding=True)
            b = logits.shape[0]
            probs[ptr:ptr + b] = softmax(logits).cpu().numpy()
            embeddings[ptr:ptr + b] = embed.cpu().numpy()
            ptr += b
    return probs, embeddings

def assign_weak_crop_labels_percentile(feat_df):
    labels = pd.Series(-1, index=feat_df.index, dtype=int)
    t1, t2, t3, t4 = feat_df['Jun06_mean'], feat_df['Jun19_mean'], feat_df['Aug14_mean'], feat_df['Oct13_mean']
    all_dates = feat_df[['Jun06_mean', 'Jun19_mean', 'Aug14_mean', 'Oct13_mean']]
    dyn_range = all_dates.max(axis=1) - all_dates.min(axis=1)
    growth = feat_df['growth_jun_aug']; decline = feat_df['harvest_decline']
    def q(s, p): return s.quantile(p)
    rice = (t1 < q(t1, 0.35)) & (t3 > q(t3, 0.65)) & (dyn_range > q(dyn_range, 0.5)) & (growth > q(growth, 0.55))
    labels[rice] = CROP_CLASSES.index("Rice")
    cotton_mask = (labels == -1) & (t2 > q(t2, 0.25)) & (t2 < q(t2, 0.85)) & (t3 > q(t3, 0.5)) & \
                  (dyn_range > q(dyn_range, 0.4)) & (growth > q(growth, 0.4))
    labels[cotton_mask] = CROP_CLASSES.index("Cotton")
    maize_mask = (labels == -1) & (t3 > q(t3, 0.65)) & (decline < -abs(q(decline, 0.35))) & (dyn_range > q(dyn_range, 0.7))
    labels[maize_mask] = CROP_CLASSES.index("Maize")
    bajra_mask = (labels == -1) & (t2 > q(t2, 0.5)) & (t3 > q(t3, 0.4)) & \
                 (decline < -abs(q(decline, 0.25))) & ((t3 - t1) > q(t3 - t1, 0.5))
    labels[bajra_mask] = CROP_CLASSES.index("Bajra")
    groundnut_mask = (labels == -1) & (feat_df['overall_std'] < q(feat_df['overall_std'], 0.35)) & \
                     (feat_df['overall_mean'].between(q(feat_df['overall_mean'], 0.2), q(feat_df['overall_mean'], 0.8))) & \
                     (dyn_range > q(dyn_range, 0.15)) & (dyn_range < q(dyn_range, 0.6))
    labels[groundnut_mask] = CROP_CLASSES.index("Groundnut")
    return labels

def assign_weak_crop_labels_clustering(feat_df, feature_cols, n_clusters=8):
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler as SS
    X = feat_df[feature_cols].fillna(0).values.astype(np.float32)
    col_std = X.std(axis=0)
    keep_cols = col_std > 1e-6
    if keep_cols.sum() < 2:
        cluster_id = np.zeros(len(feat_df), dtype=int)
        n_eff_clusters = 1
    else:
        X_kept = X[:, keep_cols]
        Xs = SS().fit_transform(X_kept)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        n_eff_clusters = min(n_clusters, max(2, len(feat_df) // 20))
        gmm = GaussianMixture(n_components=n_eff_clusters, covariance_type='diag',
                               reg_covar=1e-2, random_state=RANDOM_STATE, n_init=3)
        try: cluster_id = gmm.fit_predict(Xs)
        except ValueError:
            from sklearn.cluster import KMeans
            cluster_id = KMeans(n_clusters=n_eff_clusters, random_state=RANDOM_STATE, n_init=10).fit_predict(Xs)
    growth = feat_df['growth_jun_aug']; decline = feat_df['harvest_decline']
    all_dates = feat_df[['Jun06_mean', 'Jun19_mean', 'Aug14_mean', 'Oct13_mean']]
    dyn_range = all_dates.max(axis=1) - all_dates.min(axis=1)
    stability = -feat_df['overall_cv']
    def pct_rank(s): return s.rank(pct=True)
    growth_pr = pct_rank(growth); decline_pr = pct_rank(-decline)
    dyn_pr = pct_rank(dyn_range); stab_pr = pct_rank(stability)
    t1_pr = pct_rank(-feat_df['Jun06_mean'])
    cluster_scores = np.zeros((n_eff_clusters, N_CROPS))
    for c in range(n_eff_clusters):
        mask = cluster_id == c
        if mask.sum() == 0: continue
        g, d, dy, st, t1p = growth_pr[mask].mean(), decline_pr[mask].mean(), \
                             dyn_pr[mask].mean(), stab_pr[mask].mean(), t1_pr[mask].mean()
        cluster_scores[c, CROP_CLASSES.index("Rice")] = 0.4 * t1p + 0.35 * g + 0.25 * dy
        cluster_scores[c, CROP_CLASSES.index("Cotton")] = 0.4 * g + 0.35 * dy + 0.25 * (1 - abs(d - 0.5) * 2)
        cluster_scores[c, CROP_CLASSES.index("Maize")] = 0.45 * d + 0.35 * dy + 0.2 * g
        cluster_scores[c, CROP_CLASSES.index("Bajra")] = 0.4 * d + 0.3 * g + 0.3 * (1 - dy)
        cluster_scores[c, CROP_CLASSES.index("Groundnut")] = 0.5 * st + 0.3 * (1 - dy) + 0.2 * (1 - g)
    cluster_to_crop = cluster_scores.argmax(axis=1)
    pseudo_labels = pd.Series([cluster_to_crop[c] for c in cluster_id], index=feat_df.index, dtype=int)
    sorted_scores = np.sort(cluster_scores, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    pseudo_conf = pd.Series([margin[c] for c in cluster_id], index=feat_df.index)
    return pseudo_labels, pseudo_conf, cluster_id

def assign_weak_crop_labels(feat_df, feature_cols=None):
    rule_labels = assign_weak_crop_labels_percentile(feat_df)
    if feature_cols is None: return rule_labels
    cluster_labels, cluster_conf, _ = assign_weak_crop_labels_clustering(feat_df, feature_cols)
    combined = rule_labels.copy()
    fill_mask = combined == -1
    combined[fill_mask] = cluster_labels[fill_mask]
    return combined

def train_crop_classifier(feat_df, feature_cols, patches=None):
    labels = assign_weak_crop_labels(feat_df, feature_cols=feature_cols)
    train_mask = labels >= 0
    print(f"  Weak-labeled training farms: {train_mask.sum()} / {len(labels)}")
    print(f"  Label distribution: { {CROP_CLASSES[k]: int(v) for k, v in labels[train_mask].value_counts().items()} }")
    X = feat_df[feature_cols].fillna(0).values.astype(np.float32)
    y = labels.values
    probs_xgb = np.full((len(labels), N_CROPS), 1.0 / N_CROPS, dtype=np.float32)
    probs_lgb = np.full((len(labels), N_CROPS), 1.0 / N_CROPS, dtype=np.float32)
    if train_mask.sum() >= 20:
        X_tr, y_tr = X[train_mask], y[train_mask]
        xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, objective='multi:softprob',
                                     num_class=N_CROPS, eval_metric='mlogloss',
                                     random_state=RANDOM_STATE, n_jobs=-1)
        xgb_clf.fit(X_tr, y_tr)
        probs_xgb = xgb_clf.predict_proba(X)
        try:
            import lightgbm as lgb
            lgb_clf = lgb.LGBMClassifier(n_estimators=400, num_leaves=31, max_depth=7, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
                                          random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
            lgb_clf.fit(X_tr, y_tr)
            raw = lgb_clf.predict_proba(X)
            full = np.full((len(labels), N_CROPS), 1.0 / N_CROPS, dtype=np.float32)
            for i, c in enumerate(lgb_clf.classes_): full[:, c] = raw[:, i]
            probs_lgb = full
        except ImportError:
            print("  LightGBM not available — using XGBoost only.")
            probs_lgb = probs_xgb.copy()
    tree_ensemble = 0.5 * probs_xgb + 0.5 * probs_lgb
    row_sums = tree_ensemble.sum(axis=1, keepdims=True)
    tree_ensemble = tree_ensemble / np.maximum(row_sums, 1e-8)
    cnn_embeddings = np.zeros((len(labels), 64), dtype=np.float32)
    ensemble_probs = tree_ensemble
    if patches is not None:
        cnn_probs, cnn_embeddings = train_cnn_cbam_classifier(patches, y)
        n_valid_px = feat_df['n_valid_pixels'].values.astype(np.float32)
        cnn_weight = np.clip(n_valid_px / MIN_PIXELS_FOR_FULL_CNN_TRUST, 0.0, 1.0) * (1.0 / 3.0)
        tree_weight_each = (1.0 - cnn_weight) / 2.0
        ensemble_probs = (tree_weight_each[:, None] * probs_xgb + tree_weight_each[:, None] * probs_lgb +
                          cnn_weight[:, None] * cnn_probs)
        row_sums = ensemble_probs.sum(axis=1, keepdims=True)
        ensemble_probs = ensemble_probs / np.maximum(row_sums, 1e-8)
    pred_idx = ensemble_probs.argmax(axis=1)
    pred_crop = [CROP_CLASSES[i] for i in pred_idx]
    pred_conf = ensemble_probs.max(axis=1)
    return pred_crop, pred_conf, cnn_embeddings

# ── 5. CROP HEALTH INDEX (CHI) ────────────────────────────────────────────────
def compute_health_index(feat_df):
    def minmax_100(s):
        lo, hi = s.quantile(0.02), s.quantile(0.98)
        if hi <= lo: return pd.Series(50.0, index=s.index)
        return ((s.clip(lo, hi) - lo) / (hi - lo) * 100.0)
    growth_score = minmax_100(feat_df['growth_jun_aug'])
    homog_mean = feat_df[[f'{d}_glcm_homogeneity' for d in DATE_LABELS]].mean(axis=1)
    entropy_mean = feat_df[[f'{d}_glcm_entropy' for d in DATE_LABELS]].mean(axis=1)
    texture_raw = homog_mean - 0.3 * (entropy_mean / (entropy_mean.max() + 1e-6)) * homog_mean.max()
    texture_score = minmax_100(texture_raw)
    stability_score = minmax_100(-feat_df['overall_cv'])
    chi = (0.4 * growth_score + 0.3 * texture_score + 0.3 * stability_score).clip(0, 100)
    def status(v):
        if v >= 80: return "Healthy"
        if v >= 60: return "Moderate"
        if v >= 40: return "Stressed"
        return "Poor"
    return chi.round(1), chi.apply(status)

# ── 6. YIELD ESTIMATION ───────────────────────────────────────────────────────
def compute_yield_index(feat_df, chi, cnn_embeddings=None, village_id_col='village_id'):
    def norm01(s):
        lo, hi = s.quantile(0.02), s.quantile(0.98)
        if hi <= lo: return pd.Series(0.5, index=s.index)
        return ((s.clip(lo, hi) - lo) / (hi - lo))
    growth_factor = norm01(feat_df['growth_jun_aug'])
    stability_factor = norm01(-feat_df['overall_cv'])
    chi_factor = chi / 100.0
    raw_ryi = chi_factor * growth_factor * stability_factor
    lo, hi = raw_ryi.quantile(0.02), raw_ryi.quantile(0.98)
    ryi = pd.Series(50.0, index=raw_ryi.index) if hi <= lo else \
        ((raw_ryi.clip(lo, hi) - lo) / (hi - lo) * 100.0)
    yield_tha_base = pd.Series(0.0, index=feat_df.index)
    for crop, (lo_b, hi_b) in CROP_YIELD_BENCHMARKS_T_HA.items():
        mask = feat_df['crop_type'] == crop
        if mask.sum() == 0: continue
        frac = (ryi[mask] / 100.0).clip(0, 1)
        yield_tha_base[mask] = lo_b + frac * (hi_b - lo_b)
    conf = feat_df['crop_confidence'].fillna(0.5)
    for crop in CROP_YIELD_BENCHMARKS_T_HA:
        mask = feat_df['crop_type'] == crop
        if mask.sum() == 0: continue
        crop_median = yield_tha_base[mask].median()
        w = conf[mask].clip(0.2, 1.0)
        yield_tha_base[mask] = w * yield_tha_base[mask] + (1 - w) * crop_median
    yield_tha_reranked = yield_tha_base.copy()
    if cnn_embeddings is not None and cnn_embeddings.shape[0] == len(feat_df):
        from sklearn.ensemble import GradientBoostingRegressor
        crop_onehot = pd.get_dummies(feat_df['crop_type']).reindex(
            columns=list(CROP_YIELD_BENCHMARKS_T_HA.keys()), fill_value=0).values
        X_stack = np.column_stack([feat_df['health_index'].values, growth_factor.values,
                                   stability_factor.values, crop_onehot, cnn_embeddings]).astype(np.float32)
        y_target = yield_tha_base.values
        if np.std(cnn_embeddings) > 1e-6:
            gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                            subsample=0.8, random_state=RANDOM_STATE)
            gbr.fit(X_stack, y_target)
            reranked = gbr.predict(X_stack)
            yield_tha_reranked = 0.6 * yield_tha_base + 0.4 * pd.Series(reranked, index=feat_df.index)
            for crop, (lo_b, hi_b) in CROP_YIELD_BENCHMARKS_T_HA.items():
                mask = feat_df['crop_type'] == crop
                pad = 0.15 * (hi_b - lo_b)
                yield_tha_reranked[mask] = yield_tha_reranked[mask].clip(lo_b - pad, hi_b + pad)
    yield_tha_final = yield_tha_reranked.copy()
    if village_id_col in feat_df.columns:
        group_key = feat_df[village_id_col].astype(str) + "|" + feat_df['crop_type'].astype(str)
        group_median = yield_tha_reranked.groupby(group_key).transform('median')
        group_size = feat_df.groupby(group_key)[village_id_col].transform('size')
        smooth_w = np.clip((group_size - 2) / 6.0, 0.0, 1.0) * 0.35
        smooth_w = pd.Series(smooth_w.values, index=feat_df.index)
        yield_tha_final = (1 - smooth_w) * yield_tha_reranked + smooth_w * group_median
        for crop, (lo_b, hi_b) in CROP_YIELD_BENCHMARKS_T_HA.items():
            mask = feat_df['crop_type'] == crop
            pad = 0.15 * (hi_b - lo_b)
            yield_tha_final[mask] = yield_tha_final[mask].clip(lo_b - pad, hi_b + pad)
    return ryi.round(1), yield_tha_final.round(2)

def yield_expectation_label(v):
    if v > 85: return "Very High"
    if v >= 70: return "High"
    if v >= 55: return "Medium"
    return "Low"

# ── 7. VALIDATION ─────────────────────────────────────────────────────────────
def _validate_submission_format(submission, expected_rows):
    expected_cols = ['village_id', 'farm_id', 'crop_type', 'health_index', 'yield_estimate_to_date']
    if list(submission.columns) != expected_cols:
        raise ValueError(f"submission columns {list(submission.columns)} != expected {expected_cols}")
    if len(submission) != expected_rows:
        raise ValueError(f"submission has {len(submission)} rows, expected {expected_rows}")
    if not submission['farm_id'].is_unique:
        raise ValueError("submission has duplicate farm_id values")
    bad_crops = set(submission['crop_type'].unique()) - set(CROP_CLASSES)
    if bad_crops: raise ValueError(f"Bad crop values: {bad_crops}")
    if not submission['health_index'].between(0, 100).all():
        submission['health_index'] = submission['health_index'].clip(0, 100)
    if submission['yield_estimate_to_date'].isnull().any() or (submission['yield_estimate_to_date'] < 0).any():
        raise ValueError("null or negative yield_estimate_to_date values")
    print(f"  [validation] OK: {len(submission)} rows, columns match, farm_id unique.")

def _ensure_clean_farm_id(farms_gdf):
    raw = farms_gdf['farm_id']
    coerced = pd.to_numeric(raw, errors='coerce')
    if coerced.isnull().sum() == 0 and coerced.dropna().duplicated().sum() == 0:
        farms_gdf['farm_id'] = coerced.astype(int)
        return farms_gdf
    farms_gdf = farms_gdf.reset_index(drop=True)
    farms_gdf['farm_id'] = np.arange(1, len(farms_gdf) + 1)
    return farms_gdf

def _resolve_farm_id_column(farms_gdf):
    candidates = ['farm_id', 'Farm_ID', 'FARM_ID', 'FarmID', 'farmid', 'id', 'ID', 'FID']
    for c in candidates:
        if c in farms_gdf.columns: return c
    raise ValueError(f"No farm ID column found. Available: {[c for c in farms_gdf.columns if c != 'geometry']}")

# ── 8. VISUALIZATION SUITE (Data Science Plots) ───────────────────────────────
def generate_visualizations(feat_df, farms_gdf, village_gdf, village_id_col):
    sns.set_theme(style="whitegrid", font_scale=1.1)
    os.makedirs("visualizations", exist_ok=True)
    print("\n[Stage 6] Generating Data Science Visualization Suite...")

    # Merge back for spatial plotting
    viz_gdf = farms_gdf.copy()
    for col in ['crop_type', 'health_index', 'yield_estimate_to_date', 'health_status',
                'n_valid_pixels', 'yield_index_0_100']:
        if col in feat_df.columns:
            viz_gdf[col] = feat_df[col].values

    # ── Plot 1: All Farm Boundaries + Village Boundary ─────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    viz_gdf.plot(ax=ax, facecolor='#c8e6c9', edgecolor='#388e3c', linewidth=0.4, alpha=0.85)
    if village_gdf is not None:
        v_plot = village_gdf.to_crs(viz_gdf.crs) if village_gdf.crs != viz_gdf.crs else village_gdf
        v_plot.plot(ax=ax, facecolor='none', edgecolor='#212121', linewidth=2.5,
                    linestyle='--', label='Village Boundary', zorder=10)
        # Label village name
        for _, vrow in v_plot.iterrows():
            cx, cy = vrow.geometry.centroid.x, vrow.geometry.centroid.y
            vname = vrow.get('VILLAGE', vrow.get('village_id', 'Sokhda'))
            ax.annotate(str(vname), xy=(cx, cy), fontsize=12, fontweight='bold',
                        color='#212121', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7), zorder=11)
    ax.set_title(f"All Farm Boundaries — Sokhda Village ({len(viz_gdf)} Farms)",
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_axis_off()
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig("visualizations/1_all_farm_boundaries.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 1: All Farm Boundaries + Village Boundary")

    # ── Plot 2: Empty / Failed Farms Highlighted ───────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    empty_mask = viz_gdf['n_valid_pixels'].fillna(0) == 0
    valid_farms = viz_gdf[~empty_mask]
    empty_farms = viz_gdf[empty_mask]
    if len(valid_farms) > 0:
        valid_farms.plot(ax=ax, facecolor='#b0bec5', edgecolor='white', linewidth=0.3, label=f'Valid Farms ({len(valid_farms)})')
    if len(empty_farms) > 0:
        empty_farms.plot(ax=ax, facecolor='#d32f2f', edgecolor='#b71c1c', linewidth=0.8, label=f'Empty/Failed Farms ({len(empty_farms)})')
        for _, erow in empty_farms.iterrows():
            cx, cy = erow.geometry.centroid.x, erow.geometry.centroid.y
            ax.annotate(str(erow['farm_id']), xy=(cx, cy), fontsize=5, color='white', ha='center', va='center')
    if village_gdf is not None:
        v_plot = village_gdf.to_crs(viz_gdf.crs) if village_gdf.crs != viz_gdf.crs else village_gdf
        v_plot.plot(ax=ax, facecolor='none', edgecolor='#212121', linewidth=2.5, linestyle='--', zorder=10)
    ax.set_title(f"Empty/Failed Farms (No SAR Pixels Extracted) — {len(empty_farms)} of {len(viz_gdf)}",
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_axis_off()
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig("visualizations/2_empty_failed_farms.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 2: Empty/Failed Farms Map")

    # ── Plot 3: Temporal dB Backscatter Profiles by Crop ──────────────────────
    date_cols = [f"{d}_mean" for d in DATE_LABELS]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    for crop in CROP_CLASSES:
        crop_data = feat_df[feat_df['crop_type'] == crop]
        if len(crop_data) == 0: continue
        mean_profile = crop_data[date_cols].mean().values
        std_profile = crop_data[date_cols].std().values
        color = CROP_COLORS.get(crop, 'black')
        ax.plot(DATE_LABELS, mean_profile, marker='o', linewidth=2.5,
                label=f"{crop} (n={len(crop_data)})", color=color, markersize=8)
        ax.fill_between(DATE_LABELS, mean_profile - std_profile, mean_profile + std_profile,
                        alpha=0.12, color=color)
    ax.set_title("Temporal SAR Backscatter (dB) Profiles by Crop Type", fontsize=13, fontweight='bold')
    ax.set_xlabel("Acquisition Date", fontsize=11)
    ax.set_ylabel("Mean Backscatter (dB)", fontsize=11)
    ax.legend(title="Crop Type", fontsize=9)
    ax.grid(True, alpha=0.4)
    # Box plot per date per crop
    ax2 = axes[1]
    plot_data = []
    for crop in CROP_CLASSES:
        crop_data = feat_df[feat_df['crop_type'] == crop]
        for dcol, dlabel in zip(date_cols, DATE_LABELS):
            for val in crop_data[dcol].dropna().values:
                plot_data.append({'Crop': crop, 'Date': dlabel, 'dB': float(val)})
    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(data=plot_df, x='Date', y='dB', hue='Crop', palette=CROP_COLORS,
                    ax=ax2, linewidth=0.8, fliersize=2)
        ax2.set_title("Backscatter (dB) Distribution per Date per Crop", fontsize=13, fontweight='bold')
        ax2.set_xlabel("Acquisition Date", fontsize=11)
        ax2.set_ylabel("Backscatter (dB)", fontsize=11)
        ax2.legend(title="Crop Type", fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig("visualizations/3_temporal_db_profiles.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 3: Temporal dB Backscatter Profiles + Box Distribution")

    # ── Plot 4: Crop Type Classification Spatial Map ───────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    cmap_discrete = mcolors.ListedColormap([CROP_COLORS[c] for c in CROP_CLASSES])
    for i, crop in enumerate(CROP_CLASSES):
        crop_farms = viz_gdf[viz_gdf['crop_type'] == crop]
        if len(crop_farms) > 0:
            crop_farms.plot(ax=ax, color=CROP_COLORS[crop], edgecolor='white',
                            linewidth=0.2, label=f"{crop} ({len(crop_farms)})")
    if village_gdf is not None:
        v_plot = village_gdf.to_crs(viz_gdf.crs) if village_gdf.crs != viz_gdf.crs else village_gdf
        v_plot.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2.5, linestyle='--', zorder=10)
        for _, vrow in v_plot.iterrows():
            cx, cy = vrow.geometry.centroid.x, vrow.geometry.centroid.y
            vname = vrow.get('VILLAGE', 'Sokhda')
            ax.annotate(str(vname), xy=(cx, cy), fontsize=11, fontweight='bold', color='black',
                        ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), zorder=11)
    ax.set_title("Farm-Level Crop Type Classification Map\n(XGBoost + LightGBM + CNN-CBAM Ensemble)",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_axis_off()
    ax.legend(title="Crop Type", loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig("visualizations/4_crop_classification_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 4: Crop Classification Spatial Map")

    # ── Plot 5: Crop Health Index (CHI) Spatial Map ───────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    viz_gdf.plot(column='health_index', ax=ax, cmap='RdYlGn', vmin=0, vmax=100, legend=True,
                 legend_kwds={'label': 'Crop Health Index (0=Poor → 100=Excellent)',
                              'orientation': 'vertical', 'shrink': 0.7},
                 edgecolor='black', linewidth=0.2,
                 missing_kwds={"color": "lightgrey", "label": "No Data"})
    if village_gdf is not None:
        v_plot = village_gdf.to_crs(viz_gdf.crs) if village_gdf.crs != viz_gdf.crs else village_gdf
        v_plot.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2.5, linestyle='--', zorder=10)
        for _, vrow in v_plot.iterrows():
            cx, cy = vrow.geometry.centroid.x, vrow.geometry.centroid.y
            ax.annotate(f"Sokhda Village", xy=(cx, cy), fontsize=11, fontweight='bold',
                        color='black', ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), zorder=11)
    ax.set_title("Farm-Level Crop Health Index (CHI)\n(0.4 × Growth + 0.3 × Texture + 0.3 × Stability)",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("visualizations/5_health_index_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 5: Crop Health Index Spatial Map")

    # ── Plot 6: Yield Estimate by Crop (Box + Strip) ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    sns.boxplot(data=feat_df, x='crop_type', y='yield_estimate_to_date', palette=CROP_COLORS,
                ax=ax, order=CROP_CLASSES, linewidth=1.2)
    sns.stripplot(data=feat_df, x='crop_type', y='yield_estimate_to_date', color='black',
                  alpha=0.35, size=3, ax=ax, order=CROP_CLASSES, jitter=True)
    ax.set_title("Yield Estimate Distribution (t/ha) by Crop Type", fontsize=13, fontweight='bold')
    ax.set_xlabel("Crop Type", fontsize=11)
    ax.set_ylabel("Yield Estimate (t/ha)", fontsize=11)
    for crop, (lo_b, hi_b) in CROP_YIELD_BENCHMARKS_T_HA.items():
        x_pos = CROP_CLASSES.index(crop)
        ax.axhspan(lo_b, hi_b, xmin=(x_pos) / N_CROPS + 0.01, xmax=(x_pos + 1) / N_CROPS - 0.01,
                   alpha=0.08, color='green', label='_')
    # Bar chart of crop count
    ax2 = axes[1]
    crop_counts = feat_df['crop_type'].value_counts().reindex(CROP_CLASSES, fill_value=0)
    bars = ax2.bar(CROP_CLASSES, crop_counts.values,
                   color=[CROP_COLORS[c] for c in CROP_CLASSES], edgecolor='black', linewidth=0.8)
    for bar, count in zip(bars, crop_counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_title("Farm Count by Predicted Crop Type", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Crop Type", fontsize=11)
    ax2.set_ylabel("Number of Farms", fontsize=11)
    plt.tight_layout()
    plt.savefig("visualizations/6_yield_and_crop_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 6: Yield Distribution + Crop Count Bar Chart")

    # ── Plot 7: Yield Estimate Spatial Map ─────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    viz_gdf.plot(column='yield_estimate_to_date', ax=ax, cmap='YlOrRd', legend=True,
                 legend_kwds={'label': 'Yield Estimate (t/ha)', 'shrink': 0.7},
                 edgecolor='black', linewidth=0.2,
                 missing_kwds={"color": "lightgrey", "label": "No Data"})
    if village_gdf is not None:
        v_plot = village_gdf.to_crs(viz_gdf.crs) if village_gdf.crs != viz_gdf.crs else village_gdf
        v_plot.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2.5, linestyle='--', zorder=10)
        for _, vrow in v_plot.iterrows():
            cx, cy = vrow.geometry.centroid.x, vrow.geometry.centroid.y
            ax.annotate("Sokhda Village", xy=(cx, cy), fontsize=11, fontweight='bold',
                        color='black', ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), zorder=11)
    ax.set_title("Farm-Level Yield Estimate to Date (t/ha)\n(CHI × Growth Factor × Stability + Benchmark Calibration)",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("visualizations/7_yield_estimate_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 7: Yield Estimate Spatial Map")

    print(f"\n  ✅ ALL 7 PLOTS SAVED to 'visualizations/' folder!")
    return viz_gdf

# ── 9. MAIN PIPELINE ──────────────────────────────────────────────────────────
def process_farms(base_dir):
    print("=" * 70)
    print("  Round 2: Farm-Level SAR Intelligence — 966 Sokhda Farms (v4)")
    print("=" * 70)

    farm_matches = list(Path(base_dir).rglob("*[Ff]arm*.shp"))
    village_matches = list(Path(base_dir).rglob("*[Vv]illage*.shp"))
    if not farm_matches:
        raise FileNotFoundError("Sokhda_Farms.shp not found.")
    farms_gdf = gpd.read_file(farm_matches[0])
    village_gdf = gpd.read_file(village_matches[0]) if village_matches else None
    farms_gdf.geometry = farms_gdf.geometry.buffer(0)
    print(f"Loaded {len(farms_gdf)} farm polygons from {farm_matches[0].name}")

    farm_id_col = _resolve_farm_id_column(farms_gdf)
    if farm_id_col != 'farm_id':
        farms_gdf = farms_gdf.rename(columns={farm_id_col: 'farm_id'})
    farms_gdf = _ensure_clean_farm_id(farms_gdf)

    village_id_col = None
    if village_gdf is not None:
        for cand in ['village_id', 'Village_ID', 'VILLAGE_ID', 'ID', 'id']:
            if cand in village_gdf.columns:
                village_id_col = cand; break
        if village_id_col is None:
            village_id_col = [c for c in village_gdf.columns if c != 'geometry'][0]
            print(f"  [village] Using '{village_id_col}' as village ID column.")

    sar_paths, dem_path = load_sar_stack(base_dir)

    correction_surfaces = [None, None, None, None]
    if APPLY_DEM_CORRECTION and dem_path is not None:
        print("\n[Stage 1] Computing DEM terrain-flattening correction...")
        with rasterio.open(sar_paths[0]) as ref_src:
            corr = compute_local_incidence_correction(dem_path, ref_src)
        correction_surfaces = [corr, corr, corr, corr]
    else:
        print("\n[Stage 1] Terrain correction disabled — using sigma0 as-is.")

    print("\n[Stage 1] Farm -> Village spatial join...")
    if village_gdf is not None:
        farms_gdf = spatial_join_farms_to_villages(farms_gdf, village_gdf, 'farm_id', village_id_col)
        farms_gdf = _ensure_clean_village_id(farms_gdf, village_id_col)
    else:
        farms_gdf['village_id'] = 'UNKNOWN'
        village_id_col = 'village_id'

    farms_gdf = _ensure_clean_farm_id(farms_gdf)

    print(f"\n[Stage 2] Extracting features for {len(farms_gdf)} farms...")
    error_counter = {}
    with (rasterio.open(sar_paths[0]) as src1, rasterio.open(sar_paths[1]) as src2,
          rasterio.open(sar_paths[2]) as src3, rasterio.open(sar_paths[3]) as src4):
        sar_srcs = [src1, src2, src3, src4]
        if farms_gdf.crs != src1.crs: farms_gdf = farms_gdf.to_crs(src1.crs)
        feature_rows, areas_ha, patch_list = [], [], []
        for idx, row in farms_gdf.iterrows():
            arrays, pixel_area_m2 = extract_farm_arrays(row.geometry, sar_srcs, correction_surfaces, error_counter)
            stack = stack_valid_dates(arrays)
            areas_ha.append(row.geometry.area / 10000.0)
            if stack is None:
                feats = {f'{d}_mean': 0.0 for d in DATE_LABELS}
                feats.update({'growth_jun_aug': 0.0, 'harvest_decline': 0.0, 'd_jun06_jun19': 0.0,
                               'd_full_season': 0.0, 'ratio_growth': 0.0, 'ratio_harvest': 0.0,
                               'ratio_early_late': 1.0, 'ratio_peak_trough': 1.0,
                               'overall_mean': 0.0, 'overall_std': 0.0, 'overall_cv': 0.0, 'n_valid_pixels': 0})
                for d in DATE_LABELS:
                    feats[f'{d}_median'] = 0.0; feats[f'{d}_var'] = 0.0; feats[f'{d}_cv'] = 0.0
                    for tk in ['contrast', 'homogeneity', 'entropy', 'correlation']:
                        feats[f'{d}_glcm_{tk}'] = 0.0
                patch_list.append(np.zeros((PATCH_SIZE, PATCH_SIZE, 4), dtype=np.float32))
            else:
                feats = build_farm_features(stack)
                patch_list.append(resize_patch_fixed(stack, size=PATCH_SIZE))
            feature_rows.append(feats)
            if idx > 0 and idx % 200 == 0:
                print(f"    ...{idx}/{len(farms_gdf)} farms processed")

    if error_counter:
        print(f"  [extraction warnings] {error_counter}")

    patches = np.stack(patch_list, axis=0)
    feat_df = pd.DataFrame(feature_rows)
    feat_df['farm_id'] = farms_gdf['farm_id'].astype(str).values
    farms_gdf['farm_id'] = farms_gdf['farm_id'].astype(str)
    feat_df['village_id'] = farms_gdf[village_id_col].values
    feat_df['area_ha'] = areas_ha
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_df.fillna(0, inplace=True)
    exclude_cols = {'farm_id', 'village_id', 'area_ha'}
    feature_cols = [c for c in feat_df.columns if c not in exclude_cols]

    print("\n[Stage 3] Crop classification (XGBoost + LightGBM + CNN-CBAM ensemble)...")
    pred_crop, pred_conf, cnn_embeddings = train_crop_classifier(feat_df, feature_cols, patches=patches)
    feat_df['crop_type'] = pred_crop
    feat_df['crop_confidence'] = pred_conf.round(3)

    print("\n[Stage 4] Computing Crop Health Index (CHI)...")
    chi, status = compute_health_index(feat_df)
    feat_df['health_index'] = chi
    feat_df['health_status'] = status

    print("[Stage 5] Computing Yield Estimate...")
    ryi, yield_tha = compute_yield_index(feat_df, chi, cnn_embeddings=cnn_embeddings, village_id_col='village_id')
    feat_df['yield_index_0_100'] = ryi
    feat_df['yield_estimate_to_date'] = yield_tha
    feat_df['yield_expectation'] = ryi.apply(yield_expectation_label)

    submission = feat_df[['village_id', 'farm_id', 'crop_type', 'health_index', 'yield_estimate_to_date']].copy()
    submission['village_id'] = submission['village_id'].astype(str)
    submission['farm_id'] = pd.to_numeric(submission['farm_id'], errors='coerce').fillna(0).astype(int)
    submission['crop_type'] = submission['crop_type'].astype(str)
    submission['health_index'] = submission['health_index'].round(0).astype(int)
    submission['yield_estimate_to_date'] = submission['yield_estimate_to_date'].astype(float)
    submission = submission.sort_values('farm_id').reset_index(drop=True)
    _validate_submission_format(submission, expected_rows=len(farms_gdf))
    submission.to_csv('submission.csv', index=False)
    print("✅ Saved submission.csv")

    # Full detailed report
    df_final = feat_df[['farm_id', 'village_id', 'crop_type', 'crop_confidence',
                         'health_index', 'health_status', 'yield_index_0_100',
                         'yield_estimate_to_date', 'yield_expectation', 'area_ha']]
    df_final.to_csv('Farm_Intelligence_Report.csv', index=False)
    print("✅ Saved Farm_Intelligence_Report.csv")

    # ── Stage 6: Visualizations ────────────────────────────────────────────────
    generate_visualizations(feat_df, farms_gdf, village_gdf, village_id_col)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — Sample Output:")
    print("=" * 70)
    print(df_final.head(10).to_string(index=False))
    print("=" * 70)
    print(f"\nCrop Distribution:\n{feat_df['crop_type'].value_counts().to_string()}")
    print(f"\nHealth Status Distribution:\n{feat_df['health_status'].value_counts().to_string()}")
    return submission

# ── RUN ───────────────────────────────────────────────────────────────────────
final_submission = process_farms(DATA_PATH)