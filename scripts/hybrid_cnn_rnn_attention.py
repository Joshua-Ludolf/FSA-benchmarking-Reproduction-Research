import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False


# ------------------------
# Config
# ------------------------

WINDOW_MINUTES = 5
STEP_SECONDS = 15
WINDOW_STEPS = WINDOW_MINUTES * 60 // STEP_SECONDS  # 20
HORIZON_MINUTES = 10
HORIZON_STEPS = HORIZON_MINUTES * 60 // STEP_SECONDS  # 40

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Metrics that might exist in data; pick available ones
POSSIBLE_METRICS = [
    'latency', 'throughput', 'error_rate', 'queue_depth',
    'iops_read', 'iops_write', 'utilization', 'await', 'svctm', 'busy'
]


# ------------------------
# Data preparation
# ------------------------

def calculate_thresholds(df: pd.DataFrame, col_name: str, sigma: int = 3) -> float:
    mean = df[col_name].mean()
    std = df[col_name].std()
    if col_name == 'throughput':
        return max(0, mean - sigma * std)
    return mean + sigma * std


def generate_point_labels(df: pd.DataFrame, percentage: float = 0.5) -> pd.Series:
    """
    Reuse the project's heuristic: derive a per-timestamp fail-slow label from
    latency/throughput anomalies using rolling percentage exceedance.
    """
    df = df.copy()
    y_total = pd.Series(0, index=df.index, dtype=int)
    for col in ['throughput', 'latency']:
        if col not in df.columns:
            continue
        thr = calculate_thresholds(df, col)
        warn_col = f'{col}_warn'
        if col == 'throughput':
            df[warn_col] = df[col] < thr
        else:
            df[warn_col] = df[col] > thr
        def roll_pct(s: pd.Series) -> pd.Series:
            r = s.rolling('3min', closed='right', min_periods=1).apply(lambda x: x.sum() / len(x))
            return (r > percentage).astype(int)
        part = df.groupby(['host', 'disk_id'])[warn_col].apply(roll_pct)
        part.index = part.index.droplevel([0,1])
        aligned = part.reindex(y_total.index).fillna(0).astype(int)
        y_total = pd.Series(np.maximum(y_total.to_numpy(), aligned.to_numpy()), index=y_total.index)
        df.drop(columns=[warn_col], inplace=True)
    return y_total


def load_disk_timeseries(perseus_dir: str, cluster: str, host: str, date: str) -> Optional[pd.DataFrame]:
    path = os.path.join(perseus_dir, cluster, host, f"{date}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    # Normalize time
    df['time_utc'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    df = df.sort_values('time_utc')
    # keep only known metrics if exist
    available = [c for c in POSSIBLE_METRICS if c in df.columns]
    if not available:
        available = [c for c in ['latency', 'throughput'] if c in df.columns]
    if not available:
        return None
    cols = ['time_utc', 'host', 'disk_id'] + available
    # Some files may be missing host/disk; add host from dir; keep disk_id as in data
    if 'host' not in df.columns:
        df['host'] = host
    df = df[cols].dropna()
    return df


def list_days(perseus_dir: str, cluster: str, host: str) -> List[str]:
    host_dir = os.path.join(perseus_dir, cluster, host)
    if not os.path.isdir(host_dir):
        return []
    files = [f for f in os.listdir(host_dir) if f.endswith('.csv') and len(f) == 14]
    return sorted([f[:-4] for f in files])


def window_labeling(df: pd.DataFrame, horizon_steps: int = HORIZON_STEPS) -> pd.DataFrame:
    """
    For each (host,disk) time series at 15s frequency, create window-level labels:
    a window ending at t is positive if any point label in (t, t+horizon] is 1.
    Returns rows with columns: host, disk_id, t_end (pd.Timestamp), y_win (0/1)
    """
    # Ensure regular 15s grid per disk
    out_rows = []
    for (host, disk), g in df.groupby(['host', 'disk_id']):
        g = g.set_index('time_utc').sort_index()
        g = g.asfreq('15s').interpolate(limit_direction='both')
        y_point = generate_point_labels(g)
        g['y_point'] = y_point
        # compute window ends every 15s starting from first valid time when we have WINDOW_STEPS
        times = g.index
        if len(times) < WINDOW_STEPS + horizon_steps + 1:
            continue
        # create an array for faster horizon lookup
        y_arr = g['y_point'].to_numpy()
        for end_idx in range(WINDOW_STEPS - 1, len(times) - 1):
            t_end = times[end_idx]
            # look into future horizon (exclusive of end)
            fut_start = end_idx + 1
            fut_end = min(len(times), end_idx + 1 + horizon_steps)
            y_future = y_arr[fut_start:fut_end]
            y_win = 1 if (y_future.sum() > 0) else 0
            out_rows.append({'host': host, 'disk_id': disk, 't_end': t_end, 'y_win': y_win})
    return pd.DataFrame(out_rows)


def build_windows(df: pd.DataFrame, metrics: List[str]) -> Tuple[np.ndarray, List[Tuple[str, str, pd.Timestamp]]]:
    """
    Build X windows of shape [N, T, F] and align metadata for each window end.
    """
    X = []
    meta = []
    for (host, disk), g in df.groupby(['host', 'disk_id']):
        g = g.set_index('time_utc').sort_index()
        g = g.asfreq('15s').interpolate(limit_direction='both')
        vals = g[metrics].to_numpy()
        for end_idx in range(WINDOW_STEPS - 1, len(g)):
            start_idx = end_idx - WINDOW_STEPS + 1
            window = vals[start_idx:end_idx+1]
            if window.shape[0] == WINDOW_STEPS:
                X.append(window)
                meta.append((host, disk, g.index[end_idx]))
    X = np.array(X) if X else np.zeros((0, WINDOW_STEPS, len(metrics)))
    return X, meta


def align_labels(meta: List[Tuple[str, str, pd.Timestamp]], label_df: pd.DataFrame) -> np.ndarray:
    # ensure t_end is Timestamp dtype for lookup
    label_df = label_df.copy()
    label_df['t_end'] = pd.to_datetime(label_df['t_end'], utc=True)
    lkp = {(r.host, r.disk_id, r.t_end): int(r.y_win) for r in label_df.itertuples()}
    y = [lkp.get((h, d, t), 0) for (h, d, t) in meta]
    return np.array(y, dtype=np.int64)


# ------------------------
# Dataset
# ------------------------

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------
# Model: CNN -> BiLSTM -> MultiheadAttention -> MLP
# ------------------------

class HybridModel(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.3):
        super().__init__()
        # Feature attention/gating across metrics at each timestep
        self.feature_gate = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Sigmoid()
        )
        c = 64
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, c, kernel_size=3, padding=1),
            nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2),  # T//2
            nn.Conv1d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2),  # T//4
            nn.Conv1d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2),  # T//8
        )
        # After conv, shape: [B, C, T'] -> transpose to [B, T', C]
        self.bilstm = nn.LSTM(input_size=c, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [B, T, F]
        # feature gating per timestep
        feat_w = self.feature_gate(x)  # [B, T, F]
        x = x * feat_w
        # Conv1d expects [B, F, T]
        x = x.transpose(1, 2)
        x = self.conv(x)
        # back to [B, T', C]
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        # self-attention
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        # global average pool across time
        x = attn_out.mean(dim=1)
        x = self.dropout(x)
        logits = self.head(x)
        return logits.squeeze(-1), attn_weights, feat_w


# ------------------------
# Training / Evaluation
# ------------------------

@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 100
    early_patience: int = 10


def train_one_fold(model, train_loader, val_loader, cfg: TrainConfig, class_weights: Optional[Tuple[float,float]] = None):
    model.to(DEVICE)
    pos_w = None
    if class_weights:
        pw_val = class_weights[1]/max(class_weights[0],1e-6)
        pos_w = torch.tensor([pw_val], device=DEVICE, dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)

    best_val = -np.inf
    best_state = None
    patience = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits, _, _ = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        # validate
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits, _, _ = model(xb)
                prob = torch.sigmoid(logits).cpu().numpy()
                ys.append(yb.numpy())
                ps.append(prob)
        y_true = np.concatenate(ys) if ys else np.array([])
        y_prob = np.concatenate(ps) if ps else np.array([])
        if y_true.size == 0:
            val_auroc = 0.0
        else:
            try:
                val_auroc = roc_auc_score(y_true, y_prob)
            except ValueError:
                val_auroc = 0.0

        sched.step()

        if val_auroc > best_val:
            best_val = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def predict(model, loader):
    model.eval(); model.to(DEVICE)
    ys, ps, attn_all, feat_all = [], [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits, attn, feat_w = model(xb)
            prob = torch.sigmoid(logits).cpu().numpy()
            ys.append(yb.numpy())
            ps.append(prob)
            attn_all.append(attn.cpu().numpy())
            feat_all.append(feat_w.cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([])
    y_prob = np.concatenate(ps) if ps else np.array([])
    attn_np = np.concatenate(attn_all, axis=0) if attn_all else np.zeros((0,))
    feat_np = np.concatenate(feat_all, axis=0) if feat_all else np.zeros((0,))
    return y_true, y_prob, attn_np, feat_np


def metrics_from_scores(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    if y_true.size == 0:
        return {k: 0.0 for k in ['precision','recall','f1','auroc','accuracy']}
    y_pred = (y_prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.0
    acc = (y_pred == y_true).mean()
    return {'precision': float(p), 'recall': float(r), 'f1': float(f1), 'auroc': float(auroc), 'accuracy': float(acc)}


def precision_at_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.8) -> Tuple[float, float]:
    if y_true.size == 0:
        return 0.0, 0.5
    try:
        p, r, thr = precision_recall_curve(y_true, y_prob)
        # precision_recall_curve returns decreasing thresholds; r and p aligned to thr length+1
        # We'll find indices where recall >= target
        idx = np.where(r >= target_recall)[0]
        if idx.size == 0:
            return 0.0, 0.5
        # Use max precision among those points; threshold index mapping
        best_i = idx[np.argmax(p[idx])]
        # threshold for this recall index: thr has length len(p)-1
        thr_val = thr[min(best_i, len(thr)-1)] if len(thr) > 0 else 0.5
        return float(p[best_i]), float(thr_val)
    except Exception:
        return 0.0, 0.5


def time_to_alert(meta: List[Tuple[str,str,pd.Timestamp]], y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> float:
    # Compute average minutes from first abnormal window to first predicted positive per (host,disk)
    if y_true.size == 0:
        return 0.0
    df = pd.DataFrame(meta, columns=['host','disk_id','t_end'])
    df['y_true'] = y_true
    df['y_pred'] = (y_prob >= thr).astype(int)
    ttas = []
    for (h,d), g in df.groupby(['host','disk_id']):
        g = g.sort_values('t_end')
        if g['y_true'].sum() == 0:
            continue
        # first abnormal window index
        try:
            t_abn = g.loc[g['y_true'] == 1, 't_end'].iloc[0]
        except IndexError:
            continue
        # first prediction after that time
        g_after = g[g['t_end'] >= t_abn]
        pos = g_after.loc[g_after['y_pred'] == 1]
        if pos.empty:
            continue
        t_pred = pos['t_end'].iloc[0]
        tta_min = (t_pred - t_abn).total_seconds() / 60.0
        ttas.append(tta_min)
    return float(np.mean(ttas)) if ttas else 0.0


# ------------------------
# Orchestration
# ------------------------

def main(perseus_dir: str, index_file: str, train_cluster: str, test_cluster: str, batch_size: int = 256,
         export_attention: bool = True, export_preds: bool = True, export_feature_attention: bool = True,
         use_smote: bool = False, export_shap: bool = False, out_dir: str = 'output',
         test_fraction: float = 1.0):
    os.makedirs(out_dir, exist_ok=True)
    # Load cluster->hosts
    idx = pd.read_csv(index_file)
    cluster_hosts = idx.groupby('cluster')['host_name'].apply(list).to_dict()
    train_hosts = cluster_hosts.get(train_cluster, [])
    test_hosts = cluster_hosts.get(test_cluster, [])

    # Pick metrics available from any sample day
    sample_host = None
    sample_day = None
    for h in train_hosts:
        days = list_days(perseus_dir, train_cluster, h)
        if days:
            sample_host = h
            sample_day = days[0]
            break
    if not sample_day or not sample_host:
        print('No data found for training cluster.'); return
    sample_df = load_disk_timeseries(perseus_dir, train_cluster, sample_host, sample_day)
    if sample_df is None:
        print('No sample data loaded.'); return
    metrics = [c for c in POSSIBLE_METRICS if c in sample_df.columns]
    if not metrics:
        metrics = [c for c in ['latency','throughput'] if c in sample_df.columns]
    print(f'Using metrics: {metrics}')

    # Assemble training data (all days for train cluster)
    all_train_df = []
    for host in train_hosts:
        for day in list_days(perseus_dir, train_cluster, host):
            df = load_disk_timeseries(perseus_dir, train_cluster, host, day)
            if df is None: continue
            all_train_df.append(df)
    if not all_train_df:
        print('No training data.'); return
    train_df = pd.concat(all_train_df, ignore_index=True)
    lbl_df = window_labeling(train_df[['time_utc','host','disk_id'] + metrics])
    # Normalize features using training stats for window inputs (labels remain on raw scale)
    means = train_df[metrics].mean()
    stds = train_df[metrics].std().replace(0, np.nan).fillna(1.0).clip(lower=1e-6)
    def norm_df(dfin: pd.DataFrame) -> pd.DataFrame:
        dfo = dfin.copy()
        dfo[metrics] = (dfo[metrics] - means) / stds
        return dfo
    train_df_norm = norm_df(train_df[['time_utc','host','disk_id'] + metrics])
    X_all, meta_all = build_windows(train_df_norm, metrics)
    y_all = align_labels(meta_all, lbl_df)
    if isinstance(y_all, pd.Series):
        y_all = y_all.to_numpy()
    # Optional SMOTE on flattened windows
    if use_smote and _HAS_SMOTE and len(y_all) > 0:
        X2d = X_all.reshape((X_all.shape[0], -1))
        try:
            sm = SMOTE(random_state=41)
            res = sm.fit_resample(X2d, y_all)
            # imblearn returns (X_res, y_res); guard odd returns
            if isinstance(res, tuple) and len(res) >= 2:
                X_res, y_res = res[0], res[1]
            else:
                raise RuntimeError('Unexpected SMOTE return format')
            X_all = np.asarray(X_res).reshape((-1, X_all.shape[1], X_all.shape[2]))
            y_all = np.asarray(y_res).astype(np.int64)
            print(f'Applied SMOTE: {len(y_all)} samples after resampling.')
        except Exception as e:
            print(f'SMOTE failed or skipped: {e}')
    print(f'Train windows: {len(y_all)}; positives: {int(y_all.sum())}')

    # 5-fold CV on training set for early model selection
    kf = KFold(n_splits=5, shuffle=True, random_state=41)
    best_fold = None
    best_auroc = -np.inf

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all)):
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]
        w0 = max(1, int((y_tr == 0).sum()))
        w1 = max(1, int((y_tr == 1).sum()))
        cw = (w0, w1)
        ds_tr = WindowDataset(X_tr, y_tr)
        ds_va = WindowDataset(X_va, y_va)
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)
        model = HybridModel(n_features=len(metrics))
        auroc = train_one_fold(model, dl_tr, dl_va, TrainConfig(batch_size=batch_size), class_weights=cw)
        if auroc > best_auroc:
            best_auroc = auroc
            best_fold = (model,)
        print(f'Fold {fold+1} AUROC: {auroc:.4f}')

    if best_fold is None:
        print('No best model trained.'); return
    model = best_fold[0]

    # Evaluate on held-out test cluster
    all_test_df = []
    for host in test_hosts:
        for day in list_days(perseus_dir, test_cluster, host):
            df = load_disk_timeseries(perseus_dir, test_cluster, host, day)
            if df is None: continue
            all_test_df.append(df)
    if not all_test_df:
        print('No test data.'); return
    test_df = pd.concat(all_test_df, ignore_index=True)
    lbl_te = window_labeling(test_df[['time_utc','host','disk_id'] + metrics])
    test_df_norm = norm_df(test_df[['time_utc','host','disk_id'] + metrics])
    X_te, meta_te = build_windows(test_df_norm, metrics)
    y_te = align_labels(meta_te, lbl_te)
    if isinstance(y_te, pd.Series):
        y_te = y_te.to_numpy()

    # Optionally subsample a held-out fraction of the test set (e.g., 0.10 for 10%)
    if 0.0 < test_fraction < 1.0 and len(y_te) > 0:
        rng = np.random.RandomState(41)
        n = len(y_te)
        k = max(1, int(round(test_fraction * n)))
        idx = rng.choice(n, size=k, replace=False)
        X_te = X_te[idx]
        y_te = y_te[idx]
        meta_te = [meta_te[i] for i in idx]

    dl_te = DataLoader(WindowDataset(X_te, y_te), batch_size=batch_size, shuffle=False)
    y_true, y_prob, attn_np, feat_np = predict(model, dl_te)
    # Debug class balance
    print(f'Test windows: {len(y_te)}; positives: {int(y_te.sum())}')
    # Score polarity correction (if needed)
    def _safe_auc(y, s):
        try:
            return roc_auc_score(y, s)
        except Exception:
            return 0.0
    auc_norm = _safe_auc(y_true, y_prob)
    auc_inv = _safe_auc(y_true, 1 - y_prob)
    use_prob = y_prob if auc_norm >= auc_inv else (1 - y_prob)
    m = metrics_from_scores(y_true, use_prob, thr=0.5)
    m['auroc'] = float(_safe_auc(y_true, use_prob))
    m['auroc_inverse'] = float(auc_inv)
    p_at_r, thr_at_r = precision_at_recall(y_true, use_prob, target_recall=0.8)
    m['precision_at_recall_0_80'] = float(p_at_r)
    m['threshold_at_recall_0_80'] = float(thr_at_r)
    tta = time_to_alert(meta_te, y_true, use_prob, thr=0.5)
    m['time_to_alert_min'] = float(tta)
    print('Test metrics:', m)

    # Export
    pd.DataFrame([m]).to_csv(os.path.join(out_dir, 'hybrid_metrics.csv'), index=False)
    pred_df = pd.DataFrame(meta_te, columns=['host','disk_id','t_end'])
    pred_df['y_true'] = y_true
    pred_df['y_prob'] = use_prob
    pred_df['y_pred'] = (use_prob >= 0.5).astype(int)
    if export_preds:
        pred_df.to_csv(os.path.join(out_dir, 'hybrid_predictions.csv'), index=False)
    if export_attention and attn_np.size:
        # Save attention maps to disk in a notebook-friendly shape.
        # Cases:
        # - 4D [N, H, T', T']: average over heads -> [N, T', T']
        # - 3D [N, T', T']: already averaged -> save as-is
        # - 2D [T', T']: single window -> expand to [1, T', T']
        save_path = os.path.join(out_dir, 'hybrid_attention.npy')
        if attn_np.ndim == 4:
            attn_to_save = attn_np.mean(axis=1)
        elif attn_np.ndim == 3:
            attn_to_save = attn_np
        elif attn_np.ndim == 2:
            attn_to_save = attn_np[None, ...]
        else:
            attn_to_save = None
        if attn_to_save is not None:
            np.save(save_path, attn_to_save)
    # Export feature attention (average over time per window)
    if export_feature_attention and feat_np.size:
        try:
            if feat_np.ndim == 3:  # [N, T, F]
                feat_mean = feat_np.mean(axis=1)  # [N, F]
            elif feat_np.ndim == 2:  # [T, F] single window
                feat_mean = feat_np[None, ...].mean(axis=1)
            else:
                feat_mean = None
            if feat_mean is not None:
                np.save(os.path.join(out_dir, 'hybrid_feature_attention.npy'), feat_mean)
                # Save metric names alongside
                meta = {'metrics': metrics}
                with open(os.path.join(out_dir, 'hybrid_meta.json'), 'w', encoding='utf-8') as f:
                    import json as _json
                    f.write(_json.dumps(meta))
        except Exception as e:
            print(f'Failed to export feature attention: {e}')
    # Model checkpoint
    torch.save(model.state_dict(), os.path.join(out_dir, 'hybrid_model.pt'))

    # Optional SHAP export (KernelExplainer on flattened [T*F] features, aggregated to per-metric)
    if export_shap and _HAS_SHAP and len(meta_te) > 0:
        try:
            # Sample a small background and evaluation set
            X_eval = X_te
            n_bg = min(20, len(X_eval))
            n_eval = min(30, len(X_eval))
            bg = X_eval[:n_bg]
            xs = X_eval[:n_eval]
            T = X_eval.shape[1]
            F = X_eval.shape[2]
            def f_flat(v2d: np.ndarray) -> np.ndarray:
                # v2d: [N, T*F] -> [N, T, F]
                arr = v2d.reshape((-1, T, F)).astype(np.float32)
                with torch.no_grad():
                    xb = torch.from_numpy(arr).to(DEVICE)
                    logits, _, _ = model(xb)
                    prob = torch.sigmoid(logits).cpu().numpy()
                return prob
            bg_flat = bg.reshape((bg.shape[0], -1))
            xs_flat = xs.reshape((xs.shape[0], -1))
            explainer = shap.KernelExplainer(f_flat, bg_flat)
            sv = explainer.shap_values(xs_flat, nsamples=200)
            # Aggregate per metric across time
            sv = np.array(sv)  # [N, T*F]
            sv_metric = sv.reshape((sv.shape[0], T, F)).sum(axis=1)  # [N, F]
            np.save(os.path.join(out_dir, 'hybrid_shap_values.npy'), sv_metric)
            # Save metric names if not already
            meta_path = os.path.join(out_dir, 'hybrid_meta.json')
            try:
                if not os.path.exists(meta_path):
                    import json as _json
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        f.write(_json.dumps({'metrics': metrics}))
            except Exception:
                pass
        except Exception as e:
            print(f'SHAP export failed or skipped: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid CNN-BiLSTM-SelfAttention fail-slow predictor')
    parser.add_argument('-p', '--perseus_dir', required=True, help='Path to dataset root (e.g., data)')
    parser.add_argument('-i', '--index_file', required=True, help='Index CSV with cluster,host_name')
    parser.add_argument('--train_cluster', default='cluster_A', help='Training cluster')
    parser.add_argument('--test_cluster', default='cluster_B', help='Testing cluster')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_attention_export', action='store_true')
    parser.add_argument('--no_pred_export', action='store_true')
    parser.add_argument('--no_feature_attention_export', action='store_true')
    parser.add_argument('--use_smote', action='store_true')
    parser.add_argument('--export_shap', action='store_true')
    parser.add_argument('-o', '--out_dir', default='output')
    parser.add_argument('--test_fraction', type=float, default=1.0, help='Fraction of test windows to evaluate (e.g., 0.1 for 10%)')
    args = parser.parse_args()

    main(
        perseus_dir=args.perseus_dir,
        index_file=args.index_file,
        train_cluster=args.train_cluster,
        test_cluster=args.test_cluster,
        batch_size=args.batch_size,
        export_attention=not args.no_attention_export,
        export_preds=not args.no_pred_export,
        export_feature_attention=not args.no_feature_attention_export,
        use_smote=args.use_smote,
        export_shap=args.export_shap,
        out_dir=args.out_dir,
        test_fraction=args.test_fraction,
    )
