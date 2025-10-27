import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = Path('output')
METRICS_CSV = OUT_DIR / 'hybrid_metrics.csv'
META_JSON = OUT_DIR / 'hybrid_meta.json'
FEAT_ATT_NPY = OUT_DIR / 'hybrid_feature_attention.npy'
ATTN_NPY = OUT_DIR / 'hybrid_attention.npy'
SUMMARY_CSV = OUT_DIR / 'metrics_summary.csv'
RESULTS_MD = OUT_DIR / 'hybrid_results.md'
RESULTS_TEX = OUT_DIR / 'hybrid_results.tex'


def load_hybrid_metrics() -> dict:
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Missing {METRICS_CSV}. Run the hybrid model first.")
    df = pd.read_csv(METRICS_CSV)
    return df.iloc[0].to_dict()


def load_meta() -> dict:
    if META_JSON.exists():
        with open(META_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def plot_feature_attention(metrics: list[str]) -> Path | None:
    if not FEAT_ATT_NPY.exists():
        return None
    arr = np.load(FEAT_ATT_NPY)
    # arr shape: [N, F]; average across windows
    if arr.ndim == 1:
        mean_att = arr
    else:
        mean_att = arr.mean(axis=0)
    fig_path = OUT_DIR / 'hybrid_feature_attention.png'
    plt.figure(figsize=(6, 3.2), dpi=150)
    sns.barplot(x=metrics, y=mean_att, color='#4C72B0')
    plt.ylabel('Average feature gate (0–1)')
    plt.xlabel('Metric')
    plt.title('Hybrid: average feature attention per metric')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def plot_attention_heatmap() -> Path | None:
    if not ATTN_NPY.exists():
        return None
    arr = np.load(ATTN_NPY)
    # arr possible shapes: [N, T', T'] or [N, H, T', T'] already averaged in saver
    if arr.ndim == 4:
        # [N, H, T', T'] -> average heads and samples
        heat = arr.mean(axis=(0, 1))
    elif arr.ndim == 3:
        heat = arr.mean(axis=0)
    elif arr.ndim == 2:
        heat = arr
    else:
        return None
    fig_path = OUT_DIR / 'hybrid_attention_heatmap.png'
    plt.figure(figsize=(4.5, 4), dpi=150)
    sns.heatmap(heat, cmap='viridis')
    plt.title('Hybrid self-attention (avg over windows)')
    plt.xlabel('Time index')
    plt.ylabel('Time index')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def write_markdown(metrics: dict, feat_fig: Path | None, attn_fig: Path | None, baselines: pd.DataFrame | None):
    lines = []
    lines.append('# Hybrid Model Results')
    lines.append('')
    lines.append('Window-level evaluation on the held-out test cluster (A→B):')
    lines.append('')
    lines.append('| Metric | Value |')
    lines.append('|---|---:|')
    for k in ['precision','recall','f1','auroc','accuracy','precision_at_recall_0_80','time_to_alert_min']:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, (int, float)):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")
    lines.append('')
    if baselines is not None and not baselines.empty:
        lines.append('Baseline disk-level metrics (from metrics_summary.csv, not directly comparable to window-level):')
        lines.append('')
        lines.append('| Model | N | Precision | Recall | F1 | Accuracy |')
        lines.append('|---|---:|---:|---:|---:|---:|')
        for _, r in baselines.iterrows():
            lines.append(f"| {r['model']} | {int(r['n'])} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['accuracy']:.3f} |")
        lines.append('')
        lines.append('Note: Hybrid metrics are window-level; summary baselines are disk-level from parse_results and are included for context only.')
        lines.append('')
    if feat_fig:
        lines.append(f"![Feature attention]({feat_fig.name})")
        lines.append('')
    if attn_fig:
        lines.append(f"![Self-attention heatmap]({attn_fig.name})")
        lines.append('')
    RESULTS_MD.write_text('\n'.join(lines), encoding='utf-8')


def write_latex(metrics: dict):
    # A compact LaTeX table for inclusion
    lines = []
    lines.append('% Hybrid window-level results (A→B)')
    lines.append('\\begin{table}[h]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append('  \\begin{tabular}{lrrrrrr}')
    lines.append('    \\toprule')
    lines.append('    Metric & Precision & Recall & F1 & AUROC & Acc. & Prec.@R=0.8 \\\\')
    lines.append('    \\midrule')
    p = metrics.get('precision', 0.0)
    r = metrics.get('recall', 0.0)
    f1 = metrics.get('f1', 0.0)
    auc = metrics.get('auroc', 0.0)
    acc = metrics.get('accuracy', 0.0)
    par = metrics.get('precision_at_recall_0_80', 0.0)
    # End row with LaTeX line break \\ (escaped as \\\\ in Python string)
    lines.append(f"    Hybrid & {p:.3f} & {r:.3f} & {f1:.3f} & {auc:.3f} & {acc:.3f} & {par:.3f} \\\\ ")
    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('  \\caption{Hybrid CNN–BiLSTM–Self-Attention window-level results on test cluster (train A, test B).}')
    lines.append('  \\label{tab:hybrid_results}')
    lines.append('\\end{table}')
    RESULTS_TEX.write_text('\n'.join(lines), encoding='utf-8')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = load_hybrid_metrics()
    meta = load_meta()
    metrics = meta.get('metrics', [])
    feat_fig = plot_feature_attention(metrics) if metrics else None
    attn_fig = plot_attention_heatmap()
    baselines = None
    if SUMMARY_CSV.exists():
        try:
            baselines = pd.read_csv(SUMMARY_CSV)
        except Exception:
            baselines = None
    write_markdown(m, feat_fig, attn_fig, baselines)
    write_latex(m)
    print(f"Wrote {RESULTS_MD} and {RESULTS_TEX}.")
    if feat_fig:
        print(f"Saved feature attention figure: {feat_fig}")
    if attn_fig:
        print(f"Saved self-attention heatmap: {attn_fig}")


if __name__ == '__main__':
    main()
