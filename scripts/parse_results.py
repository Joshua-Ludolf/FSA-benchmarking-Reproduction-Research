import os
import re
import csv
import argparse
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List


def load_ground_truth(gt_path: str) -> set:
    df = pd.read_csv(gt_path)
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    cluster_col = cols.get('cluster', 'cluster')
    host_col = cols.get('host_name', 'host_name')
    disk_col = cols.get('disk_id', 'disk_id')
    truth = set()
    for _, row in df.iterrows():
        cluster = str(row[cluster_col]).strip()
        host = str(row[host_col]).strip()
        disk = str(row[disk_col]).strip()
        if cluster and host and disk and disk != 'nan':
            truth.add((cluster, host, disk))
    return truth


def parse_line(line: str) -> Optional[Tuple[str, str, str, str, Optional[float], Optional[bool]]]:
    # Expected formats:
    # 1) date, cluster, host, disk, T/F
    # 2) date, cluster, host, disk, <score>, T/F
    parts = [p.strip() for p in line.strip().split(',')]
    if len(parts) < 5:
        return None
    date, cluster, host, disk = parts[0:4]
    score = None
    pred = None
    if parts[-1] in ('T', 'F'):
        pred = parts[-1] == 'T'
        if len(parts) > 5:
            try:
                score = float(parts[-2])
            except ValueError:
                score = None
    else:
        # Sometimes there may be inconsistent whitespace; try regex
        m = re.search(r'(,\s*)(T|F)\s*$', line)
        if m:
            pred = m.group(2) == 'T'
    return date, cluster, host, disk, score, pred


def parse_text_output(file_path: str) -> pd.DataFrame:
    rows: List[dict] = []
    # PowerShell Tee-Object often writes UTF-16 LE by default; try multiple encodings
    encodings_to_try = ['utf-8-sig', 'utf-16', 'utf-16-le', 'cp1252']
    content = None
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
            if content and content.strip():
                break
        except Exception:
            continue
    if content is None:
        return pd.DataFrame()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = parse_line(line)
        if not parsed:
            continue
        date, cluster, host, disk, score, pred = parsed
        if pred is None:
            continue
        rows.append({
            'date': date,
            'cluster': cluster,
            'host': host,
            'disk': disk,
            'score': score,
            'pred': pred,
        })
    return pd.DataFrame(rows)


def parse_csv_output(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    # Normalize columns
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == 'date':
            rename_map[c] = 'date'
        elif cl == 'cluster':
            rename_map[c] = 'cluster'
        elif cl in ('host', 'host_name'):
            rename_map[c] = 'host'
        elif cl in ('disk', 'disk_id'):
            rename_map[c] = 'disk'
        elif cl in ('prediction', 'is_faulty', 'pred'):
            rename_map[c] = 'pred'
        elif cl in ('probability', 'anomaly_score', 'mse', 'score'):
            rename_map[c] = 'score'
    df = df.rename(columns=rename_map)
    # Booleans may be strings
    if 'pred' in df.columns:
        df['pred'] = df['pred'].map(lambda x: True if str(x).strip().upper() in ('TRUE', 'T', '1') else False)
    elif 'score' in df.columns:
        # Derive prediction by thresholding score when labels are absent
        try:
            df['pred'] = pd.to_numeric(df['score'], errors='coerce') >= 0.5
        except Exception:
            df['pred'] = False
    else:
        # No way to infer predictions; return empty to signal skip
        return pd.DataFrame()
    # Ensure required columns exist
    for col in ['date', 'cluster', 'host', 'disk']:
        if col not in df.columns:
            df[col] = ''
    if 'score' not in df.columns:
        df['score'] = None
    return df[['date', 'cluster', 'host', 'disk', 'score', 'pred']]


def evaluate(df: pd.DataFrame, truth: set) -> dict:
    # truth is set of (cluster, host_name, disk_id)
    if df.empty:
        return {'n': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}
    y_true = []
    y_pred = []
    for _, r in df.iterrows():
        key = (str(r['cluster']).strip(), str(r['host']).strip(), str(r['disk']).strip())
        y_true.append(key in truth)
        y_pred.append(bool(r['pred']))
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {'n': len(y_true), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}


def main(output_dir: str, truth_path: str, summary_out: str):
    truth = load_ground_truth(truth_path)

    results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse text outputs (*.out)
    for out_file in sorted(output_path.glob('*.out')):
        df = parse_text_output(str(out_file))
        metrics = evaluate(df, truth)
        model = out_file.stem
        results.append({'model': model, **metrics})
        # Save normalized predictions per model
        df.to_csv(output_path / f'{model}_parsed.csv', index=False)

    # Parse CSV outputs (e.g., iforest_output.csv, svm_output_*.csv)
    for csv_file in sorted(output_path.glob('*.csv')):
        # Skip normalized ones we generated
        if csv_file.name.endswith('_parsed.csv'):
            continue
        df = parse_csv_output(str(csv_file))
        if df.empty:
            # Skip files that don't contain recognizable prediction columns
            continue
        metrics = evaluate(df, truth)
        model = csv_file.stem
        results.append({'model': model, **metrics})
        df.to_csv(output_path / f'{model}_parsed.csv', index=False)

    # Write summary CSV
    summary_csv = Path(summary_out)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(summary_csv, index=False)

    # Also write a simple markdown
    md_lines = ["# Metrics Summary", "", "| Model | N | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        md_lines.append(f"| {r['model']} | {r['n']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['accuracy']:.3f} |")
    (summary_csv.parent / 'metrics_summary.md').write_text('\n'.join(md_lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse model outputs and compute metrics against ground truth')
    parser.add_argument('-o', '--output_dir', default='output', help='Directory containing model outputs')
    parser.add_argument('-g', '--ground_truth', default=os.path.join('index', 'slow_drive_info.csv'), help='Path to ground truth CSV')
    parser.add_argument('-s', '--summary_out', default=os.path.join('output', 'metrics_summary.csv'), help='Path to write summary CSV')
    args = parser.parse_args()
    main(args.output_dir, args.ground_truth, args.summary_out)
