import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone


def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def synth_day(ts_start: datetime, points: int = 360, step_seconds: int = 240,
              base_latency: float = 10.0, base_throughput: float = 100.0,
              noise: float = 0.1, spike_at: int | None = None) -> pd.DataFrame:
    # Generate a simple time series for one day with optional latency spike
    rows = []
    t = ts_start
    for i in range(points):
        lat = base_latency * (1 + noise * np.random.randn())
        thr = base_throughput * (1 + noise * np.random.randn())
        if spike_at is not None and i == spike_at:
            lat *= 5.0  # big latency spike to mimic fail-slow
            thr *= 0.5  # throughput dip
        rows.append({
            'ts': int(t.replace(tzinfo=timezone.utc).timestamp()),
            'throughput': max(0.0, thr),
            'latency': max(0.0, lat),
            'disk_id': 'disk1',
        })
        t += timedelta(seconds=step_seconds)
    return pd.DataFrame(rows)


def upsert_all_drive_info(index_dir: Path, cluster_hosts: dict[str, list[str]]):
    all_info = index_dir / 'all_drive_info.csv'
    if all_info.exists():
        df = pd.read_csv(all_info)
    else:
        df = pd.DataFrame(columns=['cluster', 'host_name'])
    # Append new rows that are not already present
    existing = set(zip(df['cluster'].astype(str), df['host_name'].astype(str))) if not df.empty else set()
    new_rows = []
    for cluster, hosts in cluster_hosts.items():
        for h in hosts:
            key = (cluster, h)
            if key not in existing:
                new_rows.append({'cluster': cluster, 'host_name': h})
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(all_info, index=False)
        print(f"Updated {all_info} with {len(new_rows)} new rows.")
    else:
        print(f"No changes to {all_info}.")


def main():
    p = argparse.ArgumentParser(description='Synthesize minimal Perseus-like per-host time-series under a directory.')
    p.add_argument('-p', '--perseus_dir', default='index', help='Root directory to create cluster_*/host_*/YYYY-MM-DD.csv under')
    p.add_argument('--clusters', default='cluster_A,cluster_B', help='Comma-separated cluster names to create')
    p.add_argument('--hosts-per-cluster', type=int, default=1, help='Number of hosts to create per cluster')
    p.add_argument('--days', type=int, default=1, help='Number of days per host')
    p.add_argument('--start-date', default=datetime.utcnow().strftime('%Y-%m-%d'), help='YYYY-MM-DD start date')
    p.add_argument('--augment-index', action='store_true', help='Append created hosts to index/all_drive_info.csv')
    args = p.parse_args()

    root = Path(args.perseus_dir)
    ensure_dirs(root)

    clusters = [c.strip() for c in args.clusters.split(',') if c.strip()]
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')

    rng = np.random.default_rng(42)

    cluster_hosts: dict[str, list[str]] = {}

    for c in clusters:
        hosts = []
        for i in range(1, args.hosts_per_cluster + 1):
            host = f"host_synth{i}"
            hosts.append(host)
            host_dir = root / c / host
            ensure_dirs(host_dir)
            for d in range(args.days):
                day = (start_date + timedelta(days=d)).strftime('%Y-%m-%d')
                # Place a spike in the first day's middle to emulate anomaly for the first host
                spike_idx = None
                if i == 1 and d == 0:
                    spike_idx = 100  # somewhere mid-series
                df = synth_day(
                    ts_start=datetime.strptime(day, '%Y-%m-%d'),
                    points=360,
                    step_seconds=240,
                    base_latency=10.0 + 2.0 * rng.standard_normal(),
                    base_throughput=100.0 + 10.0 * rng.standard_normal(),
                    noise=0.15,
                    spike_at=spike_idx,
                )
                out_path = host_dir / f"{day}.csv"
                df.to_csv(out_path, index=False)
        cluster_hosts[c] = hosts

    print("Created clusters and hosts:")
    for c, hs in cluster_hosts.items():
        print(f"  {c}: {', '.join(hs)}")

    # Optionally update index/all_drive_info.csv
    if args.augment_index:
        index_dir = root if (root / 'all_drive_info.csv').exists() or root.name == 'index' else Path('index')
        index_dir.mkdir(exist_ok=True)
        upsert_all_drive_info(index_dir, cluster_hosts)

    print("Done. You can now run per-host models (csr, lstm, patchTST, hybrid).")


if __name__ == '__main__':
    main()
