import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

LARGE_NUMBER_OF_DAYS = 100

def get_snapshot(df, stats=True):
    """
    Calculate snapshot statistics or retrieve the first row for each group in the data frame.

    Args:
        df (pd.DataFrame): The input data frame.
        stats (bool): If True, calculate snapshot statistics. If False, retrieve the first row for each group (default: True).

    Returns:
        pd.DataFrame: The snapshot statistics or first row data frame.
    """
    if stats:
        snapshot = df.reset_index().groupby(['host', 'disk_id']).agg({
            'throughput': ['mean', 'min', 'std'],
            'latency': ['mean', 'max', 'std']
        })
        snapshot.columns = ['_'.join(col).strip() for col in snapshot.columns.values]
    else:
        snapshot = df.reset_index().groupby(['host', 'disk_id'], as_index=False).first()
    
    return snapshot.reset_index()

def calculate_thresholds(df, col_name, sigma=3):
    """
    Calculate thresholds based on mean and standard deviation of a column.

    Args:
        df (pd.DataFrame): The input data frame.
        col_name (str): The name of the column to calculate thresholds for.
        sigma (int): The number of standard deviations to use for threshold calculation (default: 3).

    Returns:
        float: The calculated threshold value.
    """
    mean = df[col_name].mean()
    std = df[col_name].std()
    if col_name == 'throughput':
        return max(0, mean - sigma * std)
    else:
        return mean + sigma * std

def get_time_until_err_perseus(df, err_label_name, elapsed_time_label_name):
    """
    Specifically designed for the Perseus project, calculate the time from the start of observation to the first error.

    Args:
        df (pd.DataFrame): The input data frame.
        err_label_name (str): Column name indicating the error label.
        elapsed_time_label_name (str): Column name for the calculated time until the error.

    Returns:
        pd.DataFrame: Data frame containing the elapsed time until the first error, adjusted for the Perseus project.
    """
    first_error_obs_time = df[df[err_label_name] == 1].reset_index().groupby(['host', 'disk_id'])['time_utc'].min()
    first_obs_time = df.reset_index().groupby(['host', 'disk_id'])['time_utc'].min()
    res = (first_error_obs_time - first_obs_time).to_frame()
    res = res.rename(columns={'time_utc': elapsed_time_label_name})
    res[elapsed_time_label_name] = res[elapsed_time_label_name].dt.total_seconds()
    res = res[elapsed_time_label_name].fillna(LARGE_NUMBER_OF_DAYS * 24 * 3600)
    return res.to_frame()

def get_faulty_disks_perseus(predictions, y_test, M):
    """
    Identify faulty disks for the Perseus project based on prediction rankings.

    Args:
        predictions (pd.Series): Series of prediction scores or probabilities.
        y_test (pd.DataFrame): The test data set including the disk identifiers.
        M (int): The number of top-ranked disks to identify as faulty.

    Returns:
        list: List of tuples (host, disk_id) of the top M ranked faulty disks based on the threshold rank.
    """
    df_tmp = y_test.copy()
    df_tmp['rank'] = predictions
    df_tmp.sort_values('rank', inplace=True, ascending=True)
    
    threshold_rank = df_tmp['rank'].iloc[M]
    faulty_disks = df_tmp[df_tmp['rank'] <= threshold_rank].reset_index()[['host', 'disk_id']]
    
    return list(faulty_disks.apply(lambda x: (x['host'], x['disk_id']), axis=1))

def generate_label_perseus(df, col_name, label_name, lookback='3min', threshold=110, percentage=0.5):
    """
    Generate labels based on specified conditions for Perseus data.

    Args:
        df (pd.DataFrame): The input data frame.
        col_name (str): The name of the column to generate labels for.
        label_name (str): The name of the generated label column.
        lookback (str): The lookback window size (default: '3min').
        threshold (int): The threshold value for the warning condition (default: 110).
        percentage (float): The percentage threshold for labeling (default: 0.5).

    Returns:
        pd.DataFrame: The data frame with generated labels.
    """
    col_warn = f'{col_name}_warn'
    threshold = calculate_thresholds(df, col_name)
    if col_name == 'throughput':
        df[col_warn] = df[col_name] < threshold
    else:
        df[col_warn] = df[col_name] > threshold
    
    labels = df.reset_index(level=['host', 'disk_id']).groupby(['host', 'disk_id'])[col_warn]
    rolling = labels.rolling(lookback, closed='right', min_periods=1)
    rolling_percentage = rolling.apply(lambda x: x.sum() / len(x))
    
    res = rolling_percentage.apply(lambda x: int(x > percentage)).fillna(0).to_frame()
    res.rename(columns={col_warn: label_name}, inplace=True)
    res.reset_index(inplace=True)
    res.set_index(['time_utc', 'host', 'disk_id'], inplace=True)
    res.sort_index(inplace=True)
    df.drop(col_warn, axis=1, inplace=True)
    return res

def eval_model_Perseus(df, date_str, err_df, cluster):
    elapsed_time_label_name = 'time_until_err'
    ground_truth_name_throughput = 'throughput_truth_labels'
    ground_truth_name_latency = 'latency_truth_labels'
    ground_truth_label_name = 'ground_truth'

    df[ground_truth_name_throughput] = generate_label_perseus(df, 'throughput', ground_truth_name_throughput)
    df[ground_truth_name_latency] = generate_label_perseus(df, 'latency', ground_truth_name_latency)
    df[ground_truth_label_name] = np.maximum(df[ground_truth_name_throughput], df[ground_truth_name_latency])

    df_train, df_test = train_test_split(df, test_size=0.5, random_state=41)

    X_train = get_snapshot(df_train).drop(columns=['host', 'disk_id'])
    X_test = get_snapshot(df_test).drop(columns=['host', 'disk_id'])
    y_train = get_time_until_err_perseus(df_train, ground_truth_label_name, elapsed_time_label_name)
    y_test = get_time_until_err_perseus(df_test, ground_truth_label_name, elapsed_time_label_name)

    N = len(X_train)
    M = sum(y_train['time_until_err'] < LARGE_NUMBER_OF_DAYS * 24 * 3600)
    assert len(X_train) == len(y_train) == N
    assert len(X_test) == len(y_test)

    if len(X_test) == 0:
        return

    train_groups = [N]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(train_groups)
    params = {
        'objective': 'rank:pairwise',
        'learning_rate': 0.2,
        'max_depth': 8,
    }
    num_boost_round = 1000
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    test_groups = [N]
    dtest = xgb.DMatrix(X_test, y_test)
    dtest.set_group(test_groups)
    predictions = bst.predict(dtest)

    faulty_disks = get_faulty_disks_perseus(predictions, y_test, M)
    return faulty_disks

def generate_dframe_perseus(filename, host_name):
    """
    Generate a data frame from a Parquet file for Perseus data.

    Args:
        filename (str): The path to the Parquet file.
        host_name (str): The name of the host.

    Returns:
        pd.DataFrame: The generated data frame.

    Raises:
        Exception: If the file is not found or empty.
    """
    try:
        df = pd.read_csv(filename)
        if len(df) == 0:
            raise Exception('Empty file')
        df.loc[:, "time_utc"] = pd.to_datetime(df.loc[:, "ts"].values, unit="s").tz_localize("UTC")
        df['time_utc'] = pd.to_datetime(df['time_utc'])
        for col in ['throughput', 'latency']:
            df[col] = pd.to_numeric(df[col])
        df['host'] = host_name
        df.set_index(['time_utc', 'host', 'disk_id'], inplace=True)
        df.sort_index(inplace=True)
        df.drop(['ts'], axis='columns', inplace=True)
        return df
    except FileNotFoundError:
        raise Exception(f"The file was not found.")

import time

def process_Perseus(perseus_dir, cluster_host_mapping, err_df):
    """
    Processes and evaluates disk health data for each host within a cluster based on a mapping dictionary.

    Parameters:
        cluster_host_mapping (dict): Dictionary mapping clusters to lists of hosts.
        err_df (DataFrame): DataFrame containing error data for model evaluation.

    Processes each host's data within the cluster, evaluates using Perseus, and aggregates metrics.
    Prints the status and relevant metrics of disks identified as faulty.
    """
    cluster_times = {}
    total_time = 0
    num_clusters = len(cluster_host_mapping)

    for cluster, hosts in cluster_host_mapping.items():
        print(f"Processing cluster {cluster} with hosts {', '.join(hosts)}")

        start_time = time.time()

        date_dfs = defaultdict(list)
        for host in hosts:
            cluster_path = os.path.join(perseus_dir, f"{cluster}", host)
            if not os.path.isdir(cluster_path):
                print(f"Skipping host '{host}': directory not found at {cluster_path}")
                continue
            for filename in os.listdir(cluster_path):
                if filename.endswith(".csv") and len(filename) == 14:
                    file_path = os.path.join(cluster_path, filename)
                    try:
                        df = generate_dframe_perseus(file_path, host)
                        date_str = os.path.splitext(filename)[0][:10]
                        date_dfs[date_str].append(df)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        sorted_dates = sorted(date_dfs.keys())

        for date_str in sorted_dates:
            dfs = date_dfs[date_str]
            df = pd.concat(dfs)
            faulty_disks = eval_model_Perseus(df, date_str, err_df, cluster)
            if faulty_disks:
                for host, disk_id in df.index.droplevel('time_utc').unique():
                    status = "T" if (host, disk_id) in faulty_disks else "F"
                    print(f"{date_str}, {cluster}, {host}, {disk_id}, {status}")
            else:
                for host, disk_id in df.index.droplevel('time_utc').unique():
                    print(f"{date_str}, {cluster}, {host}, {disk_id}, F")

        end_time = time.time()
        cluster_time = end_time - start_time
        cluster_times[cluster] = cluster_time
        total_time += cluster_time

    print("\nCluster processing times:")
    for cluster, time_taken in cluster_times.items():
        print(f"Cluster {cluster}: {time_taken:.2f} seconds")

    avg_time = total_time / num_clusters
    print(f"\nAverage processing time per cluster: {avg_time:.2f} seconds")

def main(perseus_dir, input_file):
    all_drive_info_path = input_file
    err_path = f"{perseus_dir}/slow_drive_info.csv"
    # Fallback: if not under perseus_dir, try alongside the input_file
    if not os.path.exists(err_path):
        candidate = os.path.join(os.path.dirname(all_drive_info_path), 'slow_drive_info.csv')
        if os.path.exists(candidate):
            err_path = candidate

    if not os.path.exists(all_drive_info_path) or not os.path.exists(err_path):
        print("Required files not found in the specified directory.")
        return

    df = pd.read_csv(all_drive_info_path)
    cluster_host_mapping = df.groupby('cluster')['host_name'].apply(list).to_dict()
    err_df = pd.read_csv(err_path)
    process_Perseus(perseus_dir, cluster_host_mapping, err_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Perseus data.')
    parser.add_argument('-p', '--perseus_dir', required=True, help="Path to the directory containing Perseus data files")
    parser.add_argument('-i', '--input_file', required=True, help="Path to the drive file")
    args = parser.parse_args()

    main(args.perseus_dir, args.input_file)