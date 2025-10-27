import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import time

SEED = 41
FEATURES = ['throughput', 'latency']
PERCENTAGE_THRESHOLD = 0.5

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

def get_faulty_disks_perseus(y_pred, y_test):
    df_tmp = y_test.to_frame()
    df_tmp['prediction'] = y_pred

    disk_ids = df_tmp[df_tmp['prediction'] == 1].reset_index()[['host', 'disk_id']]
    return list(disk_ids.apply(lambda x: (x['host'], x['disk_id']), axis=1))


def eval_model_Perseus_multi_pred(df, classifier, features, err_df, cluster, date_str, scale=False, window_freq='15min', lookback='1min', percentage=0.5, seed=41):
    df = df.copy()
    # Generating ground truth
    ground_truth_name_throughput = 'throughput_truth_labels'
    ground_truth_name_latency = 'latency_truth_labels'
    ground_truth_label_name = 'ground_truth'

    df[ground_truth_name_throughput] = generate_label_perseus(
        df, 'throughput', ground_truth_name_throughput)
    df[ground_truth_name_latency] = generate_label_perseus(
        df, 'latency', ground_truth_name_latency)
    df[ground_truth_label_name] = np.maximum(
        df[ground_truth_name_throughput], df[ground_truth_name_latency])

    # Generate feature and labels data
    tmpdf = df.reset_index(level=['host', 'disk_id'], drop=False)
    tmpdf = tmpdf.dropna(subset=['throughput', 'latency'] + [ground_truth_label_name])
    feature_data = tmpdf.groupby(
        ['host', 'disk_id', pd.Grouper(freq=window_freq)])[['throughput', 'latency']].first()
    label_data = tmpdf.groupby(
        ['host', 'disk_id', pd.Grouper(freq=window_freq)])[ground_truth_label_name].sum()
    label_data = label_data.to_frame()
    label_data[ground_truth_label_name] = label_data[ground_truth_label_name].apply(lambda x: 1 if x >= 1 else 0)

    # Train
    X = feature_data
    y = label_data[ground_truth_label_name]
    if len(set(y)) == 1:
        return

    if len(X) == 0 or len(y) == 0:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed)
    if len(set(y_train)) == 1:
        print(f'Single class after split. No result')
        return

    # Check the number of samples in each class
    class_counts = y_train.value_counts()
    if class_counts.min() < 6:  # 6 is the default n_neighbors in SMOTE
        return

    if scale == True:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    sm = SMOTE(random_state=seed)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    classifier.fit(X_train_res, y_train_res)

    # Predict on the test set
    y_pred = classifier.predict(X_test)

    # Evaluate results
    faulty_disks = get_faulty_disks_perseus(y_pred, y_test)

    return faulty_disks

def process_Perseus_multi_pred(perseus_dir, cluster_host_mapping, err_df):
    cluster_times = {}
    total_time = 0
    num_clusters = len(cluster_host_mapping)

    for cluster, hosts in cluster_host_mapping.items():
        start_time = time.time()
        print(f"Processing cluster {cluster} with hosts {', '.join(hosts)}")

        date_dfs = defaultdict(list)
        for host in hosts:
            cluster_path = os.path.join(perseus_dir, cluster, host)
            if not os.path.isdir(cluster_path):
                print(f"Skipping host '{host}': directory not found at {cluster_path}")
                continue
            for filename in os.listdir(cluster_path):
                if filename.endswith(".csv") and len(filename) == 14:
                    file_path = os.path.join(cluster_path, filename)
                    try:
                        df = generate_dframe_perseus(file_path, host)  # Pass the host name
                        date_str = os.path.splitext(filename)[0][:10]
                        date_dfs[date_str].append(df)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        # Sort the dates in ascending order
        sorted_dates = sorted(date_dfs.keys())

        for date_str in sorted_dates:
            dfs = date_dfs[date_str]
            df = pd.concat(dfs)
            clf = RandomForestClassifier(n_estimators=100, random_state=SEED)

            # Check if dataframe has enough samples
            if len(df) > 1:
                faulty_disks = eval_model_Perseus_multi_pred(df, clf, FEATURES, err_df, cluster, date_str)
                if faulty_disks:
                    for host, disk_id in df.index.droplevel('time_utc').unique():
                        status = "T" if (host, disk_id) in faulty_disks else "F"
                        print(f"{date_str}, {cluster}, {host}, {disk_id}, {status}")
                else:
                    for host, disk_id in df.index.droplevel('time_utc').unique():
                        print(f"{date_str}, {cluster}, {host}, {disk_id}, F")
            else:
                print(f"Insufficient data for date: {date_str}")

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
    process_Perseus_multi_pred(perseus_dir, cluster_host_mapping, err_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Perseus data.')
    parser.add_argument('-p', '--perseus_dir', required=True, help="Path to the directory containing Perseus data files")
    parser.add_argument('-i', '--input_file', required=True, help="Path to the drive file")
    args = parser.parse_args()

    main(args.perseus_dir, args.input_file)