import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import argparse
import time

SEQ_LENGTH = 8
LATENCY_ONLY = True
RANDOM_SEED = 42

def load_host(data_path, host_name):
    # Load and concatenate all CSV files for a specific host
    df_ret = pd.DataFrame()
    dir_path = f"{data_path}/{host_name}"
    for filename in os.listdir(dir_path):
        if len(filename) == 14 and filename.endswith('.csv'):
            file_path = os.path.join(dir_path, filename)
            df_tmp = pd.read_csv(file_path)
            df_ret = pd.concat([df_ret, df_tmp])
    df_ret.reset_index(inplace=True)
    df_ret['ts'] = pd.to_datetime(df_ret['ts'], unit='s')
    df_ret.set_index('ts', inplace=True)
    df_ret.drop('index', axis=1, inplace=True)
    df_ret = df_ret.dropna()
    return df_ret

def load_host_by_days(data_path, host_name):
    # Load data for each host and organize by day
    day_data = {}
    dir_path = f"{data_path}/{host_name}"
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if len(filename) == 14 and filename.endswith('.csv'):
            file_path = os.path.join(dir_path, filename)
            df_tmp = pd.read_csv(file_path)
            df_tmp.reset_index(inplace=True)
            df_tmp['ts'] = pd.to_datetime(df_tmp['ts'], unit='s')
            df_tmp.set_index('ts', inplace=True)
            df_tmp.drop('index', axis=1, inplace=True)
            df_tmp = df_tmp.dropna()
            date = filename.split('.')[0]
            day_data[date] = df_tmp
    return day_data

def df_to_sequences(df, disk_id, seq_length=SEQ_LENGTH, latency_only=LATENCY_ONLY):
    # Convert DataFrame to sequences for a specific disk
    df_disk = df[df['disk_id'] == disk_id]
    vs = np.asarray(df_disk['latency']) if latency_only else np.asarray(df_disk[['latency', 'throughput']])
    if len(vs) < seq_length:
        return np.array([]), np.array([])
    Xs = np.array([vs[i:(i + seq_length)] for i in range(len(vs) - seq_length)])
    ys = np.array(vs[seq_length:len(vs)])
    if latency_only:
        Xs = np.reshape(Xs, (Xs.shape[0], Xs.shape[1], 1))
        ys = np.reshape(ys, (-1, 1))
    return Xs, ys

def host_to_sequences(data_path, host_name, seq_length=SEQ_LENGTH, latency_only=LATENCY_ONLY):
    # Generate sequences for all disks in a host
    df_host = load_host(data_path, host_name)
    disk_ids = df_host['disk_id'].unique()
    X, y = [], []
    for id in disk_ids:
        X_tmp, y_tmp = df_to_sequences(df_host, id, seq_length=seq_length, latency_only=latency_only)
        if X_tmp.size > 0:
            X.extend(X_tmp)
            y.extend(y_tmp)
    return np.array(X), np.array(y)

def create_training_vectors(data_path, hosts, latency_only, seq_length):
    # Create training vectors from multiple hosts
    X, y = [], []
    for host in hosts:
        X_tmp, y_tmp = host_to_sequences(data_path, host, seq_length=seq_length, latency_only=latency_only)
        if X_tmp.size > 0:
            X.extend(X_tmp)
            y.extend(y_tmp)
    return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    # LSTM model definition
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

def train(perseus_dir, cluster_host_mapping, train_cluster):
    # Train the LSTM model and return scalers and model
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    TRAIN_HOSTS = cluster_host_mapping[train_cluster]
    X, y = create_training_vectors(f"{perseus_dir}/{train_cluster}", TRAIN_HOSTS, latency_only=True, seq_length=8)
    # Scale the features and labels
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).reshape(y.shape)
    X_train, y_train = X_scaled, y_scaled
    np.random.seed(RANDOM_SEED)
    # Randomly sample data for training
    indicator = np.random.binomial(1, 80000/len(X_train), len(X_train)).astype(bool)
    X_train = X_train[indicator]
    y_train = y_train[indicator]
    X_train_tensors = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensors = torch.tensor(y_train, dtype=torch.float32)
    model = LSTMModel(1, 100, 1, 2)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.005)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    num_epochs = 200
    total_start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        X_batch = X_train_tensors.to(device)
        y_batch = y_train_tensors.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f'Training one cluster took {total_time:.2f} seconds')
    return model, scaler_X, scaler_y

def eval(cluster_host_mapping, perseus_dir, model, scaler_X, scaler_y):
    # Evaluate model on test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    disk_ids = [f"disk{i}" for i in range(1, 13)]
    
    for cluster, hosts in cluster_host_mapping.items():
        day_data_all_hosts = {}
        
        # Load data for each host and organize by day
        for host in hosts:
            days_data = load_host_by_days(f"{perseus_dir}/{cluster}", host)
            for day, df in days_data.items():
                if day not in day_data_all_hosts:
                    day_data_all_hosts[day] = []
                day_data_all_hosts[day].append((host, df))
        
        # Evaluate each day separately
        for day, data_for_day in day_data_all_hosts.items():
            cluster_mse_list = []
            for host, df_test in data_for_day:
                for disk_id in disk_ids:
                    # Generate sequences for testing
                    X_test, y_test = df_to_sequences(df_test, disk_id=disk_id, latency_only=True, seq_length=SEQ_LENGTH)
                    if X_test.size == 0 or y_test.size == 0:
                        cluster_mse_list.append((day, cluster, host, disk_id, 0, False))
                        continue
                    # Scale the test data
                    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
                    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
                    X_test_tensors = torch.tensor(X_test_scaled, dtype=torch.float32)
                    model.eval()
                    with torch.no_grad():
                        predictions_scaled = model(X_test_tensors.to(device)).cpu().numpy()
                    predictions = scaler_y.inverse_transform(predictions_scaled)
                    mse = mean_squared_error(y_test, predictions)
                    cluster_mse_list.append((day, cluster, host, disk_id, mse, None))
        
            cluster_mse_df = pd.DataFrame(cluster_mse_list, columns=['day', 'cluster', 'host', 'disk_id', 'mse', 'is_faulty'])
            if not cluster_mse_df.empty:
                # Calculate threshold to determine faulty disks
                mse_mean = cluster_mse_df['mse'][cluster_mse_df['is_faulty'].isnull()].mean()
                mse_std = cluster_mse_df['mse'][cluster_mse_df['is_faulty'].isnull()].std()
                threshold = mse_mean + 3 * mse_std
                cluster_mse_df.loc[cluster_mse_df['is_faulty'].isnull(), 'is_faulty'] = cluster_mse_df['mse'] > threshold
                cluster_mse_df['is_faulty'] = cluster_mse_df['is_faulty'].map({True: "T", False: "F"})
                
                for _, row in cluster_mse_df.iterrows():
                    print(f"{row['day']}, {row['cluster']}, {row['host']}, {row['disk_id']}, {row['mse']}, {row['is_faulty']}")

def main(perseus_dir, input_file, train_cluster):
    if not os.path.exists(input_file):
        print("Required files not found in the specified directory.")
        return
    df = pd.read_csv(input_file)
    cluster_host_mapping = df.groupby('cluster')['host_name'].apply(list).to_dict()
    model, scaler_X, scaler_y = train(perseus_dir, cluster_host_mapping, train_cluster)
    eval(cluster_host_mapping, perseus_dir, model, scaler_X, scaler_y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Perseus data.')
    parser.add_argument('-p', '--perseus_dir', required=True, help="Path to the directory containing Perseus data files")
    parser.add_argument('-i', '--input_file', required=True, help="Path to the drive file")
    parser.add_argument('-t', '--train_cluster', required=True, help="Cluster to use for training")
    args = parser.parse_args()
    main(args.perseus_dir, args.input_file, args.train_cluster)
