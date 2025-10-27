import os
import pandas as pd
import argparse
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

class IsolationForestClassifier:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
        self.scaler = StandardScaler()

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.model.fitted_ = True  # Custom attribute to check if model is fitted

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def decision_function(self, X):
        X_scaled = self.scaler.transform(X)
        # Higher scores indicate normal points; more negative = more anomalous
        return self.model.decision_function(X_scaled)

def load_data(base_path, index_file):
    file_path = os.path.join(base_path, index_file)
    print(f"Loading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded data with columns: {data.columns.tolist()}")

        # Rename columns
        data.rename(columns={'cluster': 'Cluster', 'host_name': 'Host'}, inplace=True)
        print(f"Renamed columns: {data.columns.tolist()}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"File path attempted: {file_path}")
        print(f"Current working directory: {os.getcwd()}")
        raise
    return data

def generate_iforest_response(aggregated_data, model_processor=None):
    print(f"Processing data with {len(aggregated_data)} records")
    
    if len(aggregated_data) == 0:
        print(f"No data to process")
        return pd.DataFrame(), model_processor
    
    # Ensure the necessary columns are present
    required_columns = ['Cluster', 'Host']
    missing_columns = [col for col in required_columns if col not in aggregated_data.columns]
    if missing_columns:
        print(f"Error: The following required columns are missing in the dataframe: {missing_columns}")
        return pd.DataFrame(), model_processor
    
    print(f"Columns in aggregated_data before processing:\n{aggregated_data.columns.tolist()}")
    print(f"First few rows in aggregated_data before processing:\n{aggregated_data.head()}")
    
    if model_processor is None:
        model_processor = IsolationForestClassifier()
        print(f"Initialized new model processor")
    
    features = []
    cluster_host_pairs = []
    for (cluster, host_name), group in aggregated_data.groupby(['Cluster', 'Host']):
        feature_vector = [
            group['mean'].values[0],
            group['std'].values[0],
            group['min'].values[0],
            group['max'].values[0],
            group['median'].values[0],
            group['skew'].values[0] if 'skew' in group else 0,
            group['q25'].values[0],
            group['q75'].values[0]
        ]
        features.append(feature_vector)
        cluster_host_pairs.append((cluster, host_name))
    
    features = pd.DataFrame(features, columns=[
        'mean', 'std', 'min', 'max', 'median', 'skew', 'q25', 'q75'
    ])
    
    print(f"Feature matrix shape: {features.shape}")
    
    if not hasattr(model_processor.model, 'fitted_'):
        model_processor.fit(features)
        print(f"Fitted new model")
    
    predictions = model_processor.predict(features)
    decisions = model_processor.decision_function(features)
    # Convert to anomaly_score where higher means more anomalous
    anomaly_scores = -decisions
    # Normalize anomaly score to [0,1] for convenience (robust to constant values)
    min_s, max_s = anomaly_scores.min(), anomaly_scores.max()
    if max_s - min_s > 1e-12:
        norm_scores = (anomaly_scores - min_s) / (max_s - min_s)
    else:
        norm_scores = np.zeros_like(anomaly_scores)
    
    results = []
    for idx, ((cluster, host_name), pred) in enumerate(zip(cluster_host_pairs, predictions)):
        result = {
            'Date': pd.to_datetime('today').strftime('%Y-%m-%d'),
            'Cluster': cluster,
            'Host': host_name,
            'Disk': 'disk1',
            'Prediction': pred == -1,
            'anomaly_score': float(anomaly_scores[idx]),
            'score': float(norm_scores[idx])
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    print(f"Generated {results_df['Prediction'].sum()} anomalies")
    print(f"Results DataFrame columns: {results_df.columns.tolist()}")
    print(f"First few rows of the results:\n{results_df.head()}")
    
    results_df.to_csv(f'output/iforest_output.csv', index=False)
    return results_df, model_processor

def main(base_path, index_file):
    data = load_data(base_path, index_file)
    
    # Add synthetic numeric data for testing
    data['mean'] = np.random.rand(len(data))
    data['std'] = np.random.rand(len(data))
    data['min'] = np.random.rand(len(data))
    data['max'] = np.random.rand(len(data))
    data['median'] = np.random.rand(len(data))
    data['skew'] = np.random.rand(len(data))
    data['q25'] = np.random.rand(len(data))
    data['q75'] = np.random.rand(len(data))

    # Ensure necessary columns are present in data
    print(f"Data columns before ensuring necessary columns: {data.columns.tolist()}")
    print(f"First few rows of the data before processing:\n{data.head()}")

    # Rename 'cluster' and 'host_name' to 'Cluster' and 'Host'
    data.rename(columns={'cluster': 'Cluster', 'host_name': 'Host'}, inplace=True)
    
    data['Disk'] = 'disk1'
    
    print(f"Data columns after ensuring necessary columns: {data.columns.tolist()}")
    print(f"First few rows of the data:\n{data.head()}")
    
    aggregated_data = data  # Replace with actual processing to get aggregated_data
    
    results, model_processor = generate_iforest_response(aggregated_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Isolation Forest for anomaly detection.')
    parser.add_argument('-p', '--path', required=True, help="Base path to the data directory")
    parser.add_argument('-i', '--index_file', required=True, help="Index file within the data directory")
    args = parser.parse_args()

    print(f"Script is being run with the following base path: {args.path} and index file: {args.index_file}")
    print(f"Current working directory: {os.getcwd()}")
    main(args.path, args.index_file)
