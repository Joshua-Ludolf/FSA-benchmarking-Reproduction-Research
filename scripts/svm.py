import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import warnings
from scipy import stats

def load_and_preprocess_data(base_path, index_file):
    """Enhanced data loading and preprocessing with additional features"""
    print(f"Loading data from {os.path.join(base_path, index_file)}")
    
    try:
        # Load the index file
        data = pd.read_csv(os.path.join(base_path, index_file))
        print(f"Loaded {len(data)} records")
        
        # Rename columns
        data = data.rename(columns={'cluster': 'Cluster', 'host_name': 'Host'})
        
        # Generate enhanced synthetic metrics
        np.random.seed(42)
        num_records = len(data)
        
        # Basic statistical features
        data['mean'] = np.random.normal(loc=0.5, scale=0.15, size=num_records)
        data['std'] = np.abs(np.random.normal(loc=0.2, scale=0.05, size=num_records))
        data['min'] = data['mean'] - data['std'] * np.random.uniform(1, 2, num_records)
        data['max'] = data['mean'] + data['std'] * np.random.uniform(1, 2, num_records)
        data['median'] = data['mean'] + np.random.normal(0, 0.1, num_records)
        
        # Advanced statistical features
        data['skew'] = stats.skewnorm.rvs(a=5, loc=0.1, scale=0.1, size=num_records)
        data['kurtosis'] = np.abs(np.random.normal(3, 0.5, num_records))  # Normal distribution has kurtosis = 3
        data['q25'] = data['mean'] - data['std'] * 0.67
        data['q75'] = data['mean'] + data['std'] * 0.67
        
        # Temporal features (synthetic)
        data['time_since_last_failure'] = np.random.exponential(50, size=num_records)
        data['failure_count_7d'] = np.random.poisson(lam=0.3, size=num_records)
        
        # Derived features
        data['iqr'] = data['q75'] - data['q25']
        data['cv'] = data['std'] / data['mean']  # Coefficient of variation
        data['range'] = data['max'] - data['min']
        data['mad'] = np.abs(data['mean'] - data['median'])  # Mean absolute deviation from median
        
        # Add disk information
        data['Disk'] = 'disk1'
        
        # Remove any infinite or NaN values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(data.mean())
        
        print("Data preprocessing completed successfully")
        return data
        
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise

def select_features(data):
    """Select and engineer features for the model"""
    feature_columns = [
        'mean', 'std', 'min', 'max', 'median', 
        'skew', 'kurtosis', 'q25', 'q75', 
        'time_since_last_failure', 'failure_count_7d',
        'iqr', 'cv', 'range', 'mad'
    ]
    
    return data[feature_columns]

def tune_svm_parameters(X_train, contamination=0.1):
    """Tune SVM hyperparameters using grid search"""
    param_grid = {
        'kernel': ['rbf'],
        'nu': [0.01, 0.05, 0.1, 0.15, 0.2],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
    }
    
    # Use IsolationForest to get rough labels for parameter tuning
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    rough_labels = iso_forest.fit_predict(X_train)
    rough_labels = (rough_labels + 1) // 2  # Convert to 0/1 labels
    
    # Grid search
    svm = OneClassSVM()
    grid_search = GridSearchCV(
        svm, param_grid, 
        scoring='roc_auc',
        cv=5, n_jobs=-1, verbose=1
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_train, rough_labels)
    
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_params_

def train_svm_model(data):
    """Train an improved SVM model with feature engineering and parameter tuning"""
    print("Starting model training process...")
    
    try:
        # Select features
        X = select_features(data)
        print(f"Selected {len(X.columns)} features for training")
        
        # Scale features using RobustScaler (less sensitive to outliers)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data for parameter tuning
        X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
        
        # Tune parameters
        best_params = tune_svm_parameters(X_train)
        
        # Train final model with best parameters
        svm = OneClassSVM(**best_params)
        svm.fit(X_scaled)
        
        # Get anomaly scores
        anomaly_scores = -svm.decision_function(X_scaled)
        
        # Use adaptive thresholding
        threshold = np.percentile(anomaly_scores, 95)  # Use 95th percentile as threshold
        
        # Scale scores to [0, 1] range using robust scaling
        score_scaler = RobustScaler()
        normalized_scores = score_scaler.fit_transform(anomaly_scores.reshape(-1, 1)).ravel()
        normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Cluster': data['Cluster'],
            'Host': data['Host'],
            'Disk': data['Disk'],
            'probability': normalized_scores,
            'Prediction': normalized_scores > 0.85,  # Using fixed threshold for anomaly detection
            'anomaly_score': anomaly_scores,
            'score_threshold': threshold
        })
        
        # Calculate confidence metrics
        predictions_df['confidence'] = 1 - (1 / (1 + np.exp(-(anomaly_scores - threshold))))
        
        # Add feature importance approximation
        feature_importance = calculate_feature_importance(X_scaled, svm)
        print("\nFeature Importance:")
        for feature, importance in feature_importance.items():
            print(f"{feature}: {importance:.4f}")
        
        print("\nModel Training Summary:")
        print(f"Total samples: {len(predictions_df)}")
        print(f"Anomalies detected: {predictions_df['Prediction'].sum()}")
        print(f"Anomaly rate: {(predictions_df['Prediction'].sum() / len(predictions_df)) * 100:.2f}%")
        
        return predictions_df
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise

def calculate_feature_importance(X, model):
    """Calculate approximate feature importance for OneClassSVM"""
    support_vectors = model.support_vectors_
    dual_coef = np.abs(model.dual_coef_[0])
    
    # Calculate importance as the weighted sum of support vectors
    importance = np.sum(support_vectors * dual_coef[:, np.newaxis], axis=0)
    importance = np.abs(importance)
    importance = importance / np.sum(importance)
    
    return dict(zip(select_features(pd.DataFrame()).columns, importance))

def main():
    # Define paths
    base_path = "data"
    index_file = "index/A_index.csv"
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    try:
        # Load and preprocess data
        print("Starting data preprocessing...")
        data = load_and_preprocess_data(base_path, index_file)
        
        # Train model and get predictions
        print("\nStarting model training...")
        predictions = train_svm_model(data)
        
        # Save predictions
        output_file = 'output/svm_output.csv'
        predictions.to_csv(output_file, index=False)
        print(f"\nSaved predictions to {output_file}")
        
        # Save model performance metrics
        metrics = {
            'total_samples': len(predictions),
            'anomalies_detected': predictions['Prediction'].sum(),
            'anomaly_rate': (predictions['Prediction'].sum() / len(predictions)) * 100,
            'mean_confidence': predictions['confidence'].mean(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('output/svm_metrics.csv', index=False)
        print("\nSaved model metrics to output/svm_metrics.csv")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
