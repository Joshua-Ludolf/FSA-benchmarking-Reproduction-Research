import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import sys

class ProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        sys.stdout.flush()  # Ensure output is displayed immediately
        if logs:
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            print(f"\rEpoch {epoch + 1}: loss = {loss:.6f}, val_loss = {val_loss:.6f}", end='')
            sys.stdout.flush()

def load_and_preprocess_data(base_path, index_file):
    print(f"Script is being run with the following base path: {base_path} and index file: {index_file}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Load the index file
    file_path = os.path.join(base_path, index_file)
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    print(f"Loaded data with columns: {data.columns.tolist()}")
    
    # Rename columns
    data = data.rename(columns={'cluster': 'Cluster', 'host_name': 'Host'})
    print(f"Renamed columns: {data.columns.tolist()}")
    
    # Generate synthetic metrics for each host
    np.random.seed(42)
    num_records = len(data)
    
    # Generate synthetic metrics
    data['mean'] = np.random.rand(num_records)
    data['std'] = np.random.rand(num_records)
    data['min'] = np.random.rand(num_records)
    data['max'] = np.random.rand(num_records)
    data['median'] = np.random.rand(num_records)
    data['skew'] = np.random.rand(num_records)
    data['q25'] = np.random.rand(num_records)
    data['q75'] = np.random.rand(num_records)
    
    # Add target variable (0 for normal, 1 for anomaly)
    data['target'] = 0  # Initialize all as normal
    
    # Randomly select some records as anomalies
    anomaly_indices = np.random.choice(num_records, size=int(num_records * 0.15), replace=False)
    data.loc[anomaly_indices, 'target'] = 1
    
    print("Data columns before processing:", data.columns.tolist())
    print("\nFirst few rows of the data:")
    print(data.head())
    
    return data

def train_autoencoder(data):
    # Select numeric columns for training
    numeric_columns = ['mean', 'std', 'min', 'max', 'median', 'skew', 'q25', 'q75']
    X = data[numeric_columns].values
    
    print("Numeric columns for model training:", numeric_columns)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Shape of scaled data:", X_scaled.shape)
    
    # Define autoencoder architecture
    input_dim = X_scaled.shape[1]
    encoding_dim = 4
    
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation='relu')(input_layer)
    
    # Decoder
    decoder = Dense(input_dim, activation='sigmoid')(encoder)
    
    # Autoencoder model
    autoencoder = Model(input_layer, decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Create progress callback
    progress_callback = ProgressCallback()
    
    print("\nTraining autoencoder model:")
    # Train the model with custom callback
    history = autoencoder.fit(X_scaled, X_scaled,
                            epochs=50,
                            batch_size=32,
                            shuffle=True,
                            validation_split=0.2,
                            verbose=0,
                            callbacks=[progress_callback])
    
    print("\n\nTraining completed!")
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"Final training loss: {final_loss:.6f}")
    print(f"Final validation loss: {final_val_loss:.6f}")
    
    # Save the model
    autoencoder.save('autoencoder_model.h5')
    print("Autoencoder model saved successfully.")
    
    # Get reconstruction error
    reconstructed = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
    
    # Determine threshold for anomaly detection (e.g., 95th percentile)
    threshold = np.percentile(mse, 95)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'Date': datetime.now().strftime('%Y-%m-%d'),
        'Cluster': data['Cluster'],
        'Host': data['Host'],
        'Disk': 'disk1',
        'anomaly_score': mse,
        'Prediction': mse > threshold
    })
    
    return predictions_df

def main():
    # Define paths
    base_path = "data"
    index_file = "index/A_index.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Load and preprocess data
    data = load_and_preprocess_data(base_path, index_file)
    
    # Train autoencoder and get predictions
    predictions = train_autoencoder(data)
    
    # Save predictions to CSV
    predictions.to_csv('output/autoencoder_output.csv', index=False)

if __name__ == "__main__":
    main()
