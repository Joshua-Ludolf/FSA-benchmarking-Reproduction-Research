### Project Overview

This project aims to explore and benchmark various machine learning models for detecting disks at high risk of experiencing fail-slow anomalies.

### Notebooks and Files

**Team Project - Setting up experiment - Joshua Ludolf, Yesmin Hernandez-Reyna, Matthew Trevino.ipynb**  
This notebook details the entire process of running the algorithms on Chameleon, including launching nodes. Currently, we only run two clusters of Perseus since Trovi has only 1GB of memory, which we cannot exceed. For access to all 25 clusters, please contact at xikang@uchicago.edu, and he will share the repository or you can download the dataset on https://tianchi.aliyun.com/dataset/144479, then run it locally by his & our scripts.

**Team Project - Result Parser - Joshua Ludolf, Yesmin Hernandez, Matthew Trevino.ipynb**  
This notebook shows the results from all the machine learning algorithms and provides analysis.

**REPRODUCTION_RESEARCH__FSA_BENCHMARK___Joshua_Ludolf__Yesmin_Reyna_Hernandez__Matthew_Trevino.pdf**  
This report offers a comprehensive data collected from this research.

### Directory Structure

**index directory**  
Contains the index information for each cluster.

**output**  
Holds the output from Chameleon.

**scripts**  
Contains the fail-slow detection algorithms and machine learning models.

### Machine Learning Models

1. **Cost-Sensitive Ranking Model**  
   Inspired by the paper "Improving Service Availability of Cloud Systems by Predicting Disk Error" (USENIX ATC '18), this model ranks disks based on their fail-slow risk.

2. **Multi-Prediction Models**  
   Drawing from "Improving Storage System Reliability with Proactive Error Prediction" (USENIX ATC '17), this approach uses multiple traditional machine learning models to evaluate disk health using diverse features. Various models were tested, with the Random Forest classifier proving most effective.

3. **LSTM Model**  
   This model employs Long Short-Term Memory (LSTM) networks, trained on the first day's data for each cluster and evaluated on data spanning all days. It captures temporal dependencies to accurately predict fail-slow anomalies over time.

4. **PatchTST Model**  
   An advanced sequence model that leverages transformers to handle time series prediction and fail-slow detection.

5. **GPT-4o-mini**  
   A large language model used to analyze disk metrics and detect fail-slow conditions. Please replace `openai_api_key` in the code where necessary.

6. **Autoencoder** <br>
   Model utilizes an encoder and decoder method to analyze disk metrics and fail-slow detection.
   
7.  **Isolation Forest** <br>
    The algorithm created multiple iTrees, where each tree isolated observations by randomly selecting features and split values.
    
8.  **Suport Vector Machine (SVM)** <br>
    The algorithm worked by finding the optimal hyperplane that maximized the margin between different classes. I employed techniques such as cross-validation to fine-tune the model parameters and prevent overfitting.

## Hybrid Deep-Learning for Fail-Slow Disk Detection in the FSA-Benchmark

### Overview
Fail-slow disks – where performance degrades gradually before an outright failure – are increasingly common in large-scale cloud storage systems. While traditional machine learning models (XGBoost, Random Forest) and shallow time-series methods (LSTM, SVM) have demonstrated moderate success in detecting fail-slow conditions, they struggle to capture the complex, high-frequency correlations in disk metrics that precede these events.

This research proposes a **hybrid deep learning framework** that combines convolutional-recurrent layers with self-attention mechanisms to better model both spatial and temporal dependencies in disk performance metrics. The architecture is evaluated on the same Cluster A and B splits used in the original FSA-Benchmark study.

### Architecture

The proposed hybrid model consists of the following components:

- **CNN Block**: 1D convolutional layers (kernel=3, 64 filters) with BatchNormalization, ReLU activation, and MaxPooling (size=2), repeated 3 times for hierarchical feature extraction.
- **RNN Block**: Bidirectional LSTM with hidden size of 128 to capture temporal dependencies across time-series data.
- **Self-Attention Layer**: Multi-head attention mechanism (4 heads) applied over LSTM outputs to model long-range cross-time dependencies and identify critical time periods.
- **Dense Head**: A 2-layer MLP (128→64→1) with sigmoid activation for binary classification.
- **Regularization**: Dropout (0.3) and weight decay (1e-5) to prevent overfitting.

### Methodology

#### Data Preparation
- Slide a **5-minute (20-step)** window over time-series data from each disk.
- Label windows containing a fail-slow event within the next **10 minutes** as positive.
- Input features: **20-dimensional metrics** including latency, throughput, error rate, queue depth, and other performance indicators.
- Address class imbalance using SMOTE or class-weighted loss functions.

#### Training Configuration
- **Loss Function**: Binary cross-entropy with class weights.
- **Optimizer**: AdamW (learning rate = 1e-4).
- **Learning Rate Schedule**: Cosine annealing for adaptive learning rate decay.
- **Early Stopping**: Based on validation AUROC (patience = 10 epochs).
- **Batch Size**: 128
- **Epochs**: 20

#### Evaluation Metrics
- **Precision, Recall, F1-score**: Standard classification metrics.
- **AUROC**: Area under the receiver operating characteristic curve.
- **Time-to-Alert**: Average number of minutes from the first abnormal window to the first positive prediction.
- **Cross-validation**: 5-fold on the training set with final evaluation on a held-out 10% test set (Cluster A vs. Cluster B).

#### Baselines
The hybrid model is compared against:
- XGBoost (with best hyperparameters from the original FSA-Benchmark paper)
- Random Forest
- Simple LSTM baseline

### Explainability and Validation

- **Attention Maps**: Visualize attention weights per metric and time-point to identify influential features.
- **SHAP Values**: Compute SHAP values for the dense layer to explain individual predictions.
- **Expert Validation**: Attention maps are validated with storage experts to confirm that the model highlights expected anomalies (e.g., latency spikes).

### Results

The hybrid architecture achieves improved performance by leveraging multi-scale feature extraction through the CNN block and temporal pattern recognition through the bidirectional LSTM. The self-attention mechanism enables the model to focus on the most relevant time-steps and metrics, resulting in better interpretability and higher detection accuracy compared to baseline models.

### Reproducibility

All code, data, and artifacts are available in this repository:
- **Data**: Raw time-series disk performance metrics from the FSA-Benchmark.
- **Dockerfile**: Containerized environment for consistent setup.
- **Jupyter Notebooks**: `Hybrid-Deep-Model.ipynb` for visualization and analysis.
- **Python Scripts**: `scripts/hybrid_cnn_rnn_attention.py` contains the full implementation.
- **Pre-trained Models**: Model weights and metadata saved in `output/`.

## Windows setup and how to run

The repository includes a PowerShell runner (`run_experiments.ps1`) that mirrors the bash script. Follow these steps on Windows:

1) Create and activate a virtual environment, then install dependencies

```powershell
python -m venv .venv; . .\.venv\Scripts\Activate.ps1; pip install -U pip; pip install -r requirements.txt
```

2) Prepare the dataset directory structure

- Download the Perseus dataset referenced in the notebooks/README.
- Layout should be:
   - `<PerseusDir>/<cluster>/<host>/<YYYY-MM-DD>.csv`
   - Example: `data/cluster_A/host-0001/2023-05-01.csv`
- The files in `index/` are already provided and include `all_drive_info.csv` and per-split indices `A_index.csv`, `B_index.csv` plus `slow_drive_info.csv`.

3) Set your OpenAI API key (only required for `scripts/GPT-4.py`)

```powershell
# Current session only
$env:OPENAI_API_KEY = "sk-..."
# Optional: persist for future sessions
setx OPENAI_API_KEY "sk-..."
```

4) Run all experiments (note: if your Perseus data is under `index/`, set `-PerseusDir index`; the PowerShell runner now defaults to `index`)

```powershell
./run_experiments.ps1 -PerseusDir index -IndexDir index
```

Outputs will be written under `output/` and compressed to `output.zip`.

Notes:
- If `slow_drive_info.csv` is not found under `PerseusDir`, the scripts will automatically look for it next to the provided index file (e.g., `index/slow_drive_info.csv`).
- To reduce token usage/costs for GPT-4o-mini, you can pass `-s` to `scripts/GPT-4.py` to use host-disk statistics instead of raw time series.

### Results parsing and metrics

After running models, you can compute precision/recall/F1 vs. the provided ground truth list (`index/slow_drive_info.csv`) with:

```powershell
python scripts/parse_results.py -o output -g index/slow_drive_info.csv -s output/metrics_summary.csv
```

This generates:
- `output/metrics_summary.csv` and `output/metrics_summary.md`: per-model metrics
- `output/*_parsed.csv`: normalized predictions parsed from each model’s raw output
