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
