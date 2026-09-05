# Battery RUL Estimation with Federated Weighted Average

## 1. Project Overview

This project implements a **Federated Learning framework for Remaining Useful Life (RUL) estimation of lithium-ion batteries**.

The framework combines:

- **CNN-LSTM-Attention** for RUL prediction
- **FedWeightedAvg** for performance-based federated aggregation
- **Chronological preprocessing** to avoid temporal data leakage
- **Cross-battery evaluation** using unseen global test batteries

Two independent federated subgroups are evaluated:

- **Subgroup 1:** B0005, B0006, B0007 → B0018
- **Subgroup 2:** B0025, B0026, B0027 → B0028

The repository includes preprocessing, federated training, baseline comparisons, alpha sensitivity analysis, architecture ablation, evaluation, and saved experimental results.

---

## 2. Problem Statement

Accurate **Remaining Useful Life (RUL)** estimation is important for monitoring lithium-ion battery degradation.

Battery data from different batteries can have different degradation patterns, making the data **heterogeneous and non-IID**. Centralizing all battery data is also not desirable in distributed applications.

Federated Learning enables multiple battery clients to collaboratively train a model without sharing their raw data. However, conventional **FedAvg** does not consider the validation performance of individual clients when aggregating their models.

This project addresses this problem using **FedWeightedAvg**, which assigns higher aggregation weights to clients with better local validation performance.

The framework therefore focuses on:

- Distributed battery RUL estimation
- Heterogeneous client data
- Performance-aware model aggregation
- Generalization to unseen batteries

---

## 3. Project Objective

The main objectives of this project are:

- Develop a **CNN-LSTM-Attention** model for battery RUL estimation.
- Implement **Federated Learning** across multiple battery clients.
- Use **FedWeightedAvg** to assign aggregation weights based on client validation RMSE.
- Evaluate the global model on **unseen test batteries**.
- Compare FedWeightedAvg with **FedAvg, FedProx, and FedAdam**.
- Analyze the effect of different **α values**.
- Perform **CNN, LSTM, and CNN-LSTM ablation studies**.
- Save model, convergence, validation, aggregation-weight, and evaluation results for reproducibility.
## Dataset

### NASA PCoE Lithium-Ion Battery Dataset

Source:

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Batteries used:

- B0005
- B0006
- B0007
- B0018
- B0025
- B0026
- B0027
- B0028

### Federated Experimental Groups

#### Subgroup 1

Training Clients:

- B0005
- B0006
- B0007

Global Test Battery:

- B0018

#### Subgroup 2

Training Clients:

- B0025
- B0026
- B0027

Global Test Battery:

- B0028

### Dataset Preparation

The original NASA `.mat` files are not included in the repository because of their large size.

The raw battery files are first processed to generate the RUL-labelled CSV data used by the federated learning pipeline.


## 5. Experimental Battery Groups

The experiments are divided into two independent battery subgroups.

### Subgroup 1

| Role | Battery |
|------|---------|
| Client 1 | B0005 |
| Client 2 | B0006 |
| Client 3 | B0007 |
| Global Test | B0018 |

### Subgroup 2

| Role | Battery |
|------|---------|
| Client 1 | B0025 |
| Client 2 | B0026 |
| Client 3 | B0027 |
| Global Test | B0028 |

For each subgroup:

- The three client batteries participate in federated training.
- Client data remains separated during training.
- The global test battery is not used as a training client.
- The final global model is evaluated on the unseen global test battery.

## 6. Data Preprocessing Pipeline

The raw NASA battery data is processed before federated training.

The preprocessing workflow is:

1. Extract discharge-cycle data from the original battery `.mat` files.
2. Calculate the battery **RUL** from the capacity degradation profile.
3. Generate processed CSV files containing the required battery features and RUL target.
4. Sort each battery dataset chronologically by cycle.
5. Split each battery into local training, validation, and test data.
6. Reserve the later portion of client data for global validation.
7. Normalize the features using the preprocessing pipeline.
8. Generate fixed-length sequences using a sliding window.
9. Use the resulting sequences as input to the CNN-LSTM-Attention model.

### Features Used

The processed datasets contain:

- `cycle`
- `ambient_temperature`
- `capacity`
- `voltage_measured`
- `current_measured`
- `temperature_measured`
- `current_load`
- `voltage_load`
- `time`
- `RUL`

The preprocessing is performed separately for the two battery subgroups to maintain the intended federated data separation.
## 7. Data Partitioning

Data is partitioned chronologically to prevent future battery cycles from being used to predict earlier cycles.

### Client Battery Split

For each participating client battery:

| Partition | Ratio |
|-----------|-------|
| Local Training | 64% |
| Local Validation | 20% |
| Local Test | 16% |

The **last 20% of chronological cycles** are reserved for global validation, while the earlier cycles are used for the local partitions above.

### Global Test

The designated global test battery in each subgroup is kept completely separate from federated training and validation.

- Subgroup 1 → **B0018**
- Subgroup 2 → **B0028**

This provides an evaluation of the final global model on a battery that was not used as a federated training client.
## 8. Feature Normalization

Normalization is performed separately for each federated subgroup.

### Feature Scaling

A single **Min-Max Scaler** is fitted using the combined training features of all three participating clients.

The scaler is fitted **only on client training data** and then applied unchanged to:

- Client training data
- Client validation data
- Client local-test data
- Global validation data
- Global test data

The input features are:

- `capacity`
- `ambient_temperature`
- `voltage_measured`
- `current_measured`
- `temperature_measured`
- `current_load`
- `voltage_load`

This ensures that all clients within a subgroup use the same feature scaling while preventing validation and test data from influencing the scaler.

### RUL Scaling

A common RUL scale is also used for each subgroup.

For every client, the maximum RUL in its **training set** is calculated. The largest of these three training-set maxima is selected as the **common maximum RUL** for that subgroup.

All RUL targets in the subgroup are then divided by this common value.

The global test battery is therefore scaled using the **training-derived common RUL scale**, rather than its own maximum RUL.

### Saved Normalization Parameters

The following parameters are saved in:

```text
normalization_params.pkl
```
### Data Locality

Client datasets remain separated throughout the preprocessing pipeline.
Only the training features from the three clients are pooled temporarily to
fit the common subgroup-level feature scaler; validation and test data are
never used for fitting.

The held-out global test battery is not used for fitting either the feature
scaler or the RUL scale.

## 9. Sequence Generation

After normalization, the time-series data is converted into fixed-length sequences for model training.

| Parameter | Value |
|-----------|-------|
| Window Length | 30 records |
| Input Features | 7 |
| Target | Mean RUL of final 3 records |

For each battery, a sliding window of **30 consecutive records** is created.

The seven input features are:

- `capacity`
- `ambient_temperature`
- `voltage_measured`
- `current_measured`
- `temperature_measured`
- `current_load`
- `voltage_load`

For each 30-record window, the prediction target is calculated as the **mean RUL of the final 3 records** in that window.

The resulting input shape is:

```text
(samples, 30, 7)
```
Sequences are generated separately for each battery so that a sequence never crosses from one battery into another.

## 10. Proposed Model Architecture

The proposed model is a **CNN-LSTM-Attention** network for sequence-based RUL regression.

### Architecture

```text
Input Sequence
(30 × 7)
     │
     ▼
1D CNN
7 → 64 channels
Kernel Size = 3
Padding = 1
     │
     ▼
ReLU
     │
     ▼
LSTM
Input Size = 64
Hidden Size = 64
     │
     ▼
Attention
64 → 1
     │
     ▼
Fully Connected
64 → 1
     │
     ▼
Predicted RUL
```
### Model Components

| Component | Configuration |
|-----------|---------------|
| Input | 30 × 7 |
| CNN | Conv1D, 7 → 64 |
| Kernel Size | 3 |
| Activation | ReLU |
| LSTM | 64 input, 64 hidden |
| Attention | Linear 64 → 1 |
| Output | Linear 64 → 1 |
| Task | RUL Regression |

## 11. Federated Learning Configuration

The federated training configuration used in the experiments is:

| Parameter | Value |
|-----------|-------|
| Communication Rounds | 50 |
| Local Epochs | 5 |
| Batch Size | 64 |
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| Early Stopping Patience | 5 |
| Aggregation α | 5.0 |
| Random Seed | 42 |
| Device | CUDA (if available) |

Each communication round consists of local training at the three participating clients, followed by aggregation of their trained model parameters to form the updated global model.

The global model is evaluated using the designated global validation data during training, while the held-out global test battery is reserved for final evaluation.
## 12. FedWeightedAvg Aggregation

FedWeightedAvg aggregates the three locally trained client models using weights determined from their local validation RMSE.

### Weight Calculation

For each communication round:

1. Each client trains the current global model locally.
2. Each client is evaluated on its local validation data.
3. The validation RMSE of each client is recorded.
4. Aggregation weights are calculated using a softmax-based function.
5. Clients with lower validation RMSE receive higher aggregation weights.
6. The client model parameters are combined using the calculated weights.
7. The resulting parameters form the next global model.

The weighting function is:

```text
weight_i = exp(-α × RMSE_i) / Σ exp(-α × RMSE_j)
```

where:

- `RMSE_i` is the local validation RMSE of client `i`.
- `α` controls the sensitivity of the aggregation weights.
- Higher-performing clients with lower RMSE receive larger weights.

### Configuration

| Parameter | Value |
|-----------|-------|
| Aggregation Method | FedWeightedAvg |
| Weight Metric | Local Validation RMSE |
| Weight Function | Softmax over negative RMSE |
| α | 5.0 |
| Number of Clients | 3 |

The aggregation weights for each communication round are saved in:

```text
weights.npy
```

The client validation RMSE history is saved in:

```text
client_rmse.npy
```

Local validation is used specifically for FedWeightedAvg client weighting, while the separate global validation dataset is used for global-model validation and early stopping.

## 13. Experimental Studies

The repository includes the following experimental studies:

### 1. FedWeightedAvg

The proposed CNN-LSTM-Attention model is trained using FedWeightedAvg on both battery subgroups:

- Subgroup 1: B0005, B0006, B0007 → B0018
- Subgroup 2: B0025, B0026, B0027 → B0028

### 2. Federated Aggregation Comparison

The proposed FedWeightedAvg approach is compared with:

- FedAvg
- FedProx
- FedAdam

### 3. Alpha Sensitivity Analysis

The effect of the FedWeightedAvg sensitivity parameter `α` is evaluated using different alpha values like  0.5, 1.0, 2.0, 5.0, 10.0.

### 4. Architecture Ablation Study

The contribution of the model components is evaluated using:

- CNN
- LSTM
- CNN-LSTM
- CNN-LSTM-Attention
## 14. Repository Structure and Execution

### Repository Structure

```text
.
├── data/
├── data-splits/
├── processed_data/
│
├── FL_training.py
├── fedAverage.py
├── fedProx.py
├── fedAdam.py
│
├── ablation_CNN_LSTM.py
├── ablation_CNN_only.py
├── ablation_LSTM_only.py
│
├── data_partitioning.py
├── check_normalized_data.py
│
├── OUTPUTS_FEDAVG/
├── ablation_results/
├── alpha_outputs/
│
└── README.md
```

### Execution Order

The main preprocessing and training workflow is:

```text
raw-data/ NASA Battery .mat Files
        │
        ▼
RUL Generation
        │
        ▼
data/
        │
        ▼
Data Partitioning
        │
        ▼
data-splits/
        │
        ▼
Normalization + Sequence Generation
        │
        ▼
processed_data/
        │
        ▼
Federated Training
        │
        ├── FedWeightedAvg
        ├── FedAvg
        ├── FedProx
        └── FedAdam
        │
        ▼
Evaluation and Saved Results
```

### Main Federated Training

After the processed datasets are generated, run:

```bash
python FL_training.py
```

The script automatically detects CUDA when available and trains the two configured battery subgroups.

Results are saved under:

```text
outputs/
├── output_5_18/
└── output_25_28/
```

Additional scripts in the repository are used for the aggregation comparisons and ablation experiments.
## 15. Results

The final global model was evaluated on the held-out battery of each subgroup.

### Global Test Performance

| Subgroup | Global Test Battery | MAE | RMSE | Best Round |
|----------|---------------------|-----|------|------------|
| 5_18 | B0018 | 0.072867 | 0.081630 | 6 |
| 25_28 | B0028 | 0.227095 | 0.261879 | 8 |

### Validation Performance

| Subgroup | Best Validation RMSE | Best Round |
|----------|----------------------|------------|
| 5_18 | 0.084712 | 6 |
| 25_28 | 0.419458 | 8 |

The reported values are taken from the saved experimental outputs generated by the current implementation in this repository.

## 16. Reproducibility

### Software

- Python 3.12+
- PyTorch
- NumPy
- Pandas
- Scikit-learn

### Hardware

The training code automatically uses **CUDA when an NVIDIA GPU is available**; otherwise, it runs on CPU.

### Random Seed

A fixed random seed of `42` is used for reproducibility.

```python
SEED = 42
```

The seed is applied to Python, NumPy, and PyTorch. CUDA-specific seeding is also enabled when a GPU is available.

### Reproducing the Experiments

The general workflow is:

```text
Download NASA battery data
        ↓
Generate RUL-labelled CSV files
        ↓
Partition the battery data
        ↓
Normalize and generate sequences
        ↓
Run federated training
        ↓
Run comparison / ablation experiments
        ↓
Evaluate and inspect saved results
```

The generated datasets and experimental outputs are kept separate from the original raw battery files.

## 17. Citation

If you use this project or its implementation in your research, please cite the associated research paper.

The paper describes the proposed **CNN-LSTM-Attention architecture** and **FedWeightedAvg-based federated learning framework** for battery RUL estimation.