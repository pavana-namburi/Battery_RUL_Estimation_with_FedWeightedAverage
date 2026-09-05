# ============================================================
# 1. IMPORTS
# ============================================================

import os
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FEDERATED LEARNING TRAINING")
print("CNN-LSTM-ATTENTION + FedWeightedAvg")
print("ALPHA = 10.0")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print("=" * 70)


# ============================================================
# 4. PATHS
# ============================================================
#
# IMPORTANT:
# No machine-specific path is hardcoded.
#
# BASE_DIR automatically points to the folder containing
# this Python script.
#
# Example:
#
# reproducing_FL/
#     fedweightedavg_alpha_10.0.py
#
# Then:
#
# BASE_DIR =
#     reproducing_FL/
#
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------
# Processed data root
# ------------------------------------------------------------

PROCESSED_ROOT = (
    BASE_DIR / "processed_data"
)


# ------------------------------------------------------------
# Alpha = 10.0 output roots
# ------------------------------------------------------------

OUTPUT_ROOT_5_18 = (
    BASE_DIR
    / "alpha_outputs"
    / "subgroup_5_18"
    / "alpha_10.0_results"
)

OUTPUT_ROOT_25_28 = (
    BASE_DIR
    / "alpha_outputs"
    / "subgroup_25_28"
    / "alpha_10.0_results"
)


# ============================================================
# 5. SUBGROUP CONFIGURATION
# ============================================================

SUBGROUPS = {

    "5_18": {

        "data_dir":
            PROCESSED_ROOT
            / "processed_data_5_18",

        "output_dir":
            OUTPUT_ROOT_5_18,

        "global_test_battery":
            "B0018"
    },


    "25_28": {

        "data_dir":
            PROCESSED_ROOT
            / "processed_data_25_28",

        "output_dir":
            OUTPUT_ROOT_25_28,

        "global_test_battery":
            "B0028"
    }
}


# ============================================================
# 6. CLIENT CONFIGURATION
# ============================================================

CLIENTS = [
    "client_1",
    "client_2",
    "client_3"
]


# ============================================================
# 7. TRAINING HYPERPARAMETERS
# ============================================================

LOCAL_EPOCHS = 5

BATCH_SIZE = 64

LEARNING_RATE = 1e-4

MAX_ROUNDS = 50

PATIENCE = 5


# ------------------------------------------------------------
# IMPORTANT:
# Alpha changed from 5.0 to 10.0.
# ------------------------------------------------------------

ALPHA = 10.0


# ============================================================
# 8. MODEL
# ============================================================

class CNNLSTMAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=7,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )

        self.attn = nn.Linear(
            64,
            1
        )

        self.fc = nn.Linear(
            64,
            1
        )


    def forward(
        self,
        x,
        return_attn=False
    ):

        # ----------------------------------------------------
        # Input:
        # [batch, sequence_length, features]
        #
        # Example:
        # [64, 30, 7]
        # ----------------------------------------------------

        # Conv1D requires:
        # [batch, features, sequence_length]

        x = x.permute(
            0,
            2,
            1
        )

        x = self.relu(
            self.conv(x)
        )

        # Back to:
        # [batch, sequence_length, features]

        x = x.permute(
            0,
            2,
            1
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        out, _ = self.lstm(x)


        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        attention_scores = self.attn(
            out
        )


        # ----------------------------------------------------
        # Attention weights
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )


        # ----------------------------------------------------
        # Context vector
        # ----------------------------------------------------

        context = torch.sum(
            attention_weights * out,
            dim=1
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = self.fc(
            context
        ).squeeze(-1)


        if return_attn:

            return (
                prediction,
                attention_weights.squeeze(-1)
            )

        return prediction


# ============================================================
# 9. METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred
    ).reshape(-1)


    mae = mean_absolute_error(
        y_true,
        y_pred
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


    return mae, rmse


# ============================================================
# 10. LOAD NPZ
# ============================================================

def load_npz_data(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


    return np.load(
        path,
        allow_pickle=True
    )


# ============================================================
# 11. CONVERT TO TORCH TENSOR
# ============================================================

def to_tensor(x):

    return torch.tensor(
        np.asarray(x),
        dtype=torch.float32
    )


# ============================================================
# 12. LOCAL TRAINING
# ============================================================

def train_local_model(
    global_model,
    X_train,
    y_train
):

    local_model = copy.deepcopy(
        global_model
    ).to(DEVICE)


    local_model.train()


    optimizer = optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE
    )


    criterion = nn.L1Loss()


    X_train = to_tensor(
        X_train
    )


    y_train = to_tensor(
        y_train
    ).reshape(-1)


    n_samples = len(
        X_train
    )


    for epoch in range(
        LOCAL_EPOCHS
    ):

        indices = torch.randperm(
            n_samples
        )


        for start in range(
            0,
            n_samples,
            BATCH_SIZE
        ):

            batch_indices = indices[
                start:start + BATCH_SIZE
            ]


            X_batch = X_train[
                batch_indices
            ].to(DEVICE)


            y_batch = y_train[
                batch_indices
            ].to(DEVICE)


            optimizer.zero_grad()


            predictions = local_model(
                X_batch
            )


            loss = criterion(
                predictions,
                y_batch
            )


            loss.backward()


            optimizer.step()


    return local_model


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y
):

    model.eval()


    X_tensor = to_tensor(
        X
    ).to(DEVICE)


    y_tensor = to_tensor(
        y
    ).reshape(-1)


    predictions = []


    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            BATCH_SIZE
        ):

            X_batch = X_tensor[
                start:start + BATCH_SIZE
            ]


            batch_predictions = model(
                X_batch
            )


            predictions.append(
                batch_predictions
                .cpu()
                .numpy()
            )


    predictions = np.concatenate(
        predictions
    )


    y_true = y_tensor.numpy()


    mae, rmse = calculate_metrics(
        y_true,
        predictions
    )


    return mae, rmse


# ============================================================
# 14. FEDWEIGHTEDAVG WEIGHTS
# ============================================================

def calculate_fedweightedavg_weights(
    client_rmses
):

    client_rmses = np.asarray(
        client_rmses,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Lower RMSE -> larger aggregation weight
    #
    # FedWeightedAvg:
    #
    # weight_i =
    # exp(-ALPHA * RMSE_i)
    # -----------------------------------------------
    # Normalize weights so that:
    #
    # sum(weights) = 1
    # --------------------------------------------------------

    exp_weights = np.exp(
        -ALPHA * client_rmses
    )


    weight_sum = np.sum(
        exp_weights
    )


    if (
        weight_sum == 0
        or not np.isfinite(weight_sum)
    ):

        weights = (
            np.ones(
                len(client_rmses)
            )
            /
            len(client_rmses)
        )

    else:

        weights = (
            exp_weights
            /
            weight_sum
        )


    return weights


# ============================================================
# 15. FEDWEIGHTED AVERAGE
# ============================================================

def fed_weighted_average(
    client_models,
    weights
):

    state_dicts = [
        model.state_dict()
        for model in client_models
    ]


    global_state = {}


    for key in state_dicts[0].keys():

        aggregated = torch.zeros_like(
            state_dicts[0][key]
        )


        for i in range(
            len(client_models)
        ):

            aggregated += (
                state_dicts[i][key]
                *
                float(weights[i])
            )


        global_state[key] = aggregated


    return global_state


# ============================================================
# 16. SAVE TRAINING ARRAYS
# ============================================================

def save_training_arrays(
    output_dir,
    rounds,
    client_rmse_history,
    weight_history
):

    np.save(
        output_dir / "rounds.npy",
        np.asarray(
            rounds,
            dtype=np.int32
        )
    )


    np.save(
        output_dir / "client_rmse.npy",
        np.asarray(
            client_rmse_history,
            dtype=np.float32
        )
    )


    np.save(
        output_dir / "weights.npy",
        np.asarray(
            weight_history,
            dtype=np.float32
        )
    )


# ============================================================
# 17. SAVE CLIENT ATTENTION WEIGHTS
# ============================================================

def save_client_attention(
    model,
    X_test,
    output_path
):

    model.eval()


    X_tensor = to_tensor(
        X_test
    ).to(DEVICE)


    attention_values = []


    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            BATCH_SIZE
        ):

            X_batch = X_tensor[
                start:start + BATCH_SIZE
            ]


            _, attention = model(
                X_batch,
                return_attn=True
            )


            attention_values.append(
                attention
                .cpu()
                .numpy()
            )


    attention_values = np.concatenate(
        attention_values,
        axis=0
    )


    np.save(
        output_path,
        attention_values
    )


# ============================================================
# 18. TRAIN ONE SUBGROUP
# ============================================================

def train_subgroup(
    subgroup_name,
    subgroup_config
):

    data_dir = subgroup_config[
        "data_dir"
    ]


    output_dir = subgroup_config[
        "output_dir"
    ]


    global_test_battery = subgroup_config[
        "global_test_battery"
    ]


    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    attention_dir = (
        output_dir
        / "attention_weights"
    )


    attention_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    print("\n")
    print("=" * 70)

    print(
        f"TRAINING SUBGROUP: "
        f"{subgroup_name}"
    )

    print("=" * 70)


    print(
        f"Data directory : "
        f"{data_dir}"
    )


    print(
        f"Output directory: "
        f"{output_dir}"
    )


    print(
        f"Global test    : "
        f"{global_test_battery}"
    )


    print(
        f"FedWeightedAvg ALPHA: "
        f"{ALPHA}"
    )


    # ========================================================
    # 18.1 LOAD CLIENT DATA
    # ========================================================

    client_data = {}


    for client_id in CLIENTS:

        client_file = (
            data_dir
            / f"{client_id}_data.npz"
        )


        print(
            f"\nLoading {client_id}: "
            f"{client_file}"
        )


        data = load_npz_data(
            client_file
        )


        required_keys = [
            "X_train",
            "y_train",
            "X_val",
            "y_val",
            "X_test",
            "y_test"
        ]


        for key in required_keys:

            if key not in data:

                raise KeyError(
                    f"{key} missing from "
                    f"{client_file}"
                )


        client_data[client_id] = {

            "X_train":
                data["X_train"],

            "y_train":
                data["y_train"],

            "X_val":
                data["X_val"],

            "y_val":
                data["y_val"],

            "X_test":
                data["X_test"],

            "y_test":
                data["y_test"]
        }


        print(
            f"  Train: "
            f"{data['X_train'].shape}"
        )


        print(
            f"  Val  : "
            f"{data['X_val'].shape}"
        )


        print(
            f"  Test : "
            f"{data['X_test'].shape}"
        )


    # ========================================================
    # 18.2 LOAD GLOBAL VALIDATION DATA
    #
    # global_val.npz contains:
    #
    #   X
    #   y
    #
    # NOT:
    #
    #   X_val
    #   y_val
    # ========================================================

    global_val_path = (
        data_dir
        / "global_val.npz"
    )


    print(
        f"\nLoading global validation:\n"
        f"{global_val_path}"
    )


    global_val_data = load_npz_data(
        global_val_path
    )


    if "X" not in global_val_data:

        raise KeyError(
            "X missing from global_val.npz"
        )


    if "y" not in global_val_data:

        raise KeyError(
            "y missing from global_val.npz"
        )


    X_global_val = global_val_data[
        "X"
    ]


    y_global_val = global_val_data[
        "y"
    ]


    print(
        f"Global validation: "
        f"{X_global_val.shape}"
    )


    # ========================================================
    # 18.3 LOAD GLOBAL TEST DATA
    #
    # global_test.npz contains:
    #
    #   X
    #   y
    #
    # NOT:
    #
    #   X_test
    #   y_test
    # ========================================================

    global_test_path = (
        data_dir
        / "global_test.npz"
    )


    print(
        f"\nLoading global test:\n"
        f"{global_test_path}"
    )


    global_test_data = load_npz_data(
        global_test_path
    )


    if "X" not in global_test_data:

        raise KeyError(
            "X missing from global_test.npz"
        )


    if "y" not in global_test_data:

        raise KeyError(
            "y missing from global_test.npz"
        )


    X_global_test = global_test_data[
        "X"
    ]


    y_global_test = global_test_data[
        "y"
    ]


    print(
        f"Global test: "
        f"{X_global_test.shape}"
    )


    # ========================================================
    # 18.4 INITIAL GLOBAL MODEL
    # ========================================================

    global_model = (
        CNNLSTMAttention()
        .to(DEVICE)
    )


    # ========================================================
    # 18.5 HISTORY
    # ========================================================

    rounds = []

    client_rmse_history = []

    weight_history = []

    global_val_history = []


    # ========================================================
    # 18.6 EARLY STOPPING
    # ========================================================

    best_global_rmse = float(
        "inf"
    )


    best_global_state = None


    best_round = 0


    patience_counter = 0


    stopped_round = MAX_ROUNDS


    # ========================================================
    # 18.7 FEDERATED TRAINING
    # ========================================================

    for round_number in range(
        1,
        MAX_ROUNDS + 1
    ):

        print("\n")
        print("-" * 70)


        print(
            f"ROUND "
            f"{round_number}/"
            f"{MAX_ROUNDS}"
        )


        print("-" * 70)


        # ----------------------------------------------------
        # Local models
        # ----------------------------------------------------

        client_models = []

        client_rmses = []


        # ----------------------------------------------------
        # Local training for each client
        # ----------------------------------------------------

        for client_id in CLIENTS:

            print(
                f"\nTraining "
                f"{client_id}..."
            )


            data = client_data[
                client_id
            ]


            local_model = train_local_model(
                global_model,
                data["X_train"],
                data["y_train"]
            )


            # ------------------------------------------------
            # Local validation
            # ------------------------------------------------

            val_mae, val_rmse = (
                evaluate_model(
                    local_model,
                    data["X_val"],
                    data["y_val"]
                )
            )


            print(
                f"{client_id} validation "
                f"MAE  = {val_mae:.6f}"
            )


            print(
                f"{client_id} validation "
                f"RMSE = {val_rmse:.6f}"
            )


            client_models.append(
                local_model
            )


            client_rmses.append(
                val_rmse
            )


        # ====================================================
        # FEDWEIGHTEDAVG
        # ====================================================

        weights = (
            calculate_fedweightedavg_weights(
                client_rmses
            )
        )


        print(
            "\nFedWeightedAvg weights:"
        )


        for i, client_id in enumerate(
            CLIENTS
        ):

            print(
                f"  {client_id}: "
                f"{weights[i]:.6f}"
            )


        # ====================================================
        # STORE HISTORY
        # ====================================================

        rounds.append(
            round_number
        )


        client_rmse_history.append(
            client_rmses
        )


        weight_history.append(
            weights
        )


        # ====================================================
        # AGGREGATE CLIENT MODELS
        # ========================================================

        aggregated_state = (
            fed_weighted_average(
                client_models,
                weights
            )
        )


        global_model.load_state_dict(
            aggregated_state
        )


        # ====================================================
        # GLOBAL VALIDATION
        # ========================================================

        global_val_mae, global_val_rmse = (
            evaluate_model(
                global_model,
                X_global_val,
                y_global_val
            )
        )


        print(
            "\nGlobal validation:"
        )


        print(
            f"  MAE  = "
            f"{global_val_mae:.6f}"
        )


        print(
            f"  RMSE = "
            f"{global_val_rmse:.6f}"
        )


        # ====================================================
        # STORE GLOBAL VALIDATION HISTORY
        # ====================================================

        global_val_history.append({

            "round":
                round_number,

            "global_val_mae":
                global_val_mae,

            "global_val_rmse":
                global_val_rmse,

            "client_1_val_rmse":
                client_rmses[0],

            "client_2_val_rmse":
                client_rmses[1],

            "client_3_val_rmse":
                client_rmses[2],

            "client_1_weight":
                weights[0],

            "client_2_weight":
                weights[1],

            "client_3_weight":
                weights[2]
        })


        # ====================================================
        # SAVE ARRAYS AFTER EVERY ROUND
        # ====================================================

        save_training_arrays(
            output_dir,
            rounds,
            client_rmse_history,
            weight_history
        )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            global_val_rmse
            <
            best_global_rmse
        ):

            best_global_rmse = (
                global_val_rmse
            )


            best_round = (
                round_number
            )


            best_global_state = (
                copy.deepcopy(
                    global_model.state_dict()
                )
            )


            patience_counter = 0


            print(
                "\n  *** New best global "
                "model ***"
            )


        else:

            patience_counter += 1


            print(
                f"\n  No improvement. "
                f"Patience: "
                f"{patience_counter}/"
                f"{PATIENCE}"
            )


        if (
            patience_counter
            >=
            PATIENCE
        ):

            stopped_round = (
                round_number
            )


            print(
                "\nEarly stopping triggered."
            )


            break


    # ========================================================
    # 18.8 RESTORE BEST GLOBAL MODEL
    # ========================================================

    if best_global_state is None:

        raise RuntimeError(
            "No best global model was obtained."
        )


    global_model.load_state_dict(
        best_global_state
    )


    global_model = global_model.to(
        DEVICE
    )


    # ========================================================
    # 18.9 SAVE GLOBAL MODEL
    # ========================================================

    global_model_path = (
        output_dir
        / "global_model.pt"
    )


    torch.save(
        global_model.state_dict(),
        global_model_path
    )


    print(
        f"\nBest global model saved:\n"
        f"{global_model_path}"
    )


    # ========================================================
    # 18.10 SAVE GLOBAL VALIDATION HISTORY
    # ========================================================

    global_val_df = pd.DataFrame(
        global_val_history
    )


    global_val_path = (
        output_dir
        / "global_val_history.csv"
    )


    global_val_df.to_csv(
        global_val_path,
        index=False
    )


    print(
        f"Global validation history saved:\n"
        f"{global_val_path}"
    )


    # ========================================================
    # 18.11 FINAL GLOBAL TEST
    # ========================================================

    global_test_mae, global_test_rmse = (
        evaluate_model(
            global_model,
            X_global_test,
            y_global_test
        )
    )


    print("\n")
    print("=" * 70)
    print("FINAL GLOBAL TEST")
    print("=" * 70)


    print(
        f"Battery: "
        f"{global_test_battery}"
    )


    print(
        f"MAE  : "
        f"{global_test_mae:.6f}"
    )


    print(
        f"RMSE : "
        f"{global_test_rmse:.6f}"
    )


    # ========================================================
    # 18.12 SAVE GLOBAL TEST METRICS
    # ========================================================

    global_metrics_path = (
        output_dir
        / "global_test_metrics.txt"
    )


    with open(
        global_metrics_path,
        "w"
    ) as f:

        f.write(
            "FINAL GLOBAL TEST RESULTS\n"
        )

        f.write(
            "=========================\n"
        )

        f.write(
            f"Subgroup: "
            f"{subgroup_name}\n"
        )

        f.write(
            f"Global test battery: "
            f"{global_test_battery}\n"
        )

        f.write(
            f"FedWeightedAvg Alpha: "
            f"{ALPHA}\n"
        )

        f.write(
            f"MAE: "
            f"{global_test_mae:.6f}\n"
        )

        f.write(
            f"RMSE: "
            f"{global_test_rmse:.6f}\n"
        )

        f.write(
            f"Best round: "
            f"{best_round}\n"
        )

        f.write(
            f"Best validation RMSE: "
            f"{best_global_rmse:.6f}\n"
        )


    # ========================================================
    # 18.13 FINAL LOCAL CLIENT TEST
    #
    # Best global model is evaluated separately on each
    # client's local test set.
    # ========================================================

    local_test_results = []


    for client_id in CLIENTS:

        data = client_data[
            client_id
        ]


        test_mae, test_rmse = (
            evaluate_model(
                global_model,
                data["X_test"],
                data["y_test"]
            )
        )


        print(
            f"\n{client_id} local test:"
        )


        print(
            f"  MAE  = "
            f"{test_mae:.6f}"
        )


        print(
            f"  RMSE = "
            f"{test_rmse:.6f}"
        )


        local_test_results.append({

            "client":
                client_id,

            "test_mae":
                test_mae,

            "test_rmse":
                test_rmse
        })


    # ========================================================
    # 18.14 SAVE LOCAL TEST METRICS
    # ========================================================

    local_test_df = pd.DataFrame(
        local_test_results
    )


    local_test_path = (
        output_dir
        / "local_test_metrics.csv"
    )


    local_test_df.to_csv(
        local_test_path,
        index=False
    )


    # ========================================================
    # 18.15 SAVE CLIENT ATTENTION WEIGHTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SAVING CLIENT ATTENTION WEIGHTS")
    print("=" * 70)


    for client_id in CLIENTS:

        data = client_data[
            client_id
        ]


        attention_path = (
            attention_dir
            / f"{client_id}_attention.npy"
        )


        save_client_attention(
            global_model,
            data["X_test"],
            attention_path
        )


        print(
            f"{client_id}: "
            f"{attention_path}"
        )


    # ========================================================
    # 18.16 SAVE CONVERGENCE INFORMATION
    # ========================================================

    convergence_path = (
        output_dir
        / "convergence.txt"
    )


    total_rounds = len(
        rounds
    )


    if (
        stopped_round
        <
        MAX_ROUNDS
    ):

        stopping_reason = (
            "Early stopping triggered "
            f"after {PATIENCE} rounds "
            "without validation improvement."
        )

    else:

        stopping_reason = (
            "Training reached MAX_ROUNDS."
        )


    with open(
        convergence_path,
        "w"
    ) as f:

        f.write(
            "FEDERATED TRAINING CONVERGENCE\n"
        )

        f.write(
            "===============================\n"
        )

        f.write(
            f"Subgroup: "
            f"{subgroup_name}\n"
        )

        f.write(
            f"FedWeightedAvg Alpha: "
            f"{ALPHA}\n"
        )

        f.write(
            f"Total rounds completed: "
            f"{total_rounds}\n"
        )

        f.write(
            f"Maximum rounds: "
            f"{MAX_ROUNDS}\n"
        )

        f.write(
            f"Best round: "
            f"{best_round}\n"
        )

        f.write(
            f"Best global validation RMSE: "
            f"{best_global_rmse:.6f}\n"
        )

        f.write(
            f"Stopped at round: "
            f"{stopped_round}\n"
        )

        f.write(
            f"Patience: "
            f"{PATIENCE}\n"
        )

        f.write(
            f"Stopping reason: "
            f"{stopping_reason}\n"
        )


    # ========================================================
    # 18.17 FINAL ARRAY SAVE
    # ========================================================

    save_training_arrays(
        output_dir,
        rounds,
        client_rmse_history,
        weight_history
    )


    # ========================================================
    # 18.18 VERIFY OUTPUT FILES
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        f"SUBGROUP {subgroup_name} COMPLETE"
    )

    print("=" * 70)


    expected_files = [

        "client_rmse.npy",

        "convergence.txt",

        "global_model.pt",

        "global_test_metrics.txt",

        "global_val_history.csv",

        "local_test_metrics.csv",

        "rounds.npy",

        "weights.npy"
    ]


    for filename in expected_files:

        path = (
            output_dir
            / filename
        )


        status = (
            "OK"
            if path.exists()
            else "MISSING"
        )


        print(
            f"{filename}: {status}"
        )


    # ========================================================
    # 18.19 VERIFY ATTENTION FILES
    # ========================================================

    print(
        "\nAttention files:"
    )


    for client_id in CLIENTS:

        filename = (
            f"{client_id}_attention.npy"
        )


        path = (
            attention_dir
            / filename
        )


        status = (
            "OK"
            if path.exists()
            else "MISSING"
        )


        print(
            f"{filename}: {status}"
        )


    # ========================================================
    # 18.20 CHECK ARRAY SHAPES
    # ========================================================

    rounds_array = np.load(
        output_dir
        / "rounds.npy"
    )


    rmse_array = np.load(
        output_dir
        / "client_rmse.npy"
    )


    weights_array = np.load(
        output_dir
        / "weights.npy"
    )


    print(
        "\nSaved array shapes:"
    )


    print(
        f"rounds.npy      : "
        f"{rounds_array.shape}"
    )


    print(
        f"client_rmse.npy : "
        f"{rmse_array.shape}"
    )


    print(
        f"weights.npy     : "
        f"{weights_array.shape}"
    )


    # ========================================================
    # 18.21 WEIGHT SANITY CHECK
    # ========================================================

    if len(weights_array) > 0:

        weight_sums = np.sum(
            weights_array,
            axis=1
        )


        print(
            f"\nWeight sum range: "
            f"{weight_sums.min():.6f} - "
            f"{weight_sums.max():.6f}"
        )


        if np.allclose(
            weight_sums,
            1.0,
            atol=1e-5
        ):

            print(
                "Weight check: PASSED"
            )

        else:

            print(
                "Weight check: FAILED"
            )


    print("\n")


# ============================================================
# 19. MAIN
# ============================================================

def main():

    print("\n")
    print("#" * 70)

    print(
        "# STARTING ALL FEDERATED "
        "LEARNING EXPERIMENTS"
    )

    print("#" * 70)


    print(
        f"\nFedWeightedAvg ALPHA = {ALPHA}"
    )


    print(
        "\nOutput directories:"
    )


    print(
        f"  Subgroup 5_18 : "
        f"{OUTPUT_ROOT_5_18}"
    )


    print(
        f"  Subgroup 25_28: "
        f"{OUTPUT_ROOT_25_28}"
    )


    for subgroup_name, subgroup_config in (
        SUBGROUPS.items()
    ):

        train_subgroup(
            subgroup_name,
            subgroup_config
        )


    print("\n")
    print("#" * 70)

    print(
        "# ALL SUBGROUPS COMPLETED"
    )

    print("#" * 70)


    print(
        "\nOutputs saved under:"
    )

    print(
        f"  {OUTPUT_ROOT_5_18}"
    )

    print(
        f"  {OUTPUT_ROOT_25_28}"
    )


# ============================================================
# 20. RUN
# ============================================================

if __name__ == "__main__":

    main()
