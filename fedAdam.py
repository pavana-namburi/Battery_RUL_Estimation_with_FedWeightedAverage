# ============================================================
# FEDERATED LEARNING TRAINING
# CNN-LSTM-ATTENTION + FedAdam
#
# Automatically detects all processed_data_* subgroup folders.
#
# FedAdam logic preserved:
#
# 1. Each client starts from the current global model.
# 2. Each client performs local Adam training.
# 3. Client models are averaged using STANDARD FedAvg mean.
# 4. Server computes:
#
#       delta = current_global - averaged_client_model
#
# 5. FedAdam first and second moments are updated.
# 6. Server model is updated using FedAdam.
#
# The additional code only provides:
# - automatic subgroup discovery
# - validation/test evaluation
# - best-model restoration
# - convergence information
# - training arrays
# - local/global metrics
# - attention weights
# - output verification
#
# NO absolute Windows paths are used.
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

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# 1. REPRODUCIBILITY
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
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("FEDERATED LEARNING TRAINING - FedAdam")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print("=" * 70)


# ============================================================
# 3. AUTOMATIC PATH DISCOVERY
# ============================================================
#
# The script does NOT contain:
#
# D:\reproducing_FL
#
# or any other absolute path.
#
# It searches relative to the location of this Python file.
#
# Expected structure can be either:
#
# project/
# ├── FL_training_alpha_analysis.py
# ├── processed_data/
# │   ├── processed_data_5_18/
# │   └── processed_data_25_28/
#
# OR:
#
# project/
# ├── FL_training_alpha_analysis.py
# ├── processed_data_5_18/
# └── processed_data_25_28/
#
# The code checks both automatically.
# ============================================================


SCRIPT_DIR = Path(
    __file__
).resolve().parent


def discover_subgroup_directories():

    candidates = []

    # --------------------------------------------------------
    # Possible roots
    # --------------------------------------------------------

    possible_roots = [

        SCRIPT_DIR,

        SCRIPT_DIR / "processed_data",

        SCRIPT_DIR.parent,

        SCRIPT_DIR.parent / "processed_data"
    ]

    # Remove duplicates while preserving order

    unique_roots = []

    for root in possible_roots:

        root = root.resolve()

        if root not in unique_roots:

            unique_roots.append(root)


    # --------------------------------------------------------
    # Search for processed_data_* directories
    # --------------------------------------------------------

    for root in unique_roots:

        if not root.exists():
            continue

        try:

            for directory in root.iterdir():

                if not directory.is_dir():
                    continue

                name = directory.name.lower()

                if name.startswith(
                    "processed_data_"
                ):

                    candidates.append(
                        directory.resolve()
                    )

        except PermissionError:

            continue


    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    discovered = []

    for directory in candidates:

        if directory not in discovered:

            discovered.append(directory)


    # --------------------------------------------------------
    # Validate that directories contain
    # the expected client files
    # --------------------------------------------------------

    valid = []

    for directory in discovered:

        required_client_files = [

            directory / "client_1_data.npz",

            directory / "client_2_data.npz",

            directory / "client_3_data.npz"
        ]

        if all(
            file.exists()
            for file in required_client_files
        ):

            valid.append(directory)


    return sorted(
        valid,
        key=lambda p: p.name
    )


DATA_DIRECTORIES = (
    discover_subgroup_directories()
)


if len(DATA_DIRECTORIES) == 0:

    raise FileNotFoundError(
        "\nNo valid processed_data_* subgroup "
        "directories were found.\n\n"
        "The script searched relative to:\n"
        f"{SCRIPT_DIR}\n\n"
        "Expected folders such as:\n"
        "processed_data_5_18\n"
        "processed_data_25_28\n"
        "or those folders inside a processed_data directory."
    )


# ------------------------------------------------------------
# Output root
# ------------------------------------------------------------
#
# Output is created next to the detected data root.
#
# Example:
#
# project/
# ├── processed_data_5_18/
# ├── processed_data_25_28/
# └── outputs_fedadam/
#
# If the data folders are inside processed_data/, output is
# also placed beside that data root.
# ------------------------------------------------------------


COMMON_DATA_ROOT = (
    DATA_DIRECTORIES[0].parent
)


OUTPUT_ROOT = (
    COMMON_DATA_ROOT / "outputs_fedadam"
)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. CLIENT CONFIGURATION
# ============================================================

CLIENTS = [

    "client_1",
    "client_2",
    "client_3"
]


# ============================================================
# 5. TRAINING HYPERPARAMETERS
# ============================================================

LOCAL_EPOCHS = 5

BATCH_SIZE = 64

LEARNING_RATE = 1e-4

MAX_ROUNDS = 50

PATIENCE = 5


# ============================================================
# 6. FEDADAM PARAMETERS
# ============================================================

SERVER_LR = 1e-3

BETA1 = 0.9

BETA2 = 0.99

TAU = 1e-3


# ============================================================
# 7. MODEL
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
        # ----------------------------------------------------

        x = x.permute(
            0,
            2,
            1
        )

        x = self.relu(
            self.conv(x)
        )

        x = x.permute(
            0,
            2,
            1
        )

        out, _ = self.lstm(x)

        attention_scores = self.attn(
            out
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        context = torch.sum(
            attention_weights * out,
            dim=1
        )

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
# 8. METRIC CALCULATION
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
# 9. LOAD NPZ DATA
# ============================================================

def load_npz_data(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return np.load(
        path,
        allow_pickle=True
    )


# ============================================================
# 10. TENSOR CONVERSION
# ============================================================

def to_tensor(x):

    return torch.tensor(
        np.asarray(x),
        dtype=torch.float32
    )


# ============================================================
# 11. LOCAL TRAINING
# ============================================================

def train_local_model(
    global_model,
    X_train,
    y_train
):

    # --------------------------------------------------------
    # IMPORTANT:
    # This is the same local-training logic as the original
    # FedAdam code.
    # --------------------------------------------------------

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
# 12. MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y
):

    model.eval()

    X_tensor = to_tensor(
        X
    )

    y_true = np.asarray(
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
            ].to(DEVICE)

            batch_predictions = model(
                X_batch
            )

            predictions.append(
                batch_predictions
                .cpu()
                .numpy()
            )

    if len(predictions) == 0:

        raise ValueError(
            "Cannot evaluate model: "
            "dataset contains zero samples."
        )

    predictions = np.concatenate(
        predictions
    )

    return calculate_metrics(
        y_true,
        predictions
    )


# ============================================================
# 13. MODEL PREDICTIONS + ATTENTION
# ============================================================

def predict_with_attention(
    model,
    X
):

    model.eval()

    X_tensor = to_tensor(
        X
    )

    predictions = []

    attention_values = []

    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            BATCH_SIZE
        ):

            X_batch = X_tensor[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            batch_predictions, batch_attention = (
                model(
                    X_batch,
                    return_attn=True
                )
            )

            predictions.append(
                batch_predictions
                .cpu()
                .numpy()
            )

            attention_values.append(
                batch_attention
                .cpu()
                .numpy()
            )

    predictions = np.concatenate(
        predictions
    )

    attention_values = np.concatenate(
        attention_values,
        axis=0
    )

    return (
        predictions,
        attention_values
    )


# ============================================================
# 14. STANDARD FEDAVG MEAN
# ============================================================
#
# IMPORTANT:
#
# FedAdam uses a standard mean of client models before the
# server-side Adam update.
#
# This is NOT FedWeightedAvg.
#
# No RMSE-based client weighting is introduced here.
# ============================================================

def standard_fedavg_mean(
    client_models
):

    state_dicts = [

        model.state_dict()

        for model in client_models
    ]

    global_state = copy.deepcopy(
        state_dicts[0]
    )

    for key in global_state.keys():

        global_state[key] = torch.stack(

            [

                state_dicts[i][key].float()

                for i in range(
                    len(client_models)
                )

            ],

            dim=0

        ).mean(
            dim=0
        )

    return global_state


# ============================================================
# 15. SAVE TRAINING ARRAYS
# ============================================================

def save_training_arrays(
    output_dir,
    rounds,
    client_rmse_history,
    weight_history
):

    output_dir = Path(
        output_dir
    )

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
# 16. SAVE CLIENT ATTENTION
# ============================================================

def save_client_attention(
    model,
    X_test,
    output_path
):

    _, attention_values = (
        predict_with_attention(
            model,
            X_test
        )
    )

    np.save(
        output_path,
        attention_values
    )


# ============================================================
# 17. TRAIN ONE SUBGROUP
# ============================================================

def train_subgroup(
    subgroup_name,
    data_dir
):

    data_dir = Path(
        data_dir
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT /
        f"output_{subgroup_name}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    attention_dir = (
        output_dir /
        "attention_weights"
    )

    attention_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    print("\n")
    print("=" * 70)
    print(
        f"TRAINING SUBGROUP: {subgroup_name}"
    )
    print("=" * 70)

    print(
        f"Data directory  : {data_dir}"
    )

    print(
        f"Output directory: {output_dir}"
    )


    # ========================================================
    # 17.1 LOAD CLIENT DATA
    # ========================================================

    client_data = {}

    for client_id in CLIENTS:

        client_file = (
            data_dir /
            f"{client_id}_data.npz"
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
    # 17.2 LOAD GLOBAL VALIDATION DATA
    # ========================================================

    global_val_path = (
        data_dir /
        "global_val.npz"
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

    X_global_val = (
        global_val_data["X"]
    )

    y_global_val = (
        global_val_data["y"]
    )

    print(
        f"Global validation: "
        f"{X_global_val.shape}"
    )


    # ========================================================
    # 17.3 LOAD GLOBAL TEST DATA
    # ========================================================

    global_test_path = (
        data_dir /
        "global_test.npz"
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

    X_global_test = (
        global_test_data["X"]
    )

    y_global_test = (
        global_test_data["y"]
    )

    print(
        f"Global test: "
        f"{X_global_test.shape}"
    )


    # ========================================================
    # 17.4 INITIAL GLOBAL MODEL
    # ========================================================

    global_model = (
        CNNLSTMAttention()
        .to(DEVICE)
    )


    # ========================================================
    # 17.5 FEDADAM MOMENT STATES
    # ========================================================
    #
    # These are reset independently for each subgroup.
    #
    # This preserves the original FedAdam logic.
    # ========================================================

    m = {}

    v = {}

    for key, parameter in (
        global_model.state_dict().items()
    ):

        m[key] = torch.zeros_like(
            parameter,
            dtype=torch.float32
        )

        v[key] = torch.zeros_like(
            parameter,
            dtype=torch.float32
        )


    # ========================================================
    # 17.6 TRAINING HISTORY
    # ========================================================

    rounds = []

    client_rmse_history = []

    weight_history = []

    global_val_history = []


    # ========================================================
    # 17.7 EARLY STOPPING
    # ========================================================

    best_global_rmse = float(
        "inf"
    )

    best_global_state = None

    best_round = 0

    patience_counter = 0

    stopped_round = MAX_ROUNDS


    # ========================================================
    # 17.8 FEDADAM TRAINING
    # ========================================================

    for round_number in range(
        1,
        MAX_ROUNDS + 1
    ):

        print("\n")
        print("-" * 70)

        print(
            f"ROUND "
            f"{round_number}/{MAX_ROUNDS}"
        )

        print("-" * 70)


        # ----------------------------------------------------
        # LOCAL MODELS
        # ----------------------------------------------------

        client_models = []

        client_rmses = []


        # ----------------------------------------------------
        # LOCAL TRAINING
        # ----------------------------------------------------

        for client_id in CLIENTS:

            print(
                f"\nTraining {client_id}..."
            )

            data = client_data[
                client_id
            ]

            local_model = (
                train_local_model(
                    global_model,
                    data["X_train"],
                    data["y_train"]
                )
            )


            # ------------------------------------------------
            # Local validation
            #
            # This is ONLY recorded for analysis.
            #
            # It does NOT determine FedAdam aggregation.
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
        # STANDARD FEDAVG MEAN
        # ====================================================
        #
        # IMPORTANT:
        #
        # FedAdam first calculates an ordinary mean of all
        # client models.
        #
        # Therefore the coefficients are:
        #
        # client_1 = 1/3
        # client_2 = 1/3
        # client_3 = 1/3
        #
        # These are saved in weights.npy only to document the
        # actual aggregation coefficients.
        #
        # No RMSE weighting is performed.
        # ====================================================

        num_clients = len(
            client_models
        )

        aggregation_weights = (
            np.ones(
                num_clients,
                dtype=np.float32
            )
            /
            num_clients
        )

        print(
            "\nFedAdam client averaging "
            "(standard mean):"
        )

        for i, client_id in enumerate(
            CLIENTS
        ):

            print(
                f"  {client_id}: "
                f"{aggregation_weights[i]:.6f}"
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
            aggregation_weights
        )


        # ====================================================
        # STANDARD FEDAVG MEAN
        # ====================================================

        avg_state = (
            standard_fedavg_mean(
                client_models
            )
        )


        # ====================================================
        # FEDADAM SERVER UPDATE
        # ====================================================
        #
        # ORIGINAL LOGIC PRESERVED:
        #
        # delta =
        #     current_global - avg_state
        #
        # m =
        #     BETA1*m + (1-BETA1)*delta
        #
        # v =
        #     BETA2*v + (1-BETA2)*(delta^2)
        #
        # new_global =
        #     current_global
        #     - SERVER_LR*m/(sqrt(v)+TAU)
        # ====================================================

        current_global = (
            global_model.state_dict()
        )

        new_state = copy.deepcopy(
            current_global
        )

        for key in current_global.keys():

            delta = (
                current_global[key].float()
                -
                avg_state[key]
            )

            # ------------------------------------------------
            # First moment
            # ------------------------------------------------

            m[key] = (

                BETA1 * m[key]

                +
                (1 - BETA1) * delta
            )


            # ------------------------------------------------
            # Second moment
            # ------------------------------------------------

            v[key] = (

                BETA2 * v[key]

                +
                (1 - BETA2)
                * (delta ** 2)
            )


            # ------------------------------------------------
            # Adam server update
            # ------------------------------------------------

            new_state[key] = (

                current_global[key]

                -
                SERVER_LR

                *
                m[key]

                /
                (
                    torch.sqrt(
                        v[key]
                    )
                    +
                    TAU
                )
            )


        global_model.load_state_dict(
            new_state
        )


        # ====================================================
        # GLOBAL VALIDATION
        # ====================================================

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

        history_row = {

            "round":
                round_number,

            "global_val_mae":
                global_val_mae,

            "global_val_rmse":
                global_val_rmse
        }


        # Add client validation RMSE dynamically
        # rather than hardcoding client names.

        for i, client_id in enumerate(
            CLIENTS
        ):

            history_row[
                f"{client_id}_val_rmse"
            ] = client_rmses[i]

            history_row[
                f"{client_id}_aggregation_weight"
            ] = aggregation_weights[i]


        global_val_history.append(
            history_row
        )


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

        if global_val_rmse < best_global_rmse:

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

        if patience_counter >= PATIENCE:

            stopped_round = (
                round_number
            )

            print(
                "\nEarly stopping triggered."
            )

            break


    # ========================================================
    # 17.9 RESTORE BEST GLOBAL MODEL
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
    # 17.10 SAVE GLOBAL MODEL
    # ========================================================

    global_model_path = (
        output_dir /
        "global_model.pt"
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
    # 17.11 SAVE GLOBAL VALIDATION HISTORY
    # ========================================================

    global_val_df = pd.DataFrame(
        global_val_history
    )

    global_val_path = (
        output_dir /
        "global_val_history.csv"
    )

    global_val_df.to_csv(
        global_val_path,
        index=False
    )


    # ========================================================
    # 17.12 FINAL GLOBAL TEST
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
        f"Subgroup: "
        f"{subgroup_name}"
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
    # 17.13 SAVE GLOBAL TEST METRICS
    # ========================================================

    global_metrics_path = (
        output_dir /
        "global_test_metrics.txt"
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
            f"Global test samples: "
            f"{len(y_global_test)}\n"
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
    # 17.14 FINAL LOCAL CLIENT TEST
    #
    # Best global model is evaluated on every client's local
    # test set.
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
    # 17.15 SAVE LOCAL TEST METRICS
    # ========================================================

    local_test_df = pd.DataFrame(
        local_test_results
    )

    local_test_path = (
        output_dir /
        "local_test_metrics.csv"
    )

    local_test_df.to_csv(
        local_test_path,
        index=False
    )


    # ========================================================
    # 17.16 SAVE CLIENT ATTENTION WEIGHTS
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
            attention_dir /
            f"{client_id}_attention.npy"
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
    # 17.17 SAVE CONVERGENCE INFORMATION
    # ========================================================

    convergence_path = (
        output_dir /
        "convergence.txt"
    )

    total_rounds = len(
        rounds
    )


    if stopped_round < MAX_ROUNDS:

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
            "Algorithm: FedAdam\n"
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
            f"Server learning rate: "
            f"{SERVER_LR}\n"
        )

        f.write(
            f"Beta1: "
            f"{BETA1}\n"
        )

        f.write(
            f"Beta2: "
            f"{BETA2}\n"
        )

        f.write(
            f"Tau: "
            f"{TAU}\n"
        )

        f.write(
            f"Stopping reason: "
            f"{stopping_reason}\n"
        )


    # ========================================================
    # 17.18 FINAL ARRAY SAVE
    # ========================================================

    save_training_arrays(

        output_dir,

        rounds,

        client_rmse_history,

        weight_history
    )


    # ========================================================
    # 17.19 VERIFY OUTPUT FILES
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
            output_dir /
            filename
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
    # 17.20 VERIFY ATTENTION FILES
    # ========================================================

    print(
        "\nAttention files:"
    )

    for client_id in CLIENTS:

        filename = (
            f"{client_id}_attention.npy"
        )

        path = (
            attention_dir /
            filename
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
    # 17.21 CHECK ARRAY SHAPES
    # ========================================================

    rounds_array = np.load(
        output_dir /
        "rounds.npy"
    )

    rmse_array = np.load(
        output_dir /
        "client_rmse.npy"
    )

    weights_array = np.load(
        output_dir /
        "weights.npy"
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
    # 17.22 WEIGHT SANITY CHECK
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


    # ========================================================
    # 17.23 FINAL OUTPUT SUMMARY
    # ========================================================

    print("\n")
    print(
        "Output files:"
    )

    for path in sorted(
        output_dir.rglob("*")
    ):

        if path.is_file():

            print(
                f"  {path.relative_to(output_dir)}"
            )

    print("\n")


# ============================================================
# 18. MAIN
# ============================================================

def main():

    print("\n")
    print("#" * 70)
    print(
        "# STARTING ALL FEDADAM "
        "FEDERATED LEARNING EXPERIMENTS"
    )
    print("#" * 70)

    print(
        "\nDetected subgroup directories:"
    )

    for directory in DATA_DIRECTORIES:

        print(
            f"  {directory}"
        )


    # --------------------------------------------------------
    # Train every discovered subgroup
    # --------------------------------------------------------

    for data_directory in DATA_DIRECTORIES:

        directory_name = (
            data_directory.name
        )

        # ----------------------------------------------------
        # processed_data_5_18
        #              ↓
        # subgroup_name = 5_18
        #
        # processed_data_25_28
        #              ↓
        # subgroup_name = 25_28
        # ----------------------------------------------------

        prefix = "processed_data_"

        if directory_name.startswith(
            prefix
        ):

            subgroup_name = (
                directory_name[
                    len(prefix):
                ]
            )

        else:

            subgroup_name = (
                directory_name
            )


        train_subgroup(

            subgroup_name,

            data_directory
        )


    # ========================================================
    # ALL COMPLETE
    # ========================================================

    print("\n")
    print("#" * 70)
    print(
        "# ALL SUBGROUPS COMPLETED"
    )
    print("#" * 70)

    print(
        f"\nOutputs saved under:\n"
        f"{OUTPUT_ROOT}"
    )


# ============================================================
# 19. RUN
# ============================================================

if __name__ == "__main__":

    main()
