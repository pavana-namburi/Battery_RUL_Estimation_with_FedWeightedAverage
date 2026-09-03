# ============================================================
# FEDERATED LEARNING ABLATION STUDY
# LSTM-ONLY + FedWeightedAvg
#
# Runs BOTH subgroups:
#
#   5_18  -> B0005, B0006, B0007 | Global test B0018
#   25_28 -> B0025, B0026, B0027 | Global test B0028
#
# MODEL:
#   LSTM -> Fully Connected
#
# No CNN.
# No attention.
#
# Existing baseline outputs are NOT modified.
#
# New outputs:
#   ablation_results/LSTM_only/5_18/
#   ablation_results/LSTM_only/25_28/
#
# If an output directory already contains anything,
# the experiment REFUSES TO RUN.
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
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("LSTM-ONLY FEDERATED LEARNING ABLATION")
print("=" * 70)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print("=" * 70)


# ============================================================
# 3. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_ROOT = (
    BASE_DIR
    / "processed_data"
)

ABLATION_ROOT = (
    BASE_DIR
    / "ablation_results"
    / "LSTM_only"
)


# ============================================================
# 4. SUBGROUPS
# ============================================================

SUBGROUPS = {

    "5_18": {

        "data_dir":
            PROCESSED_ROOT
            / "processed_data_5_18",

        "output_dir":
            ABLATION_ROOT
            / "5_18",

        "global_test_battery":
            "B0018"
    },

    "25_28": {

        "data_dir":
            PROCESSED_ROOT
            / "processed_data_25_28",

        "output_dir":
            ABLATION_ROOT
            / "25_28",

        "global_test_battery":
            "B0028"
    }
}


# ============================================================
# 5. CLIENTS
# ============================================================

CLIENTS = [
    "client_1",
    "client_2",
    "client_3"
]


# ============================================================
# 6. HYPERPARAMETERS
# ============================================================

LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

MAX_ROUNDS = 50
PATIENCE = 5

ALPHA = 5.0


# ============================================================
# 7. LSTM-ONLY MODEL
#
# Input:
#   [batch, sequence_length, 7]
#
# Architecture:
#   LSTM -> final hidden state -> FC
#
# No CNN.
# No attention.
# ============================================================

class LSTMOnly(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=7,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(
            64,
            1
        )

    def forward(self, x):

        out, _ = self.lstm(
            x
        )

        # Last time step

        last_output = out[:, -1, :]

        prediction = self.fc(
            last_output
        ).squeeze(-1)

        return prediction


# ============================================================
# 8. METRICS
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
# 9. LOAD NPZ
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
# 10. TENSOR
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
# 12. EVALUATION
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

            predictions.append(
                model(
                    X_batch
                ).cpu().numpy()
            )

    predictions = np.concatenate(
        predictions
    )

    return calculate_metrics(
        y_tensor.numpy(),
        predictions
    )


# ============================================================
# 13. FEDWEIGHTEDAVG WEIGHTS
# ============================================================

def calculate_fedweightedavg_weights(
    client_rmses
):

    client_rmses = np.asarray(
        client_rmses,
        dtype=np.float64
    )

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

        return (
            np.ones(
                len(client_rmses)
            )
            /
            len(client_rmses)
        )

    return (
        exp_weights
        /
        weight_sum
    )


# ============================================================
# 14. FEDERATED AVERAGING
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
# 15. SAVE ARRAYS
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
# 16. SAFETY CHECK
# ============================================================

def prepare_output_directory(
    output_dir
):

    if output_dir.exists():

        contents = list(
            output_dir.iterdir()
        )

        if len(contents) > 0:

            raise RuntimeError(
                "\n"
                + "!" * 70
                + "\n"
                + "REFUSING TO RUN\n"
                + "OUTPUT DIRECTORY IS NOT EMPTY\n"
                + "\n"
                + f"{output_dir}\n"
                + "\n"
                + "Existing files will NOT be deleted "
                  "or overwritten.\n"
                + "!" * 70
            )

    else:

        output_dir.mkdir(
            parents=True,
            exist_ok=False
        )


# ============================================================
# 17. TRAIN ONE SUBGROUP
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

    print("\n")
    print("=" * 70)
    print(
        f"LSTM-ONLY | SUBGROUP: {subgroup_name}"
    )
    print("=" * 70)

    print(
        f"Data directory   : {data_dir}"
    )

    print(
        f"Output directory : {output_dir}"
    )

    print(
        f"Global test      : {global_test_battery}"
    )

    prepare_output_directory(
        output_dir
    )


    # ========================================================
    # LOAD CLIENT DATA
    # ========================================================

    client_data = {}

    for client_id in CLIENTS:

        client_file = (
            data_dir
            / f"{client_id}_data.npz"
        )

        print(
            f"\nLoading {client_id}:"
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

            "X_train": data["X_train"],
            "y_train": data["y_train"],

            "X_val": data["X_val"],
            "y_val": data["y_val"],

            "X_test": data["X_test"],
            "y_test": data["y_test"]
        }

        print(
            f"  Train: {data['X_train'].shape}"
        )

        print(
            f"  Val  : {data['X_val'].shape}"
        )

        print(
            f"  Test : {data['X_test'].shape}"
        )


    # ========================================================
    # GLOBAL VALIDATION
    # ========================================================

    global_val_data = load_npz_data(
        data_dir / "global_val.npz"
    )

    if "X" not in global_val_data:
        raise KeyError(
            "X missing from global_val.npz"
        )

    if "y" not in global_val_data:
        raise KeyError(
            "y missing from global_val.npz"
        )

    X_global_val = global_val_data["X"]
    y_global_val = global_val_data["y"]


    # ========================================================
    # GLOBAL TEST
    # ========================================================

    global_test_data = load_npz_data(
        data_dir / "global_test.npz"
    )

    if "X" not in global_test_data:
        raise KeyError(
            "X missing from global_test.npz"
        )

    if "y" not in global_test_data:
        raise KeyError(
            "y missing from global_test.npz"
        )

    X_global_test = global_test_data["X"]
    y_global_test = global_test_data["y"]


    # ========================================================
    # INITIAL MODEL
    # ========================================================

    global_model = LSTMOnly().to(
        DEVICE
    )


    # ========================================================
    # HISTORY
    # ========================================================

    rounds = []
    client_rmse_history = []
    weight_history = []
    global_val_history = []


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    best_global_rmse = float("inf")
    best_global_state = None
    best_round = 0

    patience_counter = 0
    stopped_round = MAX_ROUNDS


    # ========================================================
    # FEDERATED TRAINING
    # ========================================================

    for round_number in range(
        1,
        MAX_ROUNDS + 1
    ):

        print("\n")
        print("-" * 70)
        print(
            f"ROUND {round_number}/{MAX_ROUNDS}"
        )
        print("-" * 70)

        client_models = []
        client_rmses = []


        for client_id in CLIENTS:

            print(
                f"\nTraining {client_id}..."
            )

            data = client_data[
                client_id
            ]

            local_model = train_local_model(
                global_model,
                data["X_train"],
                data["y_train"]
            )

            val_mae, val_rmse = evaluate_model(
                local_model,
                data["X_val"],
                data["y_val"]
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


        weights = calculate_fedweightedavg_weights(
            client_rmses
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


        rounds.append(
            round_number
        )

        client_rmse_history.append(
            client_rmses
        )

        weight_history.append(
            weights
        )


        aggregated_state = fed_weighted_average(
            client_models,
            weights
        )

        global_model.load_state_dict(
            aggregated_state
        )


        global_val_mae, global_val_rmse = evaluate_model(
            global_model,
            X_global_val,
            y_global_val
        )

        print(
            "\nGlobal validation:"
        )

        print(
            f"  MAE  = {global_val_mae:.6f}"
        )

        print(
            f"  RMSE = {global_val_rmse:.6f}"
        )


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


        save_training_arrays(
            output_dir,
            rounds,
            client_rmse_history,
            weight_history
        )


        if global_val_rmse < best_global_rmse:

            best_global_rmse = global_val_rmse
            best_round = round_number

            best_global_state = copy.deepcopy(
                global_model.state_dict()
            )

            patience_counter = 0

            print(
                "\n*** New best global model ***"
            )

        else:

            patience_counter += 1

            print(
                f"\nNo improvement. "
                f"Patience: "
                f"{patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:

            stopped_round = round_number

            print(
                "\nEarly stopping triggered."
            )

            break


    # ========================================================
    # RESTORE BEST MODEL
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
    # SAVE MODEL
    # ========================================================

    torch.save(
        global_model.state_dict(),
        output_dir / "global_model.pt"
    )


    # ========================================================
    # SAVE GLOBAL VALIDATION HISTORY
    # ========================================================

    pd.DataFrame(
        global_val_history
    ).to_csv(
        output_dir / "global_val_history.csv",
        index=False
    )


    # ========================================================
    # GLOBAL TEST
    # ========================================================

    global_test_mae, global_test_rmse = evaluate_model(
        global_model,
        X_global_test,
        y_global_test
    )

    print("\n")
    print("=" * 70)
    print("FINAL GLOBAL TEST")
    print("=" * 70)

    print(
        f"Battery: {global_test_battery}"
    )

    print(
        f"MAE  : {global_test_mae:.6f}"
    )

    print(
        f"RMSE : {global_test_rmse:.6f}"
    )


    # ========================================================
    # SAVE GLOBAL TEST METRICS
    # ========================================================

    with open(
        output_dir / "global_test_metrics.txt",
        "w"
    ) as f:

        f.write(
            "FINAL GLOBAL TEST RESULTS\n"
        )

        f.write(
            "=========================\n"
        )

        f.write(
            "Architecture: LSTM-only\n"
        )

        f.write(
            f"Subgroup: {subgroup_name}\n"
        )

        f.write(
            f"Global test battery: "
            f"{global_test_battery}\n"
        )

        f.write(
            f"Alpha: {ALPHA}\n"
        )

        f.write(
            f"MAE: {global_test_mae:.6f}\n"
        )

        f.write(
            f"RMSE: {global_test_rmse:.6f}\n"
        )

        f.write(
            f"Best round: {best_round}\n"
        )

        f.write(
            f"Best validation RMSE: "
            f"{best_global_rmse:.6f}\n"
        )


    # ========================================================
    # LOCAL TEST
    # ========================================================

    local_test_results = []

    for client_id in CLIENTS:

        data = client_data[
            client_id
        ]

        test_mae, test_rmse = evaluate_model(
            global_model,
            data["X_test"],
            data["y_test"]
        )

        print(
            f"\n{client_id} local test:"
        )

        print(
            f"  MAE  = {test_mae:.6f}"
        )

        print(
            f"  RMSE = {test_rmse:.6f}"
        )

        local_test_results.append({

            "client":
                client_id,

            "test_mae":
                test_mae,

            "test_rmse":
                test_rmse
        })


    pd.DataFrame(
        local_test_results
    ).to_csv(
        output_dir / "local_test_metrics.csv",
        index=False
    )


    # ========================================================
    # CONVERGENCE
    # ========================================================

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
        output_dir / "convergence.txt",
        "w"
    ) as f:

        f.write(
            "FEDERATED TRAINING CONVERGENCE\n"
        )

        f.write(
            "===============================\n"
        )

        f.write(
            "Architecture: LSTM-only\n"
        )

        f.write(
            f"Subgroup: {subgroup_name}\n"
        )

        f.write(
            f"Alpha: {ALPHA}\n"
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
            f"Best round: {best_round}\n"
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
            f"Patience: {PATIENCE}\n"
        )

        f.write(
            f"Stopping reason: "
            f"{stopping_reason}\n"
        )


    save_training_arrays(
        output_dir,
        rounds,
        client_rmse_history,
        weight_history
    )


    # ========================================================
    # VERIFY OUTPUTS
    # ========================================================

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

    print("\n")
    print("=" * 70)
    print("OUTPUT VERIFICATION")
    print("=" * 70)

    all_ok = True

    for filename in expected_files:

        path = output_dir / filename

        if path.exists():

            print(
                f"{filename}: OK"
            )

        else:

            print(
                f"{filename}: MISSING"
            )

            all_ok = False

    if all_ok:

        print(
            "\nAll expected output files are present."
        )

    else:

        print(
            "\nWARNING: Some expected files are missing."
        )

    print(
        f"\nLSTM-only {subgroup_name} completed."
    )


# ============================================================
# 18. MAIN
# ============================================================

def main():

    print("\n")
    print("#" * 70)
    print("# LSTM-ONLY ABLATION STUDY")
    print("#" * 70)

    print(
        "\nExisting baseline outputs are NOT touched."
    )

    print(
        f"\nLSTM-only ablation root:\n"
        f"{ABLATION_ROOT}"
    )

    print(
        f"\nAlpha: {ALPHA}"
    )

    for subgroup_name, subgroup_config in SUBGROUPS.items():

        train_subgroup(
            subgroup_name,
            subgroup_config
        )

    print("\n")
    print("#" * 70)
    print("# LSTM-ONLY ABLATION COMPLETE")
    print("#" * 70)

if __name__ == "__main__":

  main()
