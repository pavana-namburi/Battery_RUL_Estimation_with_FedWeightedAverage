import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle


# ============================================================
# FINAL FEDERATED LEARNING PREPROCESSING
# ============================================================
#
# Methodology:
#
# 1. ONE common feature scaler per subgroup
#    - fitted ONLY on the three clients' TRAIN features
#    - applied to all train / val / local-test /
#      global-val / global-test features
#
# 2. ONE common RUL scale per subgroup
#    - maximum RUL among the three clients' TRAIN sets
#    - applied to all RUL targets in that subgroup
#
# 3. Client local validation remains inside client_X_data.npz
#    - used for FedWeightedAvg RMSE weighting
#
# 4. Global validation is stored separately in global_val.npz
#    - used for global-model validation / early stopping
#
# 5. Global test is stored separately in global_test.npz
#    - held-out battery
#    - used ONCE for final evaluation
#
# 6. Window size = 30 records
# 7. Target = mean RUL of the final 3 records in each window
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(__file__).resolve().parent

RAW_DATA_DIR = BASE_PATH / "data-splits"
OUTPUT_ROOT = BASE_PATH / "processed_data"


# ============================================================
# SUBGROUP CONFIGURATION
# ============================================================

SUBGROUPS = {

    "5_18": {
        "input_dir": RAW_DATA_DIR / "data_splits_5_18",
        "output_dir": OUTPUT_ROOT / "processed_data_5_18",
        "global_test_battery": "B0018"
    },

    "25_28": {
        "input_dir": RAW_DATA_DIR / "data_splits_25_28",
        "output_dir": OUTPUT_ROOT / "processed_data_25_28",
        "global_test_battery": "B0028"
    }
}


# ============================================================
# GENERAL SETTINGS
# ============================================================

CLIENTS = [
    "client_1",
    "client_2",
    "client_3"
]

WINDOW = 30

FEATURES = [
    "capacity",
    "ambient_temperature",
    "voltage_measured",
    "current_measured",
    "temperature_measured",
    "current_load",
    "voltage_load"
]

TARGET = "RUL"


# ============================================================
# LOAD ONE CSV
# ============================================================

def load_single_csv(folder):

    csv_files = list(Path(folder).glob("*.csv"))

    if len(csv_files) != 1:
        raise ValueError(
            f"Expected exactly one CSV in:\n{folder}\n"
            f"Found: {len(csv_files)}"
        )

    return pd.read_csv(
        csv_files[0]
    ).reset_index(drop=True)


# ============================================================
# WINDOW GENERATION
# ============================================================

def windowize(df):

    X = []
    y = []

    for i in range(
        WINDOW - 1,
        len(df)
    ):

        # 30 consecutive records
        X.append(
            df[
                FEATURES
            ].iloc[
                i - WINDOW + 1:i + 1
            ].values
        )

        # Mean RUL of final 3 records
        y.append(
            df[TARGET].iloc[
                max(0, i - 2):i + 1
            ].mean()
        )

    return (
        np.asarray(X),
        np.asarray(y)
    )


# ============================================================
# PROCESS ONE SUBGROUP
# ============================================================

def process_subgroup(
    subgroup_name,
    config
):

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    global_test_battery = config[
        "global_test_battery"
    ]

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    print("\n" + "=" * 70)
    print(
        f"PROCESSING SUBGROUP {subgroup_name}"
    )
    print("=" * 70)


    # ========================================================
    # STEP 1 — LOAD CLIENT TRAINING DATA
    # ========================================================

    train_data = {}

    for client in CLIENTS:

        train_path = (
            input_dir /
            client /
            "train"
        )

        train_data[client] = (
            load_single_csv(
                train_path
            )
        )


    # ========================================================
    # STEP 2 — COMMON FEATURE SCALER
    # ========================================================
    #
    # Fit ONLY on the three clients' TRAINING FEATURES.
    #
    # No validation data.
    # No local-test data.
    # No global-validation data.
    # No global-test data.
    #
    # ========================================================

    train_feature_frames = [
        train_data[client][FEATURES]
        for client in CLIENTS
    ]

    pooled_train_features = pd.concat(
        train_feature_frames,
        ignore_index=True
    )

    feature_scaler = MinMaxScaler()

    feature_scaler.fit(
        pooled_train_features
    )


    print("\nCommon feature scaler fitted.")
    print(
        "Training rows used:",
        len(pooled_train_features)
    )


    # ========================================================
    # STEP 3 — COMMON RUL SCALE
    # ========================================================
    #
    # Find maximum RUL in each CLIENT'S TRAINING DATA.
    #
    # Then use the largest training maximum as the
    # subgroup-level RUL scale.
    #
    # ========================================================

    client_max_rul = {}

    for client in CLIENTS:

        max_rul = float(
            train_data[client][TARGET].max()
        )

        client_max_rul[client] = max_rul


    common_max_rul = max(
        client_max_rul.values()
    )


    print("\nClient training RUL maxima:")

    for client in CLIENTS:

        print(
            f"  {client}: "
            f"{client_max_rul[client]:.6f}"
        )


    print(
        f"\nCommon RUL scale: "
        f"{common_max_rul:.6f}"
    )


    # ========================================================
    # STEP 4 — SAVE NORMALIZATION PARAMETERS
    # ========================================================

    normalization_params = {

        "feature_scaler": feature_scaler,

        "common_max_rul": common_max_rul,

        "client_max_rul": client_max_rul,

        "features": FEATURES,

        "target": TARGET,

        "window": WINDOW,

        "subgroup": subgroup_name,

        "global_test_battery": global_test_battery
    }


    with open(
        output_dir /
        "normalization_params.pkl",
        "wb"
    ) as f:

        pickle.dump(
            normalization_params,
            f
        )


    # ========================================================
    # STEP 5 — PROCESS EACH CLIENT
    # ========================================================
    #
    # Each client receives:
    #
    # X_train / y_train
    # X_val   / y_val
    # X_test  / y_test
    #
    # These local validation datasets are what the
    # FedWeightedAvg algorithm will use to calculate
    # each client's RMSE and aggregation weight.
    #
    # ========================================================

    for client in CLIENTS:

        print("\n" + "-" * 60)
        print(
            f"Processing {client}"
        )
        print("-" * 60)


        # ----------------------------------------------------
        # Load raw client datasets
        # ----------------------------------------------------

        train_df = train_data[client].copy()

        val_df = load_single_csv(
            input_dir /
            client /
            "val"
        )

        test_df = load_single_csv(
            input_dir /
            client /
            "local_test"
        )


        # ----------------------------------------------------
        # Apply COMMON feature scaler
        # ----------------------------------------------------

        train_df[FEATURES] = (
            feature_scaler.transform(
                train_df[FEATURES]
            )
        )

        val_df[FEATURES] = (
            feature_scaler.transform(
                val_df[FEATURES]
            )
        )

        test_df[FEATURES] = (
            feature_scaler.transform(
                test_df[FEATURES]
            )
        )


        # ----------------------------------------------------
        # Apply COMMON subgroup RUL scale
        # ----------------------------------------------------

        train_df[TARGET] = (
            train_df[TARGET]
            / common_max_rul
        )

        val_df[TARGET] = (
            val_df[TARGET]
            / common_max_rul
        )

        test_df[TARGET] = (
            test_df[TARGET]
            / common_max_rul
        )


        # ----------------------------------------------------
        # Create windows
        # ----------------------------------------------------

        X_train, y_train = windowize(
            train_df
        )

        X_val, y_val = windowize(
            val_df
        )

        X_test, y_test = windowize(
            test_df
        )


        # ----------------------------------------------------
        # Save client data
        # ----------------------------------------------------

        np.savez(
            output_dir /
            f"{client}_data.npz",

            X_train=X_train,
            y_train=y_train,

            X_val=X_val,
            y_val=y_val,

            X_test=X_test,
            y_test=y_test,

            common_max_rul=common_max_rul
        )


        print(
            f"{client}: "
            f"train={X_train.shape}, "
            f"val={X_val.shape}, "
            f"test={X_test.shape}"
        )


    # ========================================================
    # STEP 6 — GLOBAL VALIDATION
    # ========================================================
    #
    # Global validation contains:
    #
    # B0005_global_val
    # B0006_global_val
    # B0007_global_val
    #
    # Each battery is windowized SEPARATELY.
    #
    # This prevents a sequence from crossing from one battery
    # into another battery.
    #
    # These data are used for:
    #
    #     Global model validation
    #     Early stopping
    #
    # They are NOT used for FedWeightedAvg client weighting.
    #
    # ========================================================

    global_val_dir = (
        input_dir /
        "global_val"
    )

    global_val_X = []
    global_val_y = []


    global_val_files = sorted(
        global_val_dir.glob("*.csv")
    )


    for file in global_val_files:

        battery_name = file.stem

        df = pd.read_csv(
            file
        ).reset_index(drop=True)


        # ----------------------------------------------------
        # Feature transformation
        # ----------------------------------------------------

        df[FEATURES] = (
            feature_scaler.transform(
                df[FEATURES]
            )
        )


        # ----------------------------------------------------
        # RUL transformation
        # ----------------------------------------------------

        df[TARGET] = (
            df[TARGET]
            / common_max_rul
        )


        # ----------------------------------------------------
        # Windowize this battery separately
        # ----------------------------------------------------

        X, y = windowize(
            df
        )

        global_val_X.append(
            X
        )

        global_val_y.append(
            y
        )


        print(
            f"Global validation "
            f"{battery_name}: "
            f"X={X.shape}, "
            f"y={y.shape}"
        )


    # --------------------------------------------------------
    # Combine global validation windows
    # --------------------------------------------------------

    global_val_X = np.concatenate(
        global_val_X,
        axis=0
    )

    global_val_y = np.concatenate(
        global_val_y,
        axis=0
    )


    np.savez(
        output_dir /
        "global_val.npz",

        X=global_val_X,
        y=global_val_y,

        common_max_rul=common_max_rul
    )


    print(
        "\nGlobal validation combined: "
        f"X={global_val_X.shape}, "
        f"y={global_val_y.shape}"
    )


    # ========================================================
    # STEP 7 — HELD-OUT GLOBAL TEST
    # ========================================================
    #
    # Example:
    #
    # 5_18  → B0018
    # 25_28 → B0028
    #
    # This battery is NEVER used for:
    #
    # - feature-scaler fitting
    # - RUL-scale fitting
    # - local training
    # - FedWeightedAvg weighting
    # - early stopping
    #
    # It is used ONCE for final evaluation.
    #
    # ========================================================

    global_test_dir = (
        input_dir /
        "global_test" /
        global_test_battery
    )

    global_test_df = load_single_csv(
        global_test_dir
    )


    # --------------------------------------------------------
    # Apply common feature scaler
    # --------------------------------------------------------

    global_test_df[FEATURES] = (
        feature_scaler.transform(
            global_test_df[FEATURES]
        )
    )


    # --------------------------------------------------------
    # Apply common RUL scale
    #
    # IMPORTANT:
    # Do NOT calculate the maximum RUL of the test battery.
    # --------------------------------------------------------

    global_test_df[TARGET] = (
        global_test_df[TARGET]
        / common_max_rul
    )


    # --------------------------------------------------------
    # Windowize
    # --------------------------------------------------------

    X_global_test, y_global_test = (
        windowize(
            global_test_df
        )
    )


    # --------------------------------------------------------
    # Save held-out global test
    # --------------------------------------------------------

    np.savez(
        output_dir /
        "global_test.npz",

        X=X_global_test,
        y=y_global_test,

        common_max_rul=common_max_rul,

        battery=global_test_battery
    )


    print(
        f"\nGlobal test "
        f"({global_test_battery}): "
        f"X={X_global_test.shape}, "
        f"y={y_global_test.shape}"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print(
        f"SUBGROUP {subgroup_name} COMPLETE"
    )
    print("=" * 70)

    print(
        "Feature scaler: "
        "COMMON across 3 clients"
    )

    print(
        "RUL scale: "
        "COMMON across 3 clients"
    )

    print(
        f"Common max RUL: "
        f"{common_max_rul:.6f}"
    )

    print(
        "Local validation: "
        "stored inside each client_data.npz"
    )

    print(
        "Global validation: "
        "global_val.npz"
    )

    print(
        f"Global test battery: "
        f"{global_test_battery}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )


# ============================================================
# RUN BOTH SUBGROUPS
# ============================================================

if __name__ == "__main__":

    print("\n" + "#" * 70)
    print(
        "STARTING FINAL PREPROCESSING"
    )
    print("#" * 70)

    for subgroup_name, config in SUBGROUPS.items():

        process_subgroup(
            subgroup_name,
            config
        )

    print("\n" + "=" * 70)
    print(
        "ALL SUBGROUPS PREPROCESSING COMPLETE"
    )
    print("=" * 70)
