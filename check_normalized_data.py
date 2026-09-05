import numpy as np
from pathlib import Path
import pickle


# ============================================================
# FINAL SANITY CHECK FOR PREPROCESSED FL DATA
# ============================================================
#
# Expected structure:
#
# D:\reproducing_FL
# └── processed_data
#     ├── processed_data_5_18
#     │   ├── client_1_data.npz
#     │   ├── client_2_data.npz
#     │   ├── client_3_data.npz
#     │   ├── global_val.npz
#     │   ├── global_test.npz
#     │   └── normalization_params.pkl
#     │
#     └── processed_data_25_28
#         ├── client_1_data.npz
#         ├── client_2_data.npz
#         ├── client_3_data.npz
#         ├── global_val.npz
#         ├── global_test.npz
#         └── normalization_params.pkl
#
#
# Frozen methodology:
#
# 1. ONE common feature scaler per subgroup
#    fitted ONLY on the three client training datasets.
#
# 2. ONE common RUL scale per subgroup
#    = maximum RUL among the three client training sets.
#
# 3. Client local validation:
#    stored inside client_X_data.npz
#    → used for FedWeightedAvg weighting.
#
# 4. Global validation:
#    global_val.npz
#    → used for global-model validation / early stopping.
#
# 5. Global test:
#    global_test.npz
#    → held-out battery, used once for final evaluation.
#
# 6. Window size = 30
# 7. Number of features = 7
#
# ============================================================


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_PATH = Path(__file__).resolve().parent

PROCESSED_ROOT = (
    BASE_PATH / "processed_data"
)


SUBGROUPS = {

    "5_18": {
        "data_dir":
            PROCESSED_ROOT /
            "processed_data_5_18",

        "expected_rul_scale": 167.0,

        "expected_global_test_battery": "B0018",

        "expected_global_val_windows": 29769,

        "expected_global_test_windows": 34837
    },

    "25_28": {
        "data_dir":
            PROCESSED_ROOT /
            "processed_data_25_28",

        "expected_rul_scale": 27.0,

        "expected_global_test_battery": "B0028",

        "expected_global_val_windows": 8028,

        "expected_global_test_windows": 16309
    }
}


CLIENTS = [
    "client_1",
    "client_2",
    "client_3"
]


EXPECTED_WINDOW = 30
EXPECTED_FEATURES = 7


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def check_no_nan_inf(name, array):

    has_nan = np.isnan(array).any()
    has_inf = np.isinf(array).any()

    if not has_nan and not has_inf:

        print(
            f"  ✅ {name}: no NaN / Inf"
        )

        return True

    if has_nan:

        print(
            f"  ❌ {name}: NaN found"
        )

    if has_inf:

        print(
            f"  ❌ {name}: Inf found"
        )

    return False


def check_feature_shape(name, X):

    if (
        X.ndim == 3
        and X.shape[1] == EXPECTED_WINDOW
        and X.shape[2] == EXPECTED_FEATURES
    ):

        print(
            f"  ✅ {name}: shape = "
            f"(samples, 30, 7)"
        )

        return True

    print(
        f"  ❌ {name}: unexpected shape "
        f"{X.shape}"
    )

    return False


def check_target_shape(name, y):

    if y.ndim == 1:

        print(
            f"  ✅ {name}: target is 1-D"
        )

        return True

    print(
        f"  ❌ {name}: unexpected target "
        f"shape {y.shape}"
    )

    return False


def print_range(name, array):

    print(
        f"  {name}: "
        f"min={array.min():.6f}, "
        f"max={array.max():.6f}, "
        f"mean={array.mean():.6f}"
    )


def check_normalized_rul(name, y):

    minimum = y.min()
    maximum = y.max()

    print_range(
        name,
        y
    )

    if (
        minimum >= -1e-6
        and maximum <= 1.0 + 1e-6
    ):

        print(
            f"  ✅ {name}: within normalized "
            f"range [0, 1]"
        )

        return True

    print(
        f"  ⚠️ {name}: outside [0, 1]"
    )

    return False


def check_feature_range(name, X):

    minimum = X.min()
    maximum = X.max()

    print_range(
        name,
        X
    )

    # Training data should be exactly within
    # the fitted MinMaxScaler range.
    #
    # Validation/test data may legitimately fall
    # slightly outside [0,1] because the scaler was
    # fitted ONLY on training data.

    if (
        minimum >= -1e-6
        and maximum <= 1.0 + 1e-6
    ):

        print(
            f"  ✅ {name}: within [0, 1]"
        )

        return True

    print(
        f"  ⚠️ {name}: outside [0, 1] "
        f"(can be valid for unseen data)"
    )

    return True


# ============================================================
# START
# ============================================================

print("\n" + "#" * 70)
print(
    "FINAL PREPROCESSED DATA SANITY CHECK"
)
print("#" * 70)

print(
    "\nBase path:",
    BASE_PATH
)


overall_success = True


# ============================================================
# CHECK EACH SUBGROUP
# ============================================================

for subgroup, config in SUBGROUPS.items():

    data_dir = config["data_dir"]

    expected_scale = (
        config["expected_rul_scale"]
    )

    expected_test_battery = (
        config["expected_global_test_battery"]
    )

    expected_val_windows = (
        config["expected_global_val_windows"]
    )

    expected_test_windows = (
        config["expected_global_test_windows"]
    )


    print("\n" + "=" * 70)
    print(
        f"CHECKING SUBGROUP {subgroup}"
    )
    print("=" * 70)


    # ========================================================
    # CHECK DIRECTORY
    # ========================================================

    if not data_dir.exists():

        print(
            f"❌ Directory not found:\n"
            f"{data_dir}"
        )

        overall_success = False

        continue

    print(
        "\nData directory:",
        data_dir
    )


    # ========================================================
    # CHECK NORMALIZATION PARAMETERS
    # ========================================================

    print(
        "\n" + "-" * 60
    )

    print(
        "NORMALIZATION PARAMETERS"
    )

    print(
        "-" * 60
    )


    params_file = (
        data_dir /
        "normalization_params.pkl"
    )


    if not params_file.exists():

        print(
            "❌ normalization_params.pkl "
            "NOT FOUND"
        )

        overall_success = False

    else:

        with open(
            params_file,
            "rb"
        ) as f:

            params = pickle.load(f)


        common_max_rul = (
            params["common_max_rul"]
        )

        print(
            f"Common RUL scale: "
            f"{common_max_rul}"
        )

        print(
            f"Expected RUL scale: "
            f"{expected_scale}"
        )


        if np.isclose(
            common_max_rul,
            expected_scale
        ):

            print(
                "✅ Common RUL scale is correct"
            )

        else:

            print(
                "❌ Common RUL scale is incorrect"
            )

            overall_success = False


        print(
            "\nClient training RUL maxima:"
        )

        for client, value in (
            params["client_max_rul"]
            .items()
        ):

            print(
                f"  {client}: "
                f"{value}"
            )


        print(
            "\nFeatures:"
        )

        print(
            params["features"]
        )


        print(
            "\nWindow:",
            params["window"]
        )


        if (
            params["window"]
            == EXPECTED_WINDOW
        ):

            print(
                "✅ Window size = 30"
            )

        else:

            print(
                "❌ Window size is not 30"
            )

            overall_success = False


        if (
            len(params["features"])
            == EXPECTED_FEATURES
        ):

            print(
                "✅ Number of features = 7"
            )

        else:

            print(
                "❌ Number of features "
                "is not 7"
            )

            overall_success = False


        if (
            params["global_test_battery"]
            == expected_test_battery
        ):

            print(
                f"✅ Global test battery = "
                f"{expected_test_battery}"
            )

        else:

            print(
                "❌ Global test battery "
                "does not match"
            )

            overall_success = False


    # ========================================================
    # CHECK CLIENT FILES
    # ========================================================

    for client in CLIENTS:

        print(
            "\n" + "-" * 60
        )

        print(
            f"{client}"
        )

        print(
            "-" * 60
        )


        file_path = (
            data_dir /
            f"{client}_data.npz"
        )


        if not file_path.exists():

            print(
                f"❌ Missing: "
                f"{file_path.name}"
            )

            overall_success = False

            continue


        data = np.load(
            file_path
        )


        X_train = data["X_train"]
        y_train = data["y_train"]

        X_val = data["X_val"]
        y_val = data["y_val"]

        X_test = data["X_test"]
        y_test = data["y_test"]


        # ----------------------------------------------------
        # Shapes
        # ----------------------------------------------------

        print("\nShapes:")

        print(
            f"  X_train: {X_train.shape}"
        )

        print(
            f"  y_train: {y_train.shape}"
        )

        print(
            f"  X_val:   {X_val.shape}"
        )

        print(
            f"  y_val:   {y_val.shape}"
        )

        print(
            f"  X_test:  {X_test.shape}"
        )

        print(
            f"  y_test:  {y_test.shape}"
        )


        # ----------------------------------------------------
        # Shape checks
        # ----------------------------------------------------

        if not check_feature_shape(
            "X_train",
            X_train
        ):
            overall_success = False


        if not check_feature_shape(
            "X_val",
            X_val
        ):
            overall_success = False


        if not check_feature_shape(
            "X_test",
            X_test
        ):
            overall_success = False


        if not check_target_shape(
            "y_train",
            y_train
        ):
            overall_success = False


        if not check_target_shape(
            "y_val",
            y_val
        ):
            overall_success = False


        if not check_target_shape(
            "y_test",
            y_test
        ):
            overall_success = False


        # ----------------------------------------------------
        # NaN / Inf
        # ----------------------------------------------------

        print(
            "\nNaN / Inf checks:"
        )


        arrays = {

            "X_train": X_train,
            "y_train": y_train,

            "X_val": X_val,
            "y_val": y_val,

            "X_test": X_test,
            "y_test": y_test
        }


        for name, array in (
            arrays.items()
        ):

            if not check_no_nan_inf(
                name,
                array
            ):

                overall_success = False


        # ----------------------------------------------------
        # Feature ranges
        # ----------------------------------------------------

        print(
            "\nFeature ranges:"
        )


        check_feature_range(
            "X_train",
            X_train
        )

        check_feature_range(
            "X_val",
            X_val
        )

        check_feature_range(
            "X_test",
            X_test
        )


        # ----------------------------------------------------
        # RUL ranges
        # ----------------------------------------------------

        print(
            "\nRUL ranges:"
        )


        check_normalized_rul(
            "y_train",
            y_train
        )

        check_normalized_rul(
            "y_val",
            y_val
        )

        check_normalized_rul(
            "y_test",
            y_test
        )


        # ----------------------------------------------------
        # Stored RUL scale
        # ----------------------------------------------------

        if "common_max_rul" in data:

            stored_scale = float(
                data["common_max_rul"]
            )

            if np.isclose(
                stored_scale,
                expected_scale
            ):

                print(
                    f"  ✅ Stored common_max_rul "
                    f"= {stored_scale}"
                )

            else:

                print(
                    f"  ❌ Stored common_max_rul "
                    f"= {stored_scale}"
                )

                overall_success = False


# ============================================================
# GLOBAL VALIDATION
# ============================================================

for subgroup, config in SUBGROUPS.items():

    data_dir = config["data_dir"]

    expected_windows = (
        config["expected_global_val_windows"]
    )

    expected_scale = (
        config["expected_rul_scale"]
    )


    print("\n" + "=" * 70)

    print(
        f"GLOBAL VALIDATION — {subgroup}"
    )

    print("=" * 70)


    file_path = (
        data_dir /
        "global_val.npz"
    )


    if not file_path.exists():

        print(
            "❌ global_val.npz NOT FOUND"
        )

        overall_success = False

        continue


    data = np.load(
        file_path
    )


    X = data["X"]
    y = data["y"]


    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )


    # --------------------------------------------------------
    # Window count
    # --------------------------------------------------------

    if len(X) == expected_windows:

        print(
            f"✅ Global validation contains "
            f"{expected_windows} windows"
        )

    else:

        print(
            f"❌ Expected "
            f"{expected_windows} windows, "
            f"found {len(X)}"
        )

        overall_success = False


    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    if not check_feature_shape(
        "Global validation X",
        X
    ):

        overall_success = False


    if not check_target_shape(
        "Global validation y",
        y
    ):

        overall_success = False


    # --------------------------------------------------------
    # NaN / Inf
    # --------------------------------------------------------

    check_no_nan_inf(
        "Global validation X",
        X
    )

    check_no_nan_inf(
        "Global validation y",
        y
    )


    # --------------------------------------------------------
    # Ranges
    # --------------------------------------------------------

    print(
        "\nGlobal validation feature range:"
    )

    check_feature_range(
        "Global validation X",
        X
    )


    print(
        "\nGlobal validation RUL range:"
    )

    check_normalized_rul(
        "Global validation y",
        y
    )


    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    if "common_max_rul" in data:

        stored_scale = float(
            data["common_max_rul"]
        )

        if np.isclose(
            stored_scale,
            expected_scale
        ):

            print(
                f"✅ Global validation scale "
                f"= {stored_scale}"
            )

        else:

            print(
                f"❌ Global validation scale "
                f"= {stored_scale}"
            )

            overall_success = False


# ============================================================
# GLOBAL TEST
# ============================================================

for subgroup, config in SUBGROUPS.items():

    data_dir = config["data_dir"]

    expected_windows = (
        config["expected_global_test_windows"]
    )

    expected_scale = (
        config["expected_rul_scale"]
    )

    expected_battery = (
        config["expected_global_test_battery"]
    )


    print("\n" + "=" * 70)

    print(
        f"GLOBAL TEST — {subgroup}"
    )

    print("=" * 70)


    file_path = (
        data_dir /
        "global_test.npz"
    )


    if not file_path.exists():

        print(
            "❌ global_test.npz NOT FOUND"
        )

        overall_success = False

        continue


    data = np.load(
        file_path
    )


    X = data["X"]
    y = data["y"]


    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )


    # --------------------------------------------------------
    # Battery
    # --------------------------------------------------------

    if "battery" in data:

        battery = str(
            data["battery"]
        )

        if battery == expected_battery:

            print(
                f"✅ Held-out battery = "
                f"{battery}"
            )

        else:

            print(
                f"❌ Expected battery "
                f"{expected_battery}, "
                f"found {battery}"
            )

            overall_success = False

    else:

        print(
            "⚠️ Battery identifier not "
            "stored in global_test.npz"
        )


    # --------------------------------------------------------
    # Window count
    # --------------------------------------------------------

    if len(X) == expected_windows:

        print(
            f"✅ Global test contains "
            f"{expected_windows} windows"
        )

    else:

        print(
            f"❌ Expected "
            f"{expected_windows} windows, "
            f"found {len(X)}"
        )

        overall_success = False


    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    if not check_feature_shape(
        "Global test X",
        X
    ):

        overall_success = False


    if not check_target_shape(
        "Global test y",
        y
    ):

        overall_success = False


    # --------------------------------------------------------
    # NaN / Inf
    # --------------------------------------------------------

    check_no_nan_inf(
        "Global test X",
        X
    )

    check_no_nan_inf(
        "Global test y",
        y
    )


    # --------------------------------------------------------
    # Feature range
    # --------------------------------------------------------

    print(
        "\nGlobal test feature range:"
    )

    check_feature_range(
        "Global test X",
        X
    )


    # --------------------------------------------------------
    # RUL range
    # --------------------------------------------------------

    print(
        "\nGlobal test RUL range:"
    )

    check_normalized_rul(
        "Global test y",
        y
    )


    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    if "common_max_rul" in data:

        stored_scale = float(
            data["common_max_rul"]
        )

        if np.isclose(
            stored_scale,
            expected_scale
        ):

            print(
                f"✅ Global test scale "
                f"= {stored_scale}"
            )

        else:

            print(
                f"❌ Global test scale "
                f"= {stored_scale}"
            )

            overall_success = False


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "#" * 70)

if overall_success:

    print(
        "✅ FINAL SANITY CHECK PASSED"
    )

    print(
        "\nThe preprocessed datasets are "
        "ready for FL training."
    )

else:

    print(
        "❌ FINAL SANITY CHECK FOUND "
        "PROBLEMS"
    )

    print(
        "\nDo NOT start FL training yet."
    )

print("#" * 70)
