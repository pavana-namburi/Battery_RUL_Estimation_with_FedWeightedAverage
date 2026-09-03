# ============================================================
# ALPHA SENSITIVITY ANALYSIS
#
# Existing alpha = 5.0 results are NOT rerun.
#
# New alpha values:
#
#     0.5
#     1.0
#     2.0
#     10.0
#
# Both subgroups are tested:
#
#     5_18
#     25_28
#
# The existing FL_training.py is reused.
#
# No absolute machine-specific paths are used.
#
# The repository root is automatically determined from the
# location of this script.
#
# Existing experiment directories are NEVER overwritten.
# ============================================================

import os
import sys
import subprocess
from pathlib import Path

import pandas as pd


# ============================================================
# 1. REPOSITORY ROOT
# ============================================================

# This script is expected to be in the repository root:
#
# reproducing_FL/
#     FL_training_alpha_analysis.py
#     alpha_sensitivity_analysis.py
#
# Therefore this automatically identifies the repository root.

BASE_DIR = Path(
    __file__
).resolve().parent


# ============================================================
# 2. TRAINING SCRIPT
# ============================================================

TRAINING_SCRIPT = (
    BASE_DIR
    / "FL_training_alpha_analysis.py"
)


# ============================================================
# 3. SENSITIVITY OUTPUT ROOT
# ============================================================

SENSITIVITY_ROOT = (
    BASE_DIR
    / "alpha_sensitivity_analysis"
)


# ============================================================
# 4. ALPHA VALUES
#
# IMPORTANT:
#
# Alpha = 5.0 is NOT included.
#
# Alpha = 5.0 has already been executed through
# FL_training.py and those results are stored under:
#
#     outputs/output_5_18
#     outputs/output_25_28
#
# They are not touched by this script.
# ============================================================

ALPHAS = [
    0.5,
    1.0,
    2.0,
    10.0
]


# ============================================================
# 5. SUBGROUPS
# ============================================================

SUBGROUPS = [
    "5_18",
    "25_28"
]


# ============================================================
# 6. SAFETY CHECK: TRAINING SCRIPT
# ============================================================

def check_training_script():

    if not TRAINING_SCRIPT.exists():

        raise FileNotFoundError(

            "\nFL_training.py was not found.\n"
            f"Expected location:\n"
            f"{TRAINING_SCRIPT}\n\n"

            "Make sure alpha_sensitivity_analysis.py "
            "is placed in the repository root."
        )


# ============================================================
# 7. CREATE SENSITIVITY ROOT
# ============================================================

def create_sensitivity_root():

    SENSITIVITY_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 8. CHECK EXPERIMENT OUTPUT DIRECTORY
#
# SAFETY RULE:
#
# If the directory does not exist:
#     -> create it
#     -> safe to run
#
# If it exists and is empty:
#     -> safe to run
#
# If it exists and contains ANYTHING:
#     -> refuse to run
#     -> do NOT delete anything
#     -> do NOT overwrite anything
# ============================================================

def check_output_directory(
    output_dir
):

    output_dir = Path(
        output_dir
    )

    if output_dir.exists():

        contents = list(
            output_dir.iterdir()
        )

        if len(contents) > 0:

            print("\n")
            print("!" * 70)

            print(
                "REFUSING TO RUN EXPERIMENT"
            )

            print(
                f"Output directory already contains "
                f"files/folders:\n"
                f"{output_dir}"
            )

            print(
                "\nExisting results will NOT be "
                "deleted or overwritten."
            )

            print("!" * 70)

            return False

    else:

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    return True


# ============================================================
# 9. VERIFY OUTPUTS
# ============================================================

def verify_outputs(
    output_dir
):

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

    missing_files = []

    for filename in expected_files:

        path = (
            output_dir
            /
            filename
        )

        if not path.exists():

            missing_files.append(
                filename
            )


    attention_dir = (
        output_dir
        /
        "attention_weights"
    )

    expected_attention = [

        "client_1_attention.npy",

        "client_2_attention.npy",

        "client_3_attention.npy"
    ]

    if not attention_dir.exists():

        missing_files.append(
            "attention_weights/"
        )

    else:

        for filename in expected_attention:

            path = (
                attention_dir
                /
                filename
            )

            if not path.exists():

                missing_files.append(
                    f"attention_weights/{filename}"
                )


    if missing_files:

        print("\n")
        print("!" * 70)

        print(
            "OUTPUT VERIFICATION FAILED"
        )

        print(
            "Missing files:"
        )

        for filename in missing_files:

            print(
                f"  MISSING: {filename}"
            )

        print("!" * 70)

        return False


    return True


# ============================================================
# 10. READ RESULTS
# ============================================================

def read_results(
    output_dir
):

    metrics_path = (
        output_dir
        /
        "global_test_metrics.txt"
    )

    convergence_path = (
        output_dir
        /
        "convergence.txt"
    )


    # --------------------------------------------------------
    # Read MAE and RMSE
    # --------------------------------------------------------

    mae = None
    rmse = None

    with open(
        metrics_path,
        "r"
    ) as f:

        for line in f:

            line = line.strip()

            if line.startswith(
                "MAE:"
            ):

                mae = float(
                    line.split(
                        ":",
                        1
                    )[1].strip()
                )

            elif line.startswith(
                "RMSE:"
            ):

                rmse = float(
                    line.split(
                        ":",
                        1
                    )[1].strip()
                )


    # --------------------------------------------------------
    # Read number of completed rounds
    # --------------------------------------------------------

    total_rounds = None

    with open(
        convergence_path,
        "r"
    ) as f:

        for line in f:

            line = line.strip()

            if line.startswith(
                "Total rounds completed:"
            ):

                total_rounds = int(
                    line.split(
                        ":",
                        1
                    )[1].strip()
                )


    if mae is None:

        raise ValueError(
            f"Could not find MAE in:\n"
            f"{metrics_path}"
        )

    if rmse is None:

        raise ValueError(
            f"Could not find RMSE in:\n"
            f"{metrics_path}"
        )

    if total_rounds is None:

        raise ValueError(
            f"Could not find total rounds in:\n"
            f"{convergence_path}"
        )


    return (
        mae,
        rmse,
        total_rounds
    )


# ============================================================
# 11. RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    subgroup_name,
    alpha,
    output_dir
):

    print("\n")
    print("=" * 70)

    print(
        f"SUBGROUP : {subgroup_name}"
    )

    print(
        f"ALPHA    : {alpha}"
    )

    print(
        f"OUTPUT   : {output_dir}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if not check_output_directory(
        output_dir
    ):

        print(
            f"\nSKIPPED:"
            f" subgroup={subgroup_name},"
            f" alpha={alpha}"
        )

        return None


    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = os.environ.copy()

    env["ALPHA"] = str(
        alpha
    )

    env["OUTPUT_ROOT"] = str(
        output_dir
    )

    env["SUBGROUP_FILTER"] = (
        subgroup_name
    )


    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    print(
        "\nStarting FL training:"
    )

    print(
        f"  ALPHA = {alpha}"
    )

    print(
        f"  SUBGROUP_FILTER = "
        f"{subgroup_name}"
    )

    print(
        f"  OUTPUT_ROOT = "
        f"{output_dir}"
    )


    # --------------------------------------------------------
    # Run FL_training.py
    #
    # sys.executable ensures the same Python interpreter/
    # environment is used.
    # --------------------------------------------------------

    result = subprocess.run(

        [
            sys.executable,
            str(TRAINING_SCRIPT)
        ],

        env=env,

        cwd=str(
            BASE_DIR
        ),

        text=True
    )


    # --------------------------------------------------------
    # Training failed
    # --------------------------------------------------------

    if result.returncode != 0:

        print("\n")
        print("!" * 70)

        print(
            "TRAINING FAILED"
        )

        print(
            f"Subgroup : {subgroup_name}"
        )

        print(
            f"Alpha    : {alpha}"
        )

        print(
            f"Exit code: {result.returncode}"
        )

        print("!" * 70)

        return None


    # --------------------------------------------------------
    # Verify outputs
    # --------------------------------------------------------

    print(
        "\nChecking generated output files..."
    )

    if not verify_outputs(
        output_dir
    ):

        print(
            "\nExperiment finished but "
            "output verification failed."
        )

        return None


    # --------------------------------------------------------
    # Read results
    # --------------------------------------------------------

    mae, rmse, rounds = (
        read_results(
            output_dir
        )
    )


    print("\n")
    print("-" * 70)

    print(
        f"RESULTS"
    )

    print(
        f"Subgroup : {subgroup_name}"
    )

    print(
        f"Alpha    : {alpha}"
    )

    print(
        f"MAE      : {mae}"
    )

    print(
        f"RMSE     : {rmse}"
    )

    print(
        f"Rounds   : {rounds}"
    )

    print(
        f"Output   : {output_dir}"
    )

    print("-" * 70)


    return {

        "subgroup":
            subgroup_name,

        "alpha":
            alpha,

        "MAE":
            mae,

        "RMSE":
            rmse,

        "Rounds":
            rounds,

        "output_directory":
            str(output_dir)
    }


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("\n")
    print("#" * 70)

    print(
        "# ALPHA SENSITIVITY ANALYSIS"
    )

    print("#" * 70)


    # --------------------------------------------------------
    # Show configuration
    # --------------------------------------------------------

    print(
        f"\nRepository root:\n"
        f"{BASE_DIR}"
    )

    print(
        f"\nTraining script:\n"
        f"{TRAINING_SCRIPT}"
    )

    print(
        f"\nSensitivity output root:\n"
        f"{SENSITIVITY_ROOT}"
    )

    print(
        "\nAlpha values that WILL be run:"
    )

    for alpha in ALPHAS:

        print(
            f"  {alpha}"
        )

    print(
        "\nAlpha = 5.0 will NOT be run."
    )

    print(
        "The existing alpha=5.0 results remain untouched."
    )


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    check_training_script()

    create_sensitivity_root()


    results = []


    # ========================================================
    # SUBGROUP LOOP
    # ========================================================

    for subgroup_name in SUBGROUPS:

        subgroup_root = (
            SENSITIVITY_ROOT
            /
            f"alpha_{subgroup_name}"
        )

        subgroup_root.mkdir(
            parents=True,
            exist_ok=True
        )


        # ====================================================
        # ALPHA LOOP
        # ====================================================

        for alpha in ALPHAS:

            alpha_string = (
                f"{float(alpha):.1f}"
            )

            output_dir = (
                subgroup_root
                /
                f"alpha_{alpha_string}_results"
            )


            result = run_experiment(

                subgroup_name=
                    subgroup_name,

                alpha=
                    alpha,

                output_dir=
                    output_dir
            )


            if result is not None:

                results.append(
                    result
                )


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    if results:

        results_df = pd.DataFrame(
            results
        )

        summary_path = (
            SENSITIVITY_ROOT
            /
            "alpha_sensitivity_summary.csv"
        )


        # ----------------------------------------------------
        # Never overwrite an existing summary
        # ----------------------------------------------------

        if summary_path.exists():

            print("\n")
            print("!" * 70)

            print(
                "SUMMARY FILE ALREADY EXISTS"
            )

            print(
                f"{summary_path}"
            )

            print(
                "\nExisting summary will NOT "
                "be overwritten."
            )

            print("!" * 70)

        else:

            results_df.to_csv(
                summary_path,
                index=False
            )

            print(
                f"\nSummary saved to:\n"
                f"{summary_path}"
            )


    else:

        print(
            "\nNo new experiments were completed."
        )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n")
    print("#" * 70)

    print(
        "# ALPHA SENSITIVITY ANALYSIS COMPLETE"
    )

    print("#" * 70)

    print(
        f"\nResults root:\n"
        f"{SENSITIVITY_ROOT}"
    )

    print(
        "\nAlpha = 5.0 was NOT rerun."
    )

    print(
        "Existing experiment directories were never overwritten."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
