import pandas as pd
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ==================================================
# CONFIG
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

CSV_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_ROOT = os.path.join(BASE_DIR, "data_splits")

SUBGROUPS = {
    "5_18": {
        "output_dir": os.path.join(OUTPUT_ROOT, "data_splits_5_18"),
        "client_batteries": ["B0005", "B0006", "B0007"],
        "global_test_battery": "B0018"
    },

    "25_28": {
        "output_dir": os.path.join(OUTPUT_ROOT, "data_splits_25_28"),
        "client_batteries": ["B0025", "B0026", "B0027"],
        "global_test_battery": "B0028"
    }
}

GV_RATIO = 0.20
LOCAL_TRAIN_RATIO = 0.64
LOCAL_VAL_RATIO   = 0.20
LOCAL_TEST_RATIO  = 0.16

EXPECTED_COLS = [
    'cycle', 'ambient_temperature', 'capacity',
    'voltage_measured', 'current_measured',
    'temperature_measured', 'current_load',
    'voltage_load', 'time', 'RUL'
]

# ==================================================
def verify_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns {missing}")

    return df.sort_values("cycle").reset_index(drop=True)

# ==================================================
def split_global_and_local(df):
    cycles = df.cycle.unique()
    gv_n = max(1, int(GV_RATIO * len(cycles)))

    gv_cycles = cycles[-gv_n:]
    local_cycles = cycles[:-gv_n]

    return (
        df[df.cycle.isin(local_cycles)],
        df[df.cycle.isin(gv_cycles)]
    )

def split_local(df):
    cycles = df.cycle.unique()
    n = len(cycles)

    t = int(LOCAL_TRAIN_RATIO * n)
    v = t + int(LOCAL_VAL_RATIO * n)

    return (
        df[df.cycle.isin(cycles[:t])],
        df[df.cycle.isin(cycles[t:v])],
        df[df.cycle.isin(cycles[v:])]
    )

# ==================================================
def create_dirs(output_dir, global_test_battery):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i in range(1, 4):
        for s in ["train", "val", "local_test"]:
            Path(
                f"{output_dir}/client_{i}/{s}"
            ).mkdir(
                parents=True,
                exist_ok=True
            )

    Path(
        f"{output_dir}/global_val"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    Path(
        f"{output_dir}/global_test/{global_test_battery}"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

# ==================================================
def process_subgroup(subgroup_name, config):

    output_dir = config["output_dir"]
    client_batteries = config["client_batteries"]
    global_test_battery = config["global_test_battery"]

    print("\n" + "=" * 60)
    print(f"PROCESSING SUBGROUP: {subgroup_name}")
    print("=" * 60)

    print(f"Input directory : {CSV_DIR}")
    print(f"Output directory: {output_dir}")
    print(f"Clients         : {client_batteries}")
    print(f"Global test     : {global_test_battery}")

    create_dirs(
        output_dir,
        global_test_battery
    )

    for idx, bid in enumerate(client_batteries, 1):

        df = verify_csv(
            os.path.join(
                CSV_DIR,
                f"{bid}_discharge.csv"
            )
        )

        local_df, gv_df = split_global_and_local(df)

        tr, va, te = split_local(local_df)

        tr.to_csv(
            os.path.join(
                output_dir,
                f"client_{idx}",
                "train",
                f"{bid}_train.csv"
            ),
            index=False
        )

        va.to_csv(
            os.path.join(
                output_dir,
                f"client_{idx}",
                "val",
                f"{bid}_val.csv"
            ),
            index=False
        )

        te.to_csv(
            os.path.join(
                output_dir,
                f"client_{idx}",
                "local_test",
                f"{bid}_local_test.csv"
            ),
            index=False
        )

        gv_df.to_csv(
            os.path.join(
                output_dir,
                "global_val",
                f"{bid}_global_val.csv"
            ),
            index=False
        )

    gdf = verify_csv(
        os.path.join(
            CSV_DIR,
            f"{global_test_battery}_discharge.csv"
        )
    )

    gdf.to_csv(
        os.path.join(
            output_dir,
            "global_test",
            global_test_battery,
            f"{global_test_battery}_test.csv"
        ),
        index=False
    )

    print(f"✅ Subgroup {subgroup_name} splitting complete")

# ==================================================
def main():

    print("=" * 60)
    print("FEDERATED LEARNING DATA PARTITIONING")
    print("=" * 60)

    for subgroup_name, config in SUBGROUPS.items():

        process_subgroup(
            subgroup_name,
            config
        )

    print("\n" + "=" * 60)
    print("✅ ALL DATA SPLITS CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nInput : {CSV_DIR}")
    print(f"Output: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
