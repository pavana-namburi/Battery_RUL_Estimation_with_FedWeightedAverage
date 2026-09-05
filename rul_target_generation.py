from scipy.io import loadmat
import pandas as pd
from pathlib import Path
import traceback


# ==============================================================================
# CORE PREPROCESSING FUNCTIONS
# ==============================================================================

def load_data(mat_path, battery):
    mat_path = Path(mat_path)

    if not mat_path.exists():
        raise FileNotFoundError(f"File not found: {mat_path.absolute()}")

    try:
        mat = loadmat(str(mat_path))
    except Exception as e:
        raise ValueError(f"Failed to load .mat file: {e}")

    if battery not in mat:
        available = [k for k in mat.keys() if not k.startswith('__')]
        raise KeyError(f"Battery '{battery}' not found. Keys: {available}")

    dataset = []
    counter = 0

    try:
        cycle_count = len(mat[battery][0, 0]['cycle'][0])
    except Exception as e:
        raise ValueError(f"Invalid file structure for {battery}: {e}")

    for i in range(cycle_count):
        row = mat[battery][0, 0]['cycle'][0, i]

        if row['type'][0] == 'discharge':
            ambient_temperature = row['ambient_temperature'][0][0]
            data = row['data']
            capacity = data[0][0]['Capacity'][0][0]

            num_samples = len(data[0][0]['Voltage_measured'][0])

            for j in range(num_samples):
                dataset.append([
                    counter + 1,
                    ambient_temperature,
                    capacity,
                    data[0][0]['Voltage_measured'][0][j],
                    data[0][0]['Current_measured'][0][j],
                    data[0][0]['Temperature_measured'][0][j],
                    data[0][0]['Current_load'][0][j],
                    data[0][0]['Voltage_load'][0][j],
                    data[0][0]['Time'][0][j]
                ])

            counter += 1

    if not dataset:
        raise ValueError(f"No discharge cycles found for {battery}")

    return pd.DataFrame(dataset, columns=[
        'cycle',
        'ambient_temperature',
        'capacity',
        'voltage_measured',
        'current_measured',
        'temperature_measured',
        'current_load',
        'voltage_load',
        'time'
    ])


def calculate_RUL_capacity_based(df, battery_name, threshold=0.7):
    cap_per_cycle = df.groupby('cycle')['capacity'].first()
    initial_capacity = cap_per_cycle.iloc[0]
    eol_threshold = threshold * initial_capacity

    eol_mask = cap_per_cycle <= eol_threshold
    eol_cycle = cap_per_cycle[eol_mask].index[0] if eol_mask.any() else cap_per_cycle.index.max()

    df = df.copy()
    df['RUL'] = (eol_cycle - df['cycle']).clip(lower=0)

    return df, (
        initial_capacity,
        eol_threshold,
        eol_cycle,
        cap_per_cycle.index.max(),
        cap_per_cycle.iloc[-1]
    )


def preprocess_battery_dataset(mat_path, battery_name, output_dir='./data'):
    Path(output_dir).mkdir(exist_ok=True)

    print(f"\nProcessing {battery_name}")
    df = load_data(mat_path, battery_name)

    df, stats = calculate_RUL_capacity_based(df, battery_name)

    output_path = Path(output_dir) / f"{battery_name}_discharge.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved → {output_path}")
    print(f"Samples: {len(df):,} | Cycles: {df['cycle'].max()}")

    return output_path


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    batteries = [
        "B0005",
        "B0006",
        "B0007",
        "B0018",
        "B0025",
        "B0026",
        "B0027",
        "B0028"
    ]

    input_dir = Path("./raw-data")
    output_dir = r"D:\reproducing_FL\data"

    print("\nNASA PCoE Battery Preprocessing")
    print(f"Input  : {input_dir.resolve()}")
    print(f"Output : {Path(output_dir).resolve()}")

    for battery in batteries:
        mat_file = input_dir / f"{battery}.mat"

        try:
            preprocess_battery_dataset(
                mat_path=str(mat_file),
                battery_name=battery,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"❌ Failed {battery}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
