# Raw NASA Battery Data

The original NASA PCoE battery `.mat` files are **not included in this repository** because the files are large.

## Dataset Source

Download the NASA PCoE Lithium-Ion Battery Dataset from:

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

## Required Files

Download the following battery files:

- `B0005.mat`
- `B0006.mat`
- `B0007.mat`
- `B0018.mat`
- `B0025.mat`
- `B0026.mat`
- `B0027.mat`
- `B0028.mat`

Place all downloaded `.mat` files directly inside this folder:

```text
raw-data/
├── B0005.mat
├── B0006.mat
├── B0007.mat
├── B0018.mat
├── B0025.mat
├── B0026.mat
├── B0027.mat
└── B0028.mat
```

The preprocessing script reads the files using the expected names:

```python
input_dir = Path("./raw-data")
mat_file = input_dir / f"{battery}.mat"
```

Therefore, **do not rename the downloaded files**.

After placing the files in this folder, run the battery rul_target_generation script to generate the processed CSV datasets used by the federated learning pipeline.