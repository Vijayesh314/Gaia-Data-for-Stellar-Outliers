from src.preprocessing import *
from src.features import *
from src.models import *
from src.montecarlo import *
from src.visualization import plot_hr

# Load and process
df = load_data("data/gaiaraw.csv")
df = apply_quality_cuts(df)
df = compute_physical_quantities(df)

# Feature space and model
X, idx = get_feature_space(df, mode="combined")
labels, scores = run_isolation_forest(X)

df = df.loc[idx]
df["iso_label"] = labels

# Save anomalies CSV (overwrite)
robust = df[(df["iso_label"] == -1)]
robust.to_csv("outputs/anomaly_catalog.csv", index=False)

# Save HR figure
plot_hr(df, anomaly_mask=(df["iso_label"] == -1), save_path="outputs/figures/hr_anomalies.png", show=False)
print('Saved', 'outputs/figures/hr_anomalies.png')
