from src.preprocessing import *
from src.features import *
from src.models import *
from src.montecarlo import *
from src.analysis import *
from src.visualization import *

# Load
df = load_data("data/gaiaraw.csv")

# Clean
df = apply_quality_cuts(df)
df = compute_physical_quantities(df)

# Feature space
X, idx = get_feature_space(df, mode="combined")

# Isolation Forest
labels, scores = run_isolation_forest(X)

df = df.loc[idx]
df["iso_label"] = labels
df["iso_score"] = scores

# Monte Carlo stability
stability = monte_carlo_stability(df, get_feature_space)
df["stability"] = stability

# Save robust anomalies
robust = df[(df["iso_label"] == -1) & (df["stability"] > 0.7)]
robust.to_csv("outputs/anomaly_catalog.csv", index=False)

# Plot
plot_hr(df, anomaly_mask=(df["iso_label"] == -1),
        save_path="outputs/figures/hr_anomalies.png", show=False)

# Sensitivity test
results = contamination_sensitivity(
    X, [0.005, 0.01, 0.02, 0.05]
)

print("Contamination sensitivity:", results)
