import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("gaia_clean_sample.csv")

print(df.shape)
print(df.columns)
df.head()

df["abs_mag_g"] = df["phot_g_mean_mag"] + 5*np.log10(df["parallax"]) - 10
# color index BP - RP
df["bp_rp"] = (
    df["phot_bp_mean_mag"]
    - df["phot_rp_mean_mag"]
)
df[["bp_rp", "abs_mag_g"]].describe()

plt.figure(figsize=(6,8))
plt.scatter(df["bp_rp"], df["abs_mag_g"], s=1)
plt.gca().invert_yaxis()
plt.xlabel("BP - RP")
plt.ylabel("Absolute G Magnitude")
plt.title("HR Diagram (Gaia DR3 Sample)")
plt.show()

# parallax in milliarcseconds
df["distance_pc"] = 1000 / df["parallax"]

# tangential velocity
df["pm_total"] = np.sqrt(
    df["pmra"]**2 + df["pmdec"]**2
)
df["vt_kms"] = 4.74 * df["pm_total"] / df["parallax"]