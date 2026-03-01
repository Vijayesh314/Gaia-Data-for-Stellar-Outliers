from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

df = pd.read_csv("gaia_clean_sample.csv")

features = df[['M_G', 'bp_rp', 'v_tan']].dropna()

iso = IsolationForest(contamination=0.01, random_state=42)
labels_iso = iso.fit_predict(features)

df.loc[features.index, 'iso_flag'] = labels_iso

df = pd.read_csv("gaia_clean_sample.csv")

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
labels_lof = lof.fit_predict(features)

df.loc[features.index, 'lof_flag'] = labels_lof

mean_v = df['v_tan'].mean()
std_v = df['v_tan'].std()

df['z_score_v'] = (df['v_tan'] - mean_v)/std_v
df['vel_flag'] = df['z_score_v'].abs() > 3

iso_set = set(df[df['iso_flag'] == -1].index)
lof_set = set(df[df['lof_flag'] == -1].index)
vel_set = set(df[df['vel_flag'] == True].index)

print("ISO ∩ LOF:", len(iso_set & lof_set))
print("ISO ∩ Velocity:", len(iso_set & vel_set))
print("LOF ∩ Velocity:", len(lof_set & vel_set))
