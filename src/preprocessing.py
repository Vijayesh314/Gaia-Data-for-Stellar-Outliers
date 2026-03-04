import numpy as np
import pandas as pd
from .config import QUALITY_CUTS

def load_data(path):
    return pd.read_csv(path)

def apply_quality_cuts(df):
    df = df[
        (df["parallax_over_error"] > QUALITY_CUTS["parallax_over_error"]) &
        (df["ruwe"] < QUALITY_CUTS["ruwe"])
    ].copy()
    return df

def compute_physical_quantities(df):
    df["distance_pc"] = 1000 / df["parallax"]
    df["M_G"] = df["phot_g_mean_mag"] + 5*np.log10(df["parallax"]) - 10
    df["color"] = df["bp_rp"]

    # Proper motion magnitude
    df["pm_total"] = np.sqrt(df["pmra"]**2 + df["pmdec"]**2)

    # Tangential velocity
    df["v_t"] = 4.74 * df["pm_total"] / df["parallax"]

    return df