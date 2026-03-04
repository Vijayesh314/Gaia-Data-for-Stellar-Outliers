import numpy as np
import pandas as pd
from .models import run_isolation_forest
from .config import N_MONTE_CARLO


def monte_carlo_stability(df, feature_function):

    # Use a pandas Series indexed by the original dataframe index
    # so we can safely accumulate counts using label-based indexing.
    consistency = pd.Series(0, index=df.index, dtype=float)

    for i in range(N_MONTE_CARLO):

        perturbed = df.copy()

        perturbed["parallax"] = np.random.normal(
            df["parallax"],
            df["parallax_error"]
        )

        X, idx = feature_function(perturbed)
        labels, _ = run_isolation_forest(X)

        anomaly_mask = (labels == -1).astype(int)
        # idx is an index (labels/positions) from the features selection;
        # align by label-based indexing using .loc
        consistency.loc[idx] += anomaly_mask

    consistency /= N_MONTE_CARLO
    return consistency