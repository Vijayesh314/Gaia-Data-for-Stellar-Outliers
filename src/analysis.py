import numpy as np
from sklearn.ensemble import IsolationForest

def contamination_sensitivity(X, contamination_values):

    results = []

    for c in contamination_values:
        model = IsolationForest(
            contamination=c,
            random_state=42
        )
        labels = model.fit_predict(X)
        anomaly_count = np.sum(labels == -1)
        results.append((c, anomaly_count))

    return results