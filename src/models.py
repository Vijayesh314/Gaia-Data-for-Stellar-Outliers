from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from .config import CONTAMINATION, RANDOM_STATE

def run_isolation_forest(X):
    model = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    labels = model.fit_predict(X)
    scores = model.decision_function(X)
    return labels, scores

def run_lof(X):
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=CONTAMINATION
    )
    labels = model.fit_predict(X)
    return labels