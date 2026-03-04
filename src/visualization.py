import matplotlib.pyplot as plt
import os


def plot_hr(df, anomaly_mask=None, save_path=None, show=False):

    plt.figure(figsize=(8,10))

    plt.scatter(df["color"], df["M_G"], s=1, alpha=0.2)

    if anomaly_mask is not None:
        anomalies = df[anomaly_mask]
        plt.scatter(anomalies["color"], anomalies["M_G"],
                    s=8)

    plt.gca().invert_yaxis()
    plt.xlabel("BP - RP")
    plt.ylabel("M_G")
    plt.title("HR Diagram with Anomalies")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()
