import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend

from statsmodels.tsa.holtwinters import ExponentialSmoothing

def plot(df, start, end, metric, site):
    plt.figure(figsize=(10, 6))
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    site_data = df[
    (df['Site'] == site)  &
    (df['Timestamp'] >= start) &
    (df['Timestamp'] <= end )
    ].set_index("Timestamp")[metric]

    site_data = site_data.asfreq("D")

    model = ExponentialSmoothing(site_data, trend="add").fit(smoothing_level=0.05)
    smoothed_values = model.fittedvalues

    plt.plot(site_data.index, site_data, label="Raw Data", linestyle="dashed", alpha=0.6)
    plt.plot(site_data.index, smoothed_values, label="Smooth Fit", color="red", linewidth=2)

    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.savefig("static/carbon_footprint_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


