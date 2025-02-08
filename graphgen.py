import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def plot_carbon_footprint(start, end, df, site):

    plt.figure(figsize=(10, 6))
    
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    
    site_data = df[
        (df['Site'] == site)  &
        (df['Timestamp'] >= start) &
        (df['Timestamp'] <= end )
    ].set_index("Timestamp")["Carbon_Intensity_kgCO2e_per_liter"]


    site_data = site_data.asfreq("D")

    model = ExponentialSmoothing(site_data, trend="add").fit(smoothing_level=0.05)
    smoothed_values = model.fittedvalues

    plt.plot(site_data.index, site_data, label="Actual Data", linestyle="dashed", alpha=0.6)
    plt.plot(site_data.index, smoothed_values, label="Smoothed Fit", color="red", linewidth=2)

    plt.title(f"Carbon Intensity Over Time - {site}")
    plt.ylabel("Carbon Intensity (kgCO2e per liter)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('Diageo_Scotland_Full_Year_2024_Daily_Data.csv')
    plot_carbon_footprint('2024-01-01', '2024-12-01', df, 'Cameronbridge')

