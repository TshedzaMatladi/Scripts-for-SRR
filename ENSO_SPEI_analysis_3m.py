# -*- coding: utf-8 -*-
"""
This script uses the ENSO classification from ONI_classification_results.xlsx and the 
3-month SPEI analysis from {station_name}_SPEI.csv to calculate probabilities by dividing the
number of dry/near-normal/wet occurrences for each station by the number of EN/N/LN events.
Output is written to C:/ENSO/

Created April 2025
@author: SteynAS@ufs.ac.za
"""

import pandas as pd
import os
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import numpy as np

# === Station Selection Menu ===
stations = {
    "Polokwane": (-23.8, 29.7),
    "Mbombela": (-25.5, 31.0),
    "Potchefstroom": (-26.7, 27.1),
    "Bloemfontein": (-28.9, 26.3),
    "Richards Bay": (-28.6, 32.1),
    "East London": (-33.0, 27.5),
    "Gqeberha": (-33.8, 25.3),
    "Oudtshoorn": (-33.6, 22.3)
}

print("Please choose a weather station from the following list:")
for idx, name in enumerate(stations, 1):
    print(f"{idx}. {name}")

choice = int(input("Enter the number corresponding to the station: "))
station_name = list(stations.keys())[choice - 1]
lat, lon = stations[station_name]

# === File Paths ===
spei_file_path = f"C:/StnData/StatisticalAnalysis/{station_name}_SPEI.csv"
oni_file_path = "C:/ENSO/ONI_classification_results.xlsx"
output_folder = f"C:/ENSO/{station_name}_Output"
os.makedirs(output_folder, exist_ok=True)

# === Load SPEI Data ===
spei_data = pd.read_csv(spei_file_path)
spei_data["YearMonth"] = pd.to_datetime(spei_data["YearMonth"], format="%Y%m")

# === Load ONI Data ===
oni_data = pd.read_excel(oni_file_path)
oni_data = oni_data.rename(columns={"DATE": "YearMonth", "Classification": "ENSO_Phase"})
oni_data["YearMonth"] = pd.to_datetime(oni_data["YearMonth"], format="%Y/%m/%d")

# === Filter Data ===
analysis_start_date = pd.Timestamp("1982-07-01")
spei_data = spei_data[spei_data["YearMonth"] >= analysis_start_date]
oni_data = oni_data[oni_data["YearMonth"] >= analysis_start_date]

# === Merge ===
merged_data = pd.merge(spei_data, oni_data[["YearMonth", "ENSO_Phase"]], on="YearMonth", how="inner")

# === Define Seasons ===
seasons = {
    "SON": 11,
    "DJF": 2,
    "MAM": 5,
}

# === Compile Results ===
results = []
for season_name, oni_month in seasons.items():
    season_spei = merged_data[merged_data["YearMonth"].dt.month == oni_month]
    for _, row in season_spei.iterrows():
        results.append({
            "YearMonth": row["YearMonth"],
            "Season": season_name,
            "SPEI_3": row["SPEI_3"],
            "ENSO_Phase": row["ENSO_Phase"]
        })

results_df = pd.DataFrame(results)

# === Categorize SPEI ===
def categorize_spei(value):
    if value <= -1.0:
        return "Dry"
    elif value >= 1.0:
        return "Wet"
    else:
        return "Near-Normal"

results_df["Rainfall_Category"] = results_df["SPEI_3"].apply(categorize_spei)

# === Group and Normalize ===
grouped = results_df.groupby(["Season", "ENSO_Phase", "Rainfall_Category"]).size()
index = pd.MultiIndex.from_product(
    [results_df["Season"].unique(), ["EN", "N", "LN"], ["Dry", "Near-Normal", "Wet"]],
    names=["Season", "ENSO_Phase", "Rainfall_Category"]
)
grouped = grouped.reindex(index, fill_value=0)

probabilities = grouped.unstack(fill_value=0)
probabilities = probabilities.div(probabilities.sum(axis=1), axis=0).fillna(0)
probabilities_reset = probabilities.reset_index()
probabilities_reset["ENSO_Phase"] = pd.Categorical(
    probabilities_reset["ENSO_Phase"], categories=["EN", "N", "LN"], ordered=True
)
probabilities_reset = probabilities_reset.sort_values(by=["Season", "ENSO_Phase"])

# === Chi-square ===
chi2_results = []
for season in results_df["Season"].unique():
    contingency = grouped.loc[season].unstack()
    contingency += 0.5  # to avoid 0 frequency
    chi2, p, dof, expected = chi2_contingency(contingency)
    chi2_results.append({
        "Season": season,
        "Chi2": chi2,
        "p-value": p,
        "Degrees of Freedom": dof,
        "Expected Frequencies": expected.tolist()
    })
chi2_results_df = pd.DataFrame(chi2_results)

# === Save Outputs ===
results_df.to_csv(os.path.join(output_folder, f"{station_name}_ENSO_SPEI_3Month_Results.csv"), index=False)
probabilities_reset.to_csv(os.path.join(output_folder, f"{station_name}_ENSO_SPEI_3Month_Probabilities.csv"), index=False)
chi2_results_df.to_csv(os.path.join(output_folder, f"{station_name}_ENSO_SPEI_3Month_ChiSquare.csv"), index=False)

# === Plot Pie Charts ===
colors = ["red", "lightgrey", "cornflowerblue"]
legend_labels = ["Dry", "Near-Normal", "Wet"]

for season in results_df['Season'].unique():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, enso_phase in enumerate(["EN", "N", "LN"]):
        row = probabilities_reset[
            (probabilities_reset['Season'] == season) &
            (probabilities_reset['ENSO_Phase'] == enso_phase)
        ]
        if not row.empty:
            data = row[["Dry", "Near-Normal", "Wet"]].values[0]
            wedges, texts, autotexts = axes[i].pie(
                data, labels=None, colors=colors,
                autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
                startangle=90, textprops=dict(color="black", fontweight='bold')
            )
            non_zero_segments = np.count_nonzero(data > 0)
            small_idxs = [j for j, v in enumerate(data) if v < 0.10 and v > 0]

            for j, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
                if data[j] <= 0:
                    continue
                if non_zero_segments == 1:
                    autotext.set_position((0, 0))
                    autotext.set_ha("center")
                    autotext.set_va("center")
                elif data[j] > 0.3:
                    angle = 0.5 * (wedge.theta2 + wedge.theta1)
                    radius = 0.6
                    x = radius * np.cos(np.deg2rad(angle))
                    y = radius * np.sin(np.deg2rad(angle))
                    autotext.set_position((x, y))
                elif j in small_idxs:
                    # Check previous small slice
                    prev_idx = small_idxs.index(j)
                    factor = 1.5 if prev_idx % 2 == 0 else 1.0
                    x, y = autotext.get_position()
                    autotext.set_position((x * factor, y * factor))
                else:
                    x, y = autotext.get_position()
                    autotext.set_position((x * 1.2, y * 1.2))
            axes[i].set_title(enso_phase)
        else:
            axes[i].axis("off")
    fig.legend(labels=legend_labels, loc="upper right", frameon=False)
    pval = chi2_results_df[chi2_results_df['Season'] == season]['p-value'].values[0]
    plt.suptitle(f"{season}  (p = {pval:.3f})")
    plt.savefig(os.path.join(output_folder, f"{station_name}_{season}_PieCharts.png"))
    plt.close()


