# -*- coding: utf-8 -*-
"""
This script uses the ENSO classification from ONI_classification_results.xlsx and the 
12-month SPEI analysis from {station_name}_SPEI.csv to calculate probabilities by dividing the
number of dry/near-normal/wet occurrences for each station by the number of EN/N/LN events.
The user must choose between classifying the ENSO phase according the peak ONI (turning point)
or a particular representative month which is set in line 112
This particular script produces a combined plot for JASONDJFMAMJ.
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

print("\nChoose ENSO classification method:")
print("1. Use most extreme ONI value during most important months in 12-month period")
print("2. Use a representative month for classification (March)")
method_choice = int(input("Enter 1 or 2: "))
exclude_transitions = False
if method_choice == 1:
    choice = input("Exclude extreme transition years (El Niño to La Niña or vice versa in a single year)? (y/n): ").strip().lower()
    exclude_transitions = (choice == 'y')

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

# === Define 12-Month Period: JASONDJFMAMJ ===
months_12m = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]  # Jul to Jun

results = []
extreme_log = []

for year in spei_data["YearMonth"].dt.year.unique():
    months = [(year if m >= 7 else year + 1, m) for m in months_12m]
    period_months = [pd.Timestamp(f"{y}-{m:02d}-01") for y, m in months]

    spei_subset = spei_data[spei_data["YearMonth"].isin(period_months)]
    oni_subset = oni_data[oni_data["YearMonth"].isin(period_months)]

    if spei_subset.empty or oni_subset.empty:
        continue
    
    # Check for extreme transition years (EN <-> LN within Jul–Jun)
    if exclude_transitions:
        phases_in_year = oni_subset["ENSO_Phase"].unique()
        if "EN" in phases_in_year and "LN" in phases_in_year:
            print(f"Excluded transition year {year} (contains both EN and LN)")
            continue

    if method_choice == 1:
        #Old version applied extreme ONI-value logic to entire season
        #max_abs_oni = oni_subset.copy()
        #max_abs_oni["abs"] = max_abs_oni["ANOM"].abs()
        #max_month = max_abs_oni.sort_values("abs", ascending=False).iloc[0]["YearMonth"]
        #enso_phase = oni_data[oni_data["YearMonth"] == max_month]["ENSO_Phase"].values[0]
        # New version uses configurable core ENSO period: Sep–Mar
        core_months_list = [9, 10, 11, 12, 1, 2, 3]
        core_months = oni_subset[
            oni_subset["YearMonth"].dt.month.isin(core_months_list)
        ].copy()     
        if core_months.empty:
            print(f"Skipped year {year} — no ONI data in core period {core_months_list}")
            continue
        core_months["abs"] = core_months["ANOM"].abs()
        max_month = core_months.sort_values("abs", ascending=False).iloc[0]["YearMonth"]
        enso_phase = oni_data[oni_data["YearMonth"] == max_month]["ENSO_Phase"].values[0]
    else:
        rep_year = year + 1  # March is in the following year
        rep_date = pd.Timestamp(f"{rep_year}-03-01")
        match = oni_data[oni_data["YearMonth"] == rep_date]
        if match.empty:
            print(f"Skipped year {year} — no ONI data in representative month")
            continue
        enso_phase = match["ENSO_Phase"].values[0]
        max_month = rep_date

    last_month = period_months[-1]
    spei_value = spei_subset[spei_subset["YearMonth"] == last_month]["SPEI_12"].values

    if len(spei_value) == 0:
        continue

    results.append({
        "YearMonth": last_month,
        "Season": "JASONDJFMAMJ",
        "SPEI_12": spei_value[0],
        "ENSO_Phase": enso_phase
    })

    if method_choice == 1:
        extreme_log.append({
            "YearMonth": last_month,
            "ExtremeONIMonth": max_month,
            "ENSO_Phase": enso_phase
        })

if method_choice == 1 and extreme_log:
    pd.DataFrame(extreme_log).to_csv(os.path.join(output_folder, "extremeONI_12m.csv"), index=False)

# === Analysis and Plotting ===
results_df = pd.DataFrame(results)
results_df["Rainfall_Category"] = results_df["SPEI_12"].apply(lambda x: "Dry" if x <= -1.0 else "Wet" if x >= 1.0 else "Near-Normal")

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
probabilities_reset["ENSO_Phase"] = pd.Categorical(probabilities_reset["ENSO_Phase"], categories=["EN", "N", "LN"], ordered=True)
probabilities_reset = probabilities_reset.sort_values(by=["Season", "ENSO_Phase"])

# === Chi-square ===
chi2_results = []
for season in results_df["Season"].unique():
    contingency = grouped.loc[season].unstack()
    contingency += 0.5
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
results_df.to_csv(os.path.join(output_folder, f"{station_name}_ENSO_SPEI_12Month_Results.csv"), index=False)
probabilities_reset.to_csv(os.path.join(output_folder, f"{station_name}_ENSO_SPEI_12Month_Probabilities.csv"), index=False)
chi2_results_df.to_csv(os.path.join(output_folder, f"{station_name}_ENSO_SPEI_12Month_ChiSquare.csv"), index=False)

# === Combined Plot ===
colors = ["red", "lightgrey", "cornflowerblue"]
legend_labels = ["Dry", "Near-Normal", "Wet"]
enso_labels = {"EN": "El Niño", "N": "Neutral", "LN": "La Niña"}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
season = "JASONDJFMAMJ"

for col_idx, enso_key in enumerate(["EN", "N", "LN"]):
    ax = axes[col_idx]
    row = probabilities_reset[(probabilities_reset['Season'] == season) &
                              (probabilities_reset['ENSO_Phase'] == enso_key)]
    if not row.empty:
        data = row[["Dry", "Near-Normal", "Wet"]].values[0]
        wedges, texts, autotexts = ax.pie(
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
                idx_pos = small_idxs.index(j)
                factor = 1.5 if idx_pos % 2 == 0 else 1.0
                x, y = autotext.get_position()
                autotext.set_position((x * factor, y * factor))
            else:
                x, y = autotext.get_position()
                autotext.set_position((x * 1.2, y * 1.2))
    else:
        ax.axis("off")

    ax.set_title(enso_labels[enso_key], fontweight='bold')
    if col_idx == 0:
        pval = chi2_results_df[chi2_results_df['Season'] == season]['p-value'].values[0]
        ax.text(-1.3, 0.15, season, fontsize=12, fontweight='bold', va="center", ha="right")
        ax.text(-1.15, -0.15, f"(p = {pval:.3f})", fontsize=10, va="center", ha="right")

fig.legend(labels=legend_labels, loc="upper right", frameon=False)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(os.path.join(output_folder, f"{station_name}_ENSO_12Month_CombinedPieCharts.png"))
plt.close()