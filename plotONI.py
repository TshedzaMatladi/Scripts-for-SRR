# -*- coding: utf-8 -*-
"""
This script plots ONI evolution for every water year (JASONDJFMAMJ over the summer rainfall region)

Created April 2025
@author: SteynAS@ufs.ac.za
"""

import pandas as pd
import matplotlib.pyplot as plt

# === File Path ===
oni_file_path = "C:/ENSO/ONI_classification_results.xlsx"

# === Load ONI Data ===
oni_df = pd.read_excel(oni_file_path)
oni_df["Date"] = pd.to_datetime(oni_df["DATE"], format="%Y/%m/%d %H:%M:%S")
oni_df["Year"] = oni_df["Date"].dt.year
oni_df["Month"] = oni_df["Date"].dt.month
oni_df["ANOM"] = oni_df["ANOM"].astype(float)

# === Assign Water Year: JASONDJFMAMJ ===
oni_df["WaterYear"] = oni_df.apply(
    lambda row: f"{row['Year']}/{row['Year']+1}" if row["Month"] >= 7 else f"{row['Year']-1}/{row['Year']}",
    axis=1
)

# === Filter for Water Years from 1982/1983 to 2023/2024 ===
oni_df = oni_df[(oni_df["WaterYear"] >= "1982/1983") & (oni_df["WaterYear"] <= "2023/2024")]

# === Month Order and Labels ===
month_order = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
month_labels = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]

# === Highlighted Water Years ===
highlight_years = ["1997/1998", "2015/2016", "2010/2011"]
highlight_colors = {
    "1997/1998": "red",
    "2015/2016": "orange",
    "2010/2011": "blue"
}

# === Plot ONI Values per Water Year ===
fig, ax = plt.subplots(figsize=(15, 8))
for water_year, group in oni_df.groupby("WaterYear"):
    group = group[group["Month"].isin(month_order)]
    group["MonthOrder"] = group["Month"].apply(lambda m: month_order.index(m))
    group_sorted = group.sort_values("MonthOrder")
    color = highlight_colors.get(water_year, "gray")
    alpha = 0.8 if water_year in highlight_years else 0.3
    linewidth = 2 if water_year in highlight_years else 1
    ax.plot(
        month_labels,
        group_sorted["ANOM"],
        label=water_year if water_year in highlight_years else "",
        color=color,
        alpha=alpha,
        linewidth=linewidth
    )

ax.set_title("Monthly ONI Values per Water Year (1982/1983 - 2023/2024)")
ax.set_ylabel("ONI Anomaly")
ax.set_xlabel("Month")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend(title="Highlighted Years")
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()