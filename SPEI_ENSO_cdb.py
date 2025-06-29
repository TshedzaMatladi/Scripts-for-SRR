# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 10:07:41 2025

@author: MatladiT
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# === PATHS ===
oni_path = r"C:\Users\matladit\ONI_ENSO\ONI2024.xlsx"
spei_dir = r"C:\Users\matladit\StnData\StnData\StatisticalAnalysis\CBD_SPEI"
output_dir = r"C:\Users\matladit\ONI_ENSO\PieChart_Tables"
os.makedirs(output_dir, exist_ok=True)

# === SEASON DEFINITIONS ===
seasons_3m = {'Spring (SON)': [9, 10, 11], 'Summer (DJF)': [12, 1, 2], 'Autumn (MAM)': [3, 4, 5]}
seasons_6m = {'Early Summer (SONDJF)': [9, 10, 11, 12, 1, 2], 'Late Summer (DJFMAM)': [12, 1, 2, 3, 4, 5]}
seasons_12m = {'Water Year (JASONDJFMAMJ)': list(range(7, 13)) + list(range(1, 7))}
all_seasons = {**seasons_3m, **seasons_6m, **seasons_12m}

# === COLOUR MAP ===
color_map = {'Dry': 'red', 'Wet': '#66b3ff', 'Normal': 'gray'}

# === READ ONI DATA ===
oni_df = pd.read_excel(oni_path)
oni_df['date'] = pd.to_datetime(oni_df['date'])
oni_df = oni_df[oni_df['date'].dt.year >= 1979]
oni_df['YearMonth'] = oni_df['date'].dt.strftime('%Y%m')
oni_df['ENSO_3m'] = np.where(oni_df['3m'] >= 0.5, 'El Nino', np.where(oni_df['3m'] <= -0.5, 'La Nina', 'Neutral'))
oni_df['ENSO_6m'] = np.where(oni_df['6m'] >= 0.5, 'El Nino', np.where(oni_df['6m'] <= -0.5, 'La Nina', 'Neutral'))
oni_df['ENSO_12m'] = np.where(oni_df['12m'] >= 0.5, 'El Nino', np.where(oni_df['12m'] <= -0.5, 'La Nina', 'Neutral'))

# === STATION FILES ===
station_files = {
    "Polokwane": ['ClimateTimeSeries_AgERA5_-23.8_29.7.csv'],
    "Mbombela": ['ClimateTimeSeries_AgERA5_-25.4_30.9.csv'],
    "Potchefstroom": ['ClimateTimeSeries_AgERA5_-26.7_27.1.csv'],
    "Bloemfontein": ['ClimateTimeSeries_AgERA5_-28.9_26.3.csv'],
    "Richards Bay": ['ClimateTimeSeries_AgERA5_-28.6_32.1.csv'],
    "East London": ['ClimateTimeSeries_AgERA5_-33.0_27.8.csv'],
    "Gqeberha": ['ClimateTimeSeries_AgERA5_-33.8_25.3.csv']
}

# === CHI-SQUARE TEST ===
def is_significant(df):
    if df.empty or df['SPEI_Group'].nunique() < 2:
        return False, None
    table = pd.crosstab(df['ENSO'], df['SPEI_Group'])
    if table.shape[0] < 2 or table.shape[1] < 2 or (table.values == 0).any():
        return False, None
    chi2, p, _, _ = chi2_contingency(table)
    return p < 0.05, round(p, 3)

# === PLOTTING FUNCTION ===
def plot_station_pie_table(station, df):
    fig, axes = plt.subplots(nrows=len(all_seasons), ncols=3, figsize=(12, 16))
    
    # Pie Charts
    for row, (season, months) in enumerate(all_seasons.items()):
        scale = 'SPEI_3' if season in seasons_3m else 'SPEI_6' if season in seasons_6m else 'SPEI_12'
        enso_col = 'ENSO_3m' if scale == 'SPEI_3' else 'ENSO_6m' if scale == 'SPEI_6' else 'ENSO_12m'
        season_df = df[df['Month'].isin(months)].copy()
        season_df['ENSO'] = season_df[enso_col]

        for col, phase in enumerate(['El Nino', 'Neutral', 'La Nina']):
            ax = axes[row, col]
            pie_df = season_df[season_df['ENSO'] == phase].copy()

            if pie_df.empty or scale not in pie_df.columns:
                ax.axis('off')
                continue

            pie_df['SPEI_Group'] = pd.cut(pie_df[scale], [-np.inf, -0.5, 0.5, np.inf],
                                          labels=['Dry', 'Normal', 'Wet'])
            counts = pie_df['SPEI_Group'].value_counts().reindex(['Dry', 'Wet', 'Normal'], fill_value=0)
            counts = counts[counts > 0]

            wedges, texts, autotexts = ax.pie(
                counts,
                colors=[color_map[k] for k in counts.index],
                startangle=90,
                autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else '',
                textprops={'color': 'black', 'weight': 'bold'}
            )

            ax.set_title(f"({chr(97 + col)}) {phase}", fontsize=9)
            if col == 0:
                ax.text(-1.3, 0, season, va='center', ha='right', fontsize=10, weight='bold')

            sig, p_val = is_significant(pie_df)
            if sig:
                ax.text(0.0, -1.3, f"* (p={p_val})", ha='center', fontsize=8, color='black')

    # Legend
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=k,
                                 markerfacecolor=v, markersize=10) for k, v in color_map.items()]
    fig.legend(handles=legend_patches, loc='upper center', ncol=3, frameon=False, fontsize=10, title="Colours")

    # Station name at bottom
    fig.text(0.5, 0.01, station, ha='center', fontsize=12, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{station}_SPEI_Seasonal_PieCharts.png"))
    plt.close()
    print(f"Saved: {station}")

# === MAIN SCRIPT ===
for station, files in station_files.items():
    station_dfs = []
    for fname in files:
        fpath = os.path.join(spei_dir, fname)
        if not os.path.exists(fpath):
            print(f"Missing file: {fpath}")
            continue
        df = pd.read_csv(fpath)
        df['YearMonth'] = df['YearMonth'].astype(str)
        df['Month'] = pd.to_datetime(df['YearMonth'], format='%Y%m').dt.month
        merged = pd.merge(df, oni_df[['YearMonth', 'ENSO_3m', 'ENSO_6m', 'ENSO_12m']], on='YearMonth', how='left')
        station_dfs.append(merged)
    if station_dfs:
        full_df = pd.concat(station_dfs, ignore_index=True)
        plot_station_pie_table(station, full_df)
