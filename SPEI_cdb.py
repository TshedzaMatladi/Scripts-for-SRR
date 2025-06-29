# -*- coding: utf-8 -*-
r"""
Created on Fri 3 Jan 2025
Input: C:/AgERA5/ClimDataBase/ClimateTimeSeries_AgERA5_{lat}_{lon}.csv, 
This script reads csv file containing daily reanalysis (cdb) data for
Tmax, Tmin, RHmax, RHmin, WSmean, SR, Pr and ETo. It uses the ETo as
proxy for Potential Evapotranspiration (PET) and Pr to calculate the 
water balance (Pr - PET). It applies a statistical standardization 
process to the water balance over a given time scale aggregation).
This involves fitting the water balance data to the Pearson Type III
probability distribution and deriving SPEI values from cumulative 
probabilities at 3-, 6- and 12-month timescales. This script is 
based on the R-script used by Beguería & Vicente-Serrano at
https://CRAN.R-project.org/package=SPEI 
3m, 6m, 12m SPEI values are output to 
C:\StnData\StatisticalAnalysis\{station_name}_SPEI.csv
Time series are plotted as SPEI_3_{station_name}_Plot.png etc. and 
All_SPEI_{station_name}_Plot.png under C:\StnData\StatisticalAnalysis\Plots

@author: SteynAS@ufs.ac.za
"""

import pandas as pd
import numpy as np
from scipy.stats import pearson3, norm
import os
import matplotlib.pyplot as plt

# Function to calculate SPEI
def calculate_spei_custom(data, timescale):
    rolling_balance = data['WaterBalance'].rolling(timescale, min_periods=timescale).sum()
    spei_values = []
    for i in range(len(rolling_balance)):
        if i < timescale - 1:
            spei_values.append(np.nan)
        else:
            rolling_window = rolling_balance[: i + 1]
            params = pearson3.fit(rolling_window.dropna())
            standardized = pearson3.cdf(rolling_balance[i], *params)
            spei_values.append(norm.ppf(standardized))
    return pd.Series(spei_values, index=data.index)

# Function to plot SPEI
def plot_spei(data, column, station_name, output_dir):
    filtered = data.iloc[30:].copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(filtered['YearMonth'], filtered[column], color='green')
    for threshold in [-2, -1, 1, 2]:
        ax.axhline(threshold, color='black', linestyle='--', linewidth=0.6)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylim(-5, 5)
    ax.set_ylabel("Standardized Precipitation Evapotranspiration Index (SPEI)")
    filtered['FormattedYearMonth'] = pd.to_datetime(filtered['YearMonth'], format='%Y%m').dt.strftime('%Y-%m')
    ax.set_xticks(range(len(filtered['YearMonth'])))
    ax.set_xticklabels([date if int(date[-2:]) in [1, 7] else '' for date in filtered['FormattedYearMonth']], rotation=90)
    ax.set_yticks(np.arange(-5, 6, 1))
    ax.tick_params(axis='x', pad=5)
    ax.set_xlim(left=0, right=len(filtered['YearMonth']) - 1)
    ax.set_title(f"{column.replace('SPEI_', '').replace('3', '3-month').replace('6', '6-month').replace('12', '12-month')} SPEI for {station_name}")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{column}_{station_name}_Plot.png"))
    plt.close()

# Function to plot all SPEI timescales on one plot
def plot_all_spei(data, station_name, output_dir):
    filtered = data.iloc[30:].copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(filtered['YearMonth'], filtered['SPEI_3'], color='red', linestyle='-', label='3m SPEI')
    ax.plot(filtered['YearMonth'], filtered['SPEI_6'], color='green', linestyle='--', label='6m SPEI')
    ax.plot(filtered['YearMonth'], filtered['SPEI_12'], color='blue', linestyle='-.', label='12m SPEI')
    for threshold in [-2, -1, 1, 2]:
        ax.axhline(threshold, color='black', linestyle='--', linewidth=0.6)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylim(-5, 5)
    ax.set_ylabel("Standardized Precipitation Evapotranspiration Index (SPEI)")
    filtered['FormattedYearMonth'] = pd.to_datetime(filtered['YearMonth'], format='%Y%m').dt.strftime('%Y-%m')
    ax.set_xticks(range(len(filtered['YearMonth'])))
    ax.set_xticklabels([date if int(date[-2:]) in [1, 7] else '' for date in filtered['FormattedYearMonth']], rotation=90)
    ax.set_yticks(np.arange(-5, 6, 1))
    ax.tick_params(axis='x', pad=5)
    ax.set_xlim(left=0, right=len(filtered['YearMonth']) - 1)
    ax.set_title(f"SPEI for {station_name}")
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"All_SPEI_{station_name}_Plot.png"))
    plt.close()

# Station selection
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

input_file_name = f"ClimateTimeSeries_AgERA5_{lat}_{lon}.csv"
input_file = f"C:\\AgERA5\\ClimDataBase\\{input_file_name}"
start_date = input("Enter the start date (YYYYMMDD): ")
end_date = input("Enter the end date (YYYYMMDD): ")

# Read and preprocess the data
data = pd.read_csv(input_file)
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Aggregate daily data to monthly
data['YearMonth'] = data['Date'].dt.to_period('M')
monthly_data = data.groupby('YearMonth').agg({'Pr': 'sum', 'ETo': 'sum'}).reset_index()
monthly_data['YearMonth'] = monthly_data['YearMonth'].dt.strftime('%Y%m')
monthly_data['WaterBalance'] = monthly_data['Pr'] - monthly_data['ETo']

# Calculate SPEI for 3, 6, and 12 months
monthly_data['SPEI_3'] = calculate_spei_custom(monthly_data, 3)
monthly_data['SPEI_6'] = calculate_spei_custom(monthly_data, 6)
monthly_data['SPEI_12'] = calculate_spei_custom(monthly_data, 12)

# Save to a new CSV file
output_file = f"C:\\StnData\\StatisticalAnalysis\\{station_name}_SPEI.csv"
monthly_data.to_csv(output_file, index=False)
print(f"SPEI values saved to: {output_file}")

# Create plots directory
output_dir = "C:\\StnData\\StatisticalAnalysis\\Plots"
os.makedirs(output_dir, exist_ok=True)

# Generate and save plots
plot_spei(monthly_data, 'SPEI_3', station_name, output_dir)
plot_spei(monthly_data, 'SPEI_6', station_name, output_dir)
plot_spei(monthly_data, 'SPEI_12', station_name, output_dir)
plot_all_spei(monthly_data, station_name, output_dir)

