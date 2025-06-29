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
C:\StnData\StatisticalAnalysis\{lat}_{lon}_SPEI.csv
Time series are plotted as SPEI_3_{lat}_{lon}_Plot.png etc. and 
All_SPEI_{lat}_{lon}_Plot.png under C:\StnData\StatisticalAnalysis\Plots

Must still alter the plotting section to be similar to SPEI_cdb.py
    
@author: SteynAS@ufs.ac.za
"""

import pandas as pd
import numpy as np
from scipy.stats import pearson3, norm
import os
import matplotlib.pyplot as plt

# Function to calculate SPEI
def calculate_spei_custom(data, timescale):
    """
    Calculate SPEI using rolling sums and a Pearson Type III distribution.

    Parameters:
    - data (pd.DataFrame): DataFrame with a 'WaterBalance' column.
    - timescale (int): The timescale for SPEI (e.g., 3, 6, 12 months).

    Returns:
    - pd.Series: SPEI values for the given timescale.
    """
    # Calculate rolling sum for the water balance
    rolling_balance = data['WaterBalance'].rolling(timescale, min_periods=timescale).sum()

    # Fit a Pearson Type III distribution to the rolling sums
    spei_values = []
    for i in range(len(rolling_balance)):
        if i < timescale - 1:  # Not enough data to calculate
            spei_values.append(np.nan)
        else:
            rolling_window = rolling_balance[: i + 1]  # Data up to current point
            params = pearson3.fit(rolling_window.dropna())  # Fit distribution
            standardized = pearson3.cdf(rolling_balance[i], *params)  # Convert to probability
            spei_values.append(norm.ppf(standardized))  # Convert to Z-score

    return pd.Series(spei_values, index=data.index)

# Function to plot SPEI
def plot_spei(data, column, lat, lon, output_dir):
    """
    Plot SPEI over time.

    Parameters:
    - data (pd.DataFrame): DataFrame containing SPEI values and dates.
    - column (str): Column name for the SPEI values to plot.
    - lat (float): Latitude for the caption.
    - lon (float): Longitude for the caption.
    - output_dir (str): Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 7))  # Unified layout for all plots
    ax.plot(data['YearMonth'], data[column], color='green')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylim(-5, 5)
    ax.set_ylabel("Standardized Precipitation Evapotranspiration Index (SPEI)")
    data['FormattedYearMonth'] = pd.to_datetime(data['YearMonth'], format='%Y%m').dt.strftime('%Y-%m')
    ax.set_xticks(range(len(data['YearMonth'])))
    ax.set_xticklabels([date if int(date[-2:]) in [1, 7] else '' for date in data['FormattedYearMonth']], rotation=90)
    ax.set_yticks(np.arange(-5, 6, 1))
    ax.tick_params(axis='x', pad=5)  # Adjust padding for label clarity
    ax.set_xlim(left=0, right=len(data['YearMonth']) - 1)  # Align first x-axis marker with y-axis
    ax.set_title(f"{column.replace('SPEI_', '').replace('3', '3-month').replace('6', '6-month').replace('12', '12-month')} SPEI for lat = {lat}, lon = {lon}")
    fig.tight_layout()  # Ensure consistent layout
    plt.savefig(os.path.join(output_dir, f"{column}_{lat}_{lon}_Plot.png"))
    plt.close()

# Function to plot all SPEI timescales on one plot
def plot_all_spei(data, lat, lon, output_dir):
    """
    Plot all SPEI timescales on one plot.

    Parameters:
    - data (pd.DataFrame): DataFrame containing SPEI values and dates.
    - lat (float): Latitude for the caption.
    - lon (float): Longitude for the caption.
    - output_dir (str): Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 7))  # Unified layout for all plots
    ax.plot(data['YearMonth'], data['SPEI_3'], color='red', linestyle='-', label='3m SPEI')
    ax.plot(data['YearMonth'], data['SPEI_6'], color='green', linestyle='--', label='6m SPEI')
    ax.plot(data['YearMonth'], data['SPEI_12'], color='blue', linestyle='-.', label='12m SPEI')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylim(-5, 5)
    ax.set_ylabel("Standardized Precipitation Evapotranspiration Index (SPEI)")
    data['FormattedYearMonth'] = pd.to_datetime(data['YearMonth'], format='%Y%m').dt.strftime('%Y-%m')
    ax.set_xticks(range(len(data['YearMonth'])))
    ax.set_xticklabels([date if int(date[-2:]) in [1, 7] else '' for date in data['FormattedYearMonth']], rotation=90)
    ax.set_yticks(np.arange(-5, 6, 1))
    ax.tick_params(axis='x', pad=5)  # Adjust padding for label clarity
    ax.set_xlim(left=0, right=len(data['YearMonth']) - 1)  # Align first x-axis marker with y-axis
    ax.set_title(f"SPEI for lat = {lat}, lon = {lon}")
    ax.legend()
    fig.tight_layout()  # Ensure consistent layout
    plt.savefig(os.path.join(output_dir, f"All_SPEI_{lat}_{lon}_Plot.png"))
    plt.close()

# Input prompts
lat = round(float(input("Enter the latitude of the location (e.g., -20.1): ")), 1)
lon = round(float(input("Enter the longitude of the location (e.g., 15.1): ")), 1)
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
output_file = f"C:\\StnData\\StatisticalAnalysis\\{lat}_{lon}_SPEI.csv"
monthly_data.to_csv(output_file, index=False)
print(f"SPEI values saved to: {output_file}")

# Create plots directory
plots_dir = "C:\\StnData\\StatisticalAnalysis\\Plots"
os.makedirs(plots_dir, exist_ok=True)

# Generate and save plots
plot_spei(monthly_data, 'SPEI_3', lat, lon, plots_dir)
plot_spei(monthly_data, 'SPEI_6', lat, lon, plots_dir)
plot_spei(monthly_data, 'SPEI_12', lat, lon, plots_dir)
plot_all_spei(monthly_data, lat, lon, plots_dir)
