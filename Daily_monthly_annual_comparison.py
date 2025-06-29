# -*- coding: utf-8 -*-
"""
Created on Sat 21 Dec 2024
Input: C:/StnData/StatisticalAnalysis/Merged_StationName_stn_cdb.csv, 
monthly_StationName.csv, annual_StationName.csv
This script reads csv file containing daily/monthly/annual station (stn)
and reanalysis (cdb) data for Tmax, Tmin, RHmax, RHmin, WSmean, SR, 
Pr and ETo. It excludes missing (NaN) values and calculates a suite of
appropriate statistics (R2, Pearson's r, MAE, RMSE, Bias, ABIAS) and
produced daily/monthly/annual scatterplots for each variable.
Cumuative distribution functions (CDFs) are also plotted
for annual means/totals. Stats are printed to the console while plots 
are saved to C:\StnData\StatisticalAnalysis\Plots
    
@author: SteynAS@ufs.ac.za
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Station selection menu
stations = [
    "Polokwane", "Mbombela", "Potchefstroom", "Bloemfontein",
    "Richards Bay", "East London", "Gqeberha", "Oudtshoorn"
]

print("Please choose a weather station from the following list:")
for idx, station in enumerate(stations, 1):
    print(f"{idx}. {station}")

while True:
    try:
        station_choice = int(input("Enter the number corresponding to your choice: "))
        if 1 <= station_choice <= len(stations):
            selected_station = stations[station_choice - 1]
            print(f"You have selected: {selected_station}")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and", len(stations))
    except ValueError:
        print("Invalid input. Please enter a number.")

# File paths
base_path = r"C:\Users\matladit\StnData\StnData\StatisticalAnalysis"
daily_file = os.path.join(base_path, f"Merged_{selected_station.replace(' ', '')}_stn_cdb.csv")
monthly_file = os.path.join(base_path, f"monthly_{selected_station.replace(' ', '')}.csv")
annual_file = os.path.join(base_path, f"annual_{selected_station.replace(' ', '')}.csv")

# Load data with explicit date formats
daily_data = pd.read_csv(daily_file, parse_dates=["Date"], date_format="%Y%m%d", index_col="Date")
monthly_data = pd.read_csv(monthly_file, parse_dates=["date"], date_format="%Y%m", index_col="date")
annual_data = pd.read_csv(annual_file, parse_dates=["date"], date_format="%Y", index_col="date")

# Function to calculate regression statistics
def calculate_regression(data, stn_var, cdb_var):
    data_cleaned = data[[stn_var, cdb_var]].dropna()
    x = data_cleaned[stn_var]
    y = data_cleaned[cdb_var]

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = r2_score(y, y_pred)
    r, _ = pearsonr(x, y)
    mae = np.mean(np.abs(y - x))
    rmse = np.sqrt(np.mean((y - x) ** 2))
    bias = np.mean(y - x)
    n = len(data_cleaned)
    abias = (np.abs(bias) / np.mean(x)) * 100 if np.mean(x) != 0 else np.nan  # Relative Absolute Bias
    return slope, intercept, r2, r, mae, rmse, bias, abias, n

def plot_scatter(data, stn_var, cdb_var, variable, selected_station, stats_values):
    units = {
        "Tmax": "°C", "Tmin": "°C", "RHmax": "%", "RHmin": "%",
        "WSmean": "ms⁻¹", "SR": "MJ/m²", "Pr": "mm", "ETo": "mm"
    }
    unit = units.get(variable, "")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)
    time_scales = ['Daily', 'Monthly', 'Annual']

    for i, (resample_rule, ax) in enumerate(zip(['D', 'ME', 'YE'], axes)):
        if resample_rule == 'D':
            resampled_data = data[[stn_var, cdb_var]].dropna()
        else:
            resampled_data = data[[stn_var, cdb_var]].resample(resample_rule).mean().dropna()

        x = resampled_data[stn_var]
        y = resampled_data[cdb_var]
        _, _, r2, r, _, _, _, _, n = stats_values[i]

        ax.scatter(x, y, alpha=0.6, edgecolors='k')
        ax.plot(ax.get_xlim(), ax.get_xlim(), color='g', linestyle=':', label='1:1 Line')
        ax.set_title(f'{time_scales[i]} Comparison')
        ax.set_xlabel(f'{stn_var} ({unit})')
        ax.set_ylabel(f'{cdb_var} ({unit})')
        ax.text(0.05, 0.95, f'R²={r2:.2f}\nr={r:.2f}\nn={n}', 
                transform=ax.transAxes, fontsize=10, color='grey', va='top')

    plt.tight_layout()
    plot_path = f"C:/Users/matladit/StnData/StnData/StatisticalAnalysis/Plots/scatter_{variable}_{selected_station.replace(' ', '_')}.png"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Save the plot
    plt.savefig(plot_path)
    plt.close()
    print(f"Scatterplot saved as {plot_path}.")

def plot_annual_cdfs(annual_data, variable, selected_station):
    """
    Plot CDFs of annual means/totals for stn_ and cdb_ variables with fixed x-axis scales and custom tick intervals.

    Parameters:
        annual_data (DataFrame): Annual data containing stn_ and cdb_ variables.
        variable (str): Variable name (e.g., "Pr", "Tmax").
        selected_station (str): Name of the weather station.
    """
    # Extract the variables
    stn_var = f"stn_{variable}"
    cdb_var = f"cdb_{variable}"

    if stn_var in annual_data.columns and cdb_var in annual_data.columns:
        # Drop NaNs and sort data
        stn_data = annual_data[stn_var].dropna().sort_values()
        cdb_data = annual_data[cdb_var].dropna().sort_values()

        # Calculate cumulative probabilities using x / (n + 1)
        stn_cumulative_probs = np.arange(1, len(stn_data) + 1) / (len(stn_data) + 1)
        cdb_cumulative_probs = np.arange(1, len(cdb_data) + 1) / (len(cdb_data) + 1)

        # Captions and units for variables
        captions_units = {
            "Tmax": ("Mean annual maximum temperature", "°C"),
            "Tmin": ("Mean annual minimum temperature", "°C"),
            "RHmax": ("Mean annual maximum relative humidity", "%"),
            "RHmin": ("Mean annual minimum relative humidity", "%"),
            "WSmean": ("Mean annual wind speed", "ms⁻¹"),
            "SR": ("Mean annual total solar radiation", "MJ/m²"),
            "Pr": ("Mean annual total precipitation", "mm"),
            "ETo": ("Mean annual total reference evapotranspiration", "mm"),
        }
        caption, unit = captions_units.get(variable, (variable, ""))

        # Fixed x-axis limits for each variable
        fixed_scales = {
            "Tmax": (15, 35),
            "Tmin": (5, 25),
            "RHmax": (0, 100),
            "RHmin": (0, 100),
            "WSmean": (0, 5),
            "SR": (0, 10000),
            "Pr": (0, 2000),
            "ETo": (0, 2000),
        }
        x_min, x_max = fixed_scales.get(variable, (None, None))

        # Plot CDFs
        plt.figure(figsize=(10, 6))
        plt.plot(
            stn_data, stn_cumulative_probs, label=f"stn_{variable}", 
            color='black', linestyle='-', marker='o'
        )
        plt.plot(
            cdb_data, cdb_cumulative_probs, label=f"cdb_{variable}", 
            color='green', linestyle='--', marker='^'
        )

        # Title and axis labels
        plt.title(f"{caption} ({selected_station})", fontsize=14)
        plt.xlabel(f"{caption} ({unit})", fontsize=12)
        plt.ylabel("Probability of Non-Exceedance", fontsize=12)

        # Set y-axis scale and ticks
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, 0.1))  # Intervals of 0.1

        # Set fixed x-axis limits and custom ticks
        if x_min is not None and x_max is not None:
            plt.xlim(x_min, x_max)
            if variable in ["Tmax", "Tmin"]:  # Use integers for Tmax and Tmin
                plt.xticks(np.arange(x_min, x_max + 1, 1))
            elif variable in ["RHmin", "RHmax"]:  # Use ticks every 10 for RHmin and RHmax
                plt.xticks(np.arange(x_min, x_max + 1, 10))
            elif variable == "SR":  # Use ticks every 1000 for SR
                plt.xticks(np.arange(x_min, x_max + 1, 1000))

        # Add grid, legend, and layout
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10, loc="upper left")  # Legend in top-left corner
        plt.tight_layout()

        # Save the plot
        plot_path = f"C:/Users/matladit/StnData/StnData/StatisticalAnalysis/Plots/cdf_annual_{variable}_{selected_station.replace(' ', '_')}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"CDF plot saved as {plot_path}.")
    else:
        print(f"Variable {variable} not found in annual data.")

# Perform analysis
variables = ["Tmax", "Tmin", "RHmax", "RHmin", "WSmean", "SR", "Pr", "ETo"]
print("\n--- Daily, Monthly, and Annual Analysis ---")
for variable in variables:
    stn_var = f"stn_{variable}"
    cdb_var = f"cdb_{variable}"

    if stn_var in daily_data.columns and cdb_var in daily_data.columns:
        print(f"\nAnalyzing variable: {variable}")

        # Daily
        daily_stats = calculate_regression(daily_data, stn_var, cdb_var)

        # Monthly
        monthly_stats = calculate_regression(monthly_data, stn_var, cdb_var)

        # Annual
        annual_stats = calculate_regression(annual_data, stn_var, cdb_var)

        # Print results
        for scale, stats in zip(["Daily", "Monthly", "Annual"], [daily_stats, monthly_stats, annual_stats]):
            slope, intercept, r2, r, mae, rmse, bias, abias, n = stats
            print(f"{scale}: Slope={slope:.3f}, Intercept={intercept:.3f}, R²={r2:.3f}, r={r:.3f}, "
                  f"MAE={mae:.3f}, RMSE={rmse:.3f}, Bias={bias:.3f}, ABIAS={abias:.2f}%, n={n}")

        # Create scatterplots
        stats_values = [daily_stats, monthly_stats, annual_stats]
        plot_scatter(daily_data, stn_var, cdb_var, variable, selected_station, stats_values)

# Generate CDF plots for annual means/totals
print("\n--- Generating CDF Plots for Annual Means/Totals ---")
for variable in ["Tmax", "Tmin", "RHmax", "RHmin", "WSmean", "SR", "Pr", "ETo"]:
    plot_annual_cdfs(annual_data, variable, selected_station)
