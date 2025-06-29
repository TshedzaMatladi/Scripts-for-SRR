# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 09:17:04 2024
Input: C:/StnData/StatisticalAnalysis/Merged_StationName_stn_cdb.csv
This script reads a csv file containing daily station (stn) and
reanalysis (cdb) data for Tmax, Tmin, RHmax, RHmin, WSmean, SR, 
Pr and ETo. It applies a quality control threshold before aggregating 
monthly and annual means (Tmax, Tmin, RHmax, RHmin, WSmean) and 
totals (SR, Pr, ETo). An option is provided to calculate the 
annual aggregates according to water years (Jan-Dec or Jul-Jun).
Output: monthly_StationName.csv, annual_StationName.csv
    
@author: SteynAS@ufs.ac.za
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Weather station menu
stations = [
    "Polokwane", "Mbombela", "Potchefstroom", "Bloemfontein", 
    "Richards Bay", "East London", "Gqeberha", "Oudtshoorn"
]

# Display station selection menu
print("Please choose a weather station from the following list:")
for idx, station in enumerate(stations, 1):
    print(f"{idx}. {station}")

# Get station choice
while True:
    try:
        choice = int(input("Enter the number corresponding to your choice: "))
        if 1 <= choice <= len(stations):
            selected_station = stations[choice - 1]
            print(f"You have selected: {selected_station}")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and", len(stations))
    except ValueError:
        print("Invalid input. Please enter a number.")

# File path for the selected station
input_file = f"Merged_{selected_station.replace(' ', '')}_stn_cdb.csv"
file_path = os.path.join("C:/Users/matladit/StatisticalAnalysis", input_file)

# Prompt the user for the quality control threshold
try:
    user_threshold = float(input("Enter the quality control threshold (default is 60%): ") or 60)
except ValueError:
    print("Invalid input. Using the default threshold of 60%.")
    user_threshold = 60

# Load data for plotting
data = pd.read_csv(file_path)
data["Date"] = pd.to_datetime(data["Date"], format="%Y%m%d")
data.set_index("Date", inplace=True)

# Compute the long-term monthly mean precipitation totals
monthly_pr = data["stn_Pr"].resample("ME").sum()  # Resample to month-end totals
monthly_pr_means = monthly_pr.groupby(monthly_pr.index.month).mean()  # Group by month to compute long-term means

# Save and display the bar chart for long-term monthly Pr totals
plt.ioff()  # Turn off interactive plotting
plt.figure(figsize=(10, 6))
plt.bar(
    monthly_pr_means.index, monthly_pr_means,
    tick_label=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)
plt.title(f"Long-Term Monthly Mean Precipitation Totals ({selected_station})")
plt.xlabel("Month")
plt.ylabel("Mean Precipitation (mm)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Save the plot
plot_dir = os.path.join("C:/StnData/StatisticalAnalysis/Plots")
os.makedirs(plot_dir, exist_ok=True)  # Ensure the directory exists
plot_path = os.path.join(plot_dir, f"long_term_monthly_pr_{selected_station.replace(' ', '_')}.png")
plt.savefig(plot_path)
plt.close()  # Close the plot explicitly
print(f"Plot saved as {plot_path}.")

# Prompt the user to select the annual aggregation season
print("\nSelect the annual aggregation season:")
print("1. Calendar Year (Jan-Dec)")
print("2. Water Year (Jul-Jun)")
try:
    user_option = int(input("Enter 1 or 2 (default is 1): ") or 1)
    annual_season_option = "water" if user_option == 2 else "calendar"
except ValueError:
    print("Invalid input. Using the default: Calendar Year (Jan-Dec).")
    annual_season_option = "calendar"

# Define the output directory and station name
output_dir = r"C:/StnData/StatisticalAnalysis"
station_name = selected_station.replace(" ", "")

# Aggregate data function
def aggregate_data(input_path, output_path, station_name, threshold=60, annual_season="calendar"):
    # Convert threshold to decimal
    threshold = threshold / 100.0

    # Load data
    data = pd.read_csv(input_path)
    data["Date"] = pd.to_datetime(data["Date"], format="%Y%m%d")

    # Extract year and month
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["YearMonth"] = data["Date"].dt.strftime("%Y%m")  # Format as YYYYMM

    # Adjust for water year if selected
    if annual_season == "water":
        # Assign the end year of the water year (July to June)
        data["WaterYear"] = data["Date"].apply(lambda x: x.year + 1 if x.month >= 7 else x.year)
        annual_column = "WaterYear"
    else:
        annual_column = "Year"

    # Define variables for aggregation
    total_variables = ["stn_SR", "stn_Pr", "stn_ETo", "cdb_SR", "cdb_Pr", "cdb_ETo"]
    mean_variables = [
        "stn_Tmax", "stn_Tmin", "stn_RHmax", "stn_RHmin", "stn_WSmean",
        "cdb_Tmax", "cdb_Tmin", "cdb_RHmax", "cdb_RHmin", "cdb_WSmean"
    ]

    def aggregate(grouped, time_unit):
        results = []
        for name, group in grouped:
            # Determine actual days for the group
            start_date = group["Date"].min()
            end_date = group["Date"].max()
            total_days = (end_date - start_date).days + 1  # Inclusive of start and end

            row = {time_unit: name}

            # Total variables
            for var in total_variables:
                valid_days = group[var].notna().sum()
                if valid_days / total_days >= threshold:
                    row[var] = group[var].sum()
                else:
                    row[var] = np.nan

            # Mean variables
            for var in mean_variables:
                valid_days = group[var].notna().sum()
                if valid_days / total_days >= threshold:
                    row[var] = group[var].mean()
                else:
                    row[var] = np.nan

            results.append(row)
        return pd.DataFrame(results)

    # Perform monthly aggregation
    data["YearMonth"] = data["Date"].dt.strftime("%Y%m")  # Format as YYYYMM for monthly grouping
    monthly_grouped = data.groupby(data["YearMonth"])
    monthly_data = aggregate(monthly_grouped, "YearMonth")
    monthly_data.rename(columns={"YearMonth": "date"}, inplace=True)

    # Perform annual aggregation
    annual_grouped = data.groupby(data[annual_column])
    annual_data = aggregate(annual_grouped, annual_column)
    annual_data.rename(columns={annual_column: "date"}, inplace=True)

    # Additional quality control for the last year
    last_year = annual_data["date"].max()  # Get the last year in the aggregated data
    if not pd.isna(last_year):  # Ensure there's a valid last year
        last_year_data = data[data[annual_column] == last_year]  # Filter data for the last year
        total_days_in_year = 365  # Total days in a full year
        for var in total_variables + mean_variables:  # Apply to all variables
            valid_days = last_year_data[var].notna().sum()  # Count valid observations
            if valid_days < total_days_in_year * threshold:  # Check against the threshold
                annual_data.loc[annual_data["date"] == last_year, var] = np.nan  # Overwrite with NaN

    # Define column order
    columns = [
        "date", "stn_Tmax", "stn_Tmin", "stn_RHmax", "stn_RHmin", "stn_WSmean",
        "stn_SR", "stn_Pr", "stn_ETo", "cdb_Tmax", "cdb_Tmin", "cdb_RHmax",
        "cdb_RHmin", "cdb_WSmean", "cdb_SR", "cdb_Pr", "cdb_ETo"
    ]
    monthly_data = monthly_data.reindex(columns=columns)
    annual_data = annual_data.reindex(columns=columns)

    # Save outputs
    monthly_file = os.path.join(output_path, f"monthly_{station_name}.csv")
    annual_file = os.path.join(output_path, f"annual_{station_name}.csv")

    monthly_data.to_csv(monthly_file, index=False, na_rep="NaN")
    annual_data.to_csv(annual_file, index=False, na_rep="NaN")

    print(f"Monthly aggregated data saved to: {monthly_file}")
    print(f"Annual aggregated data saved to: {annual_file}")


# Call the function for aggregation
aggregate_data(file_path, output_dir, station_name, user_threshold, annual_season_option)
