# -*- coding: utf-8 -*-
"""
Script to merge weather station data and climate database data for statistical analysis.

- Supports multiple station files (stn) with respective climate database (cdb) files.
- Uses the closest climate database data for each station based on longitude and latitude.
Input: StationName_combined_cleaned.xlsx, ClimateTimeSeries_AgERA5_coordinates.csv
Output: Merged_StationName_stn_cdb.csv

@author: SteynAS@ufs.ac.za
"""

import os
import pandas as pd

# Define paths
stn_dir = r'C:\Users\matladit\StnData\QControlled'
cdb_dir = r'C:\Users\matladit\AgERA5\ClimDataBase'
output_dir = r'C:\Users\matladit/StatisticalAnalysis'

# Define the directory path
directory = r'C:\Users\matladit\StatisticalAnalysis'

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)


# Menu options for stations and corresponding mappings
station_options = {
    1: ("Polokwane", "Polokwane_combined_cleaned.xlsx", {
        (29.69368, -23.83615): 'ClimateTimeSeries_AgERA5_-23.8_29.7.csv',
        (29.693475, -23.836094): 'ClimateTimeSeries_AgERA5_-23.8_29.7.csv'
    }),
    2: ("Mbombela", "Mbombela_combined_cleaned.xlsx", {
        (30.85000038, -25.35000038): 'ClimateTimeSeries_AgERA5_-25.4_30.9.csv',
        (30.97159, -25.45455): 'ClimateTimeSeries_AgERA5_-25.5_31.0.csv'
    }),
    3: ("Potchefstroom", "Potchefstroom_combined_cleaned.xlsx", {
        (27.08333397, -26.73333359): 'ClimateTimeSeries_AgERA5_-26.7_27.1.csv',
        (27.07553, -26.73607): 'ClimateTimeSeries_AgERA5_-26.7_27.1.csv'
    }),
    4: ("Bloemfontein", "GlenCollege_combined_cleaned.xlsx", {
        (26.35000038, -28.95000076): 'ClimateTimeSeries_AgERA5_-28.9_26.3.csv',
        (26.32633, -28.92957): 'ClimateTimeSeries_AgERA5_-28.9_26.3.csv',
        (26.32631, -28.92957): 'ClimateTimeSeries_AgERA5_-28.9_26.3.csv'
    }),
    5: ("Richards Bay", "RichardsBay_combined_cleaned.xlsx", {
        (31.89813, -28.72496): 'ClimateTimeSeries_AgERA5_-28.7_31.9.csv',
        (32.06295, -28.63707): 'ClimateTimeSeries_AgERA5_-28.6_32.1.csv'
    }),
    6: ("East London", "EastLondon_combined_cleaned.xlsx", {
        (27.83333397, -33.03333282): 'ClimateTimeSeries_AgERA5_-33.0_27.8.csv',
        (27.5, -33.03333282): 'ClimateTimeSeries_AgERA5_-33.0_27.5.csv'
    }),
    7: ("Gqeberha", "PortElizabeth_combined_cleaned.xlsx", {
        (25.31666756, -33.76666641): 'ClimateTimeSeries_AgERA5_-33.8_25.3.csv',
        (25.327288, -33.774138): 'ClimateTimeSeries_AgERA5_-33.8_25.3.csv'
    }),
    8: ("Oudtshoorn", "Oudtshoorn_combined_cleaned.xlsx", {
        (22.25, -33.63333511): 'ClimateTimeSeries_AgERA5_-33.6_22.3.csv',
        (22.257684, -33.630323): 'ClimateTimeSeries_AgERA5_-33.6_22.3.csv'
    })
}

# Display menu and get user input
print("Select a station:")
for key, value in station_options.items():
    print(f"{key}: {value[0]}")
    
selected_option = int(input("\nEnter the number corresponding to your choice: "))

if selected_option not in station_options:
    print("Invalid selection. Exiting.")
    exit()

# Get the selected station details
station_name, station_file, stn_cdb_mapping = station_options[selected_option]

# Initialize an empty DataFrame to store merged results
merged_results = []

# Process each mapping for the selected station
for (longitude, latitude), cdb_file in stn_cdb_mapping.items():
    # Paths for station and climate database files
    file_path_stn = os.path.join(stn_dir, station_file).replace("\\", "/")
    file_path_cdb = os.path.join(cdb_dir, cdb_file).replace("\\", "/")
    
    # Load station data
    df_stn = pd.read_excel(file_path_stn, parse_dates=['Date'])

    # Filter station data based on coordinates (optional, if data has mixed locations in a single file)
    df_stn = df_stn[(df_stn['Longitude'] == longitude) & (df_stn['Latitude'] == latitude)]

    # Rename station data columns
    stn_rename_mapping = {
        'Tmax': 'stn_Tmax',
        'Tmin': 'stn_Tmin',
        'RHmax': 'stn_RHmax',
        'RHmin': 'stn_RHmin',
        'WSmean': 'stn_WSmean',
        'SR': 'stn_SR',
        'Pr': 'stn_Pr',
        'ETo': 'stn_ETo'
    }
    df_stn.rename(columns=stn_rename_mapping, inplace=True)

    # Load climate database data
    df_cdb = pd.read_csv(file_path_cdb, parse_dates=['Date'])

    # Rename climate database columns
    cdb_rename_mapping = {
        'Tmax': 'cdb_Tmax',
        'Tmin': 'cdb_Tmin',
        'RHmax': 'cdb_RHmax',
        'RHmin': 'cdb_RHmin',
        'WSmean': 'cdb_WSmean',
        'SR': 'cdb_SR',
        'Pr': 'cdb_Pr',
        'ETo': 'cdb_ETo'
    }
    df_cdb.rename(columns=cdb_rename_mapping, inplace=True)

    # Merge station data with climate database data on 'Date'
    df_merged = pd.merge(df_stn, df_cdb, on='Date', how='left')

    # Format 'Date' as YYYYMMDD
    df_merged['Date'] = df_merged['Date'].dt.strftime('%Y%m%d')

    # Add results to the list
    merged_results.append(df_merged)

# Concatenate all merged results into a single DataFrame
final_merged = pd.concat(merged_results, ignore_index=True)

# Save the final merged DataFrame to a CSV file
final_output_file = f"Merged_{station_name.replace(' ', '')}_stn_cdb.csv"
final_output_path = os.path.join(output_dir, final_output_file).replace("\\", "/")
final_merged.to_csv(final_output_path, index=False, na_rep='NaN')  # Explicitly write 'NaN' for missing values

print(f"Merged dataset saved to: {final_output_path}")

