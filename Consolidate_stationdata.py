# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:30:40 2024
This script reads and performs quality control on weather station data.
It reads from multiple input files, performs quality control and writes out
in a consistent format: 
Lat, Lon, Alt, Date, Tmax, Tmin, RHmax, RHmin, WSmean, SR, Pr, ETo
@author: SteynAS
"""

import pandas as pd
import numpy as np
import os

# File paths
directory = r'C:/Users/matladit/StnData/QControlled'
file1_name = 'Mbombela_1985-2001.xlsx'
file2_name = 'Mbombela_2000-2024.xlsx'
output_file_name = 'Mbombela_2000-2024_combined_cleaned.xlsx'

file1_path = os.path.join(directory, file1_name)
file2_path = os.path.join(directory, file2_name)
output_path = os.path.join(directory, output_file_name)

# Columns to retain in the final output
columns_to_keep = [
    'Longitude', 'Latitude', 'Altitude', 'Date',
    'Tmax', 'Tmin', 'RHmax', 'RHmin',
    'WSmean', 'SR', 'Pr', 'ETo'
]

# Weather variables
weather_columns = ['Tmax', 'Tmin', 'RHmax', 'RHmin', 'WSmean', 'SR']
weather_columns_with_pr = weather_columns + ['Pr']

# Standardize missing values to NaN
missing_values = ['--', 9999, -9999, -999, 'NaN', 'nan']

# Function to clean duplicate columns
def clean_duplicate_columns(df):
    column_priority = {
        'Tmax': ['Tmax', 'Tmx', 'Tx'],  # Maximum temperature
        'Tmin': ['Tmin', 'Tmn', 'Tn'],  # Minimum temperature
        'RHmax': ['RHmax', 'Rhmax', 'RHmx', 'Rhx', 'RHx'],  # Max humidity
        'RHmin': ['RHmin', 'Rhmin', 'RHmn', 'Rhn', 'RHn'],  # Min humidity
        'Pr': ['Pr', 'Rain'],  # Precipitation
        'WSmean': ['WSmean', 'U2'],  # Wind speed
        'SR': ['SR', 'Rs', 'Rs est', 'Rs est.'],  # Solar radiation
        'ETo': ['ETo', 'ET0', 'PM ET0', 'PM ETo']  # Reference evapotranspiration
    }

    # Resolve duplicate column names
    df = df.loc[:, ~df.columns.duplicated(keep='last')]

    cleaned_df = pd.DataFrame()
    for key, aliases in column_priority.items():
        matching_cols = [col for col in df.columns if col in aliases]
        if matching_cols:
            cleaned_df[key] = df[matching_cols[0]]

    # Retain other necessary columns
    non_duplicate_cols = [col for col in df.columns if col not in sum(column_priority.values(), [])]
    cleaned_df = pd.concat([df[non_duplicate_cols], cleaned_df], axis=1)

    return cleaned_df

# Apply quality control filter
def apply_quality_control(df):
    # Define quality control thresholds
    qc_thresholds = {
        'Tmax': (-9.9, 50.9),
        'Tmin': (-20.9, 34.9),
        'RHmax': (0, 100),
        'RHmin': (0, 100),
        'WSmean': (0, 49.9),
        'SR': (0, 39.9),
        'Pr': (0, 597),
        'ETo': (0, 15)
    }

    # Apply threshold rules
    for column, (min_val, max_val) in qc_thresholds.items():
        if column in df.columns:
            df.loc[(df[column] < min_val) | (df[column] > max_val), column] = np.nan

    # Set Tmax, Tmin to NaN if Tmax > 30 and Tmin < 0
    if 'Tmax' in df.columns and 'Tmin' in df.columns:
        mask = (df['Tmax'] > 30) & (df['Tmin'] < 0)
        columns_to_nan = ['Tmax', 'Tmin', 'RHmax', 'RHmin']
        for col in columns_to_nan:
            if col in df.columns:
                df.loc[mask, col] = np.nan

    if 'Tmax' in df.columns and 'Tmin' in df.columns:
        # Set Tmax, Tmin to NaN when Tmax < Tmin
        mask_tmax_tmin = df['Tmax'] < df['Tmin']
        df.loc[mask_tmax_tmin, ['Tmax', 'Tmin']] = np.nan

        # Set Tmax, Tmin to NaN when Tmax = Tmin = 0
        mask_tmax_tmin_zero = (df['Tmax'] == 0) & (df['Tmin'] == 0)
        df.loc[mask_tmax_tmin_zero, ['Tmax', 'Tmin']] = np.nan

    if 'RHmax' in df.columns and 'RHmin' in df.columns:
        # Set RHmax, RHmin to NaN when RHmax < RHmin
        mask_rhmax_rhmin = df['RHmax'] < df['RHmin']
        df.loc[mask_rhmax_rhmin, ['RHmax', 'RHmin']] = np.nan
        # Set RHmax, RHmin to NaN when RHmax = RHmin = 0
        mask_rhmax_rhmin_zero = (df['RHmax'] == 0) & (df['RHmin'] == 0)
        df.loc[mask_rhmax_rhmin_zero, ['RHmax', 'RHmin']] = np.nan

    if 'ETo' in df.columns:
        # Set ETo to NaN if any of the weather variables are missing
        weather_vars_missing = df[weather_columns].isnull().any(axis=1)
        df.loc[weather_vars_missing, 'ETo'] = np.nan
        
    return df

# Function to process a single file
def process_file(file_path):
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()  # Strip spaces from column names
    except Exception as err:
        raise Exception(f"Error reading the file: {err}")

    # Ensure 'Year', 'Month', 'Day' columns are present
    if {'Year', 'Month', 'Day'}.issubset(df.columns):
        try:
            # Convert 'Year', 'Month', 'Day' to numeric, coercing errors to NaN
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
            df['Day'] = pd.to_numeric(df['Day'], errors='coerce')

            # Drop rows with invalid Year, Month, or Day values
            df.dropna(subset=['Year', 'Month', 'Day'], inplace=True)

            # Construct the 'Date' column in YYYYMMDD format
            df['Date'] = pd.to_datetime(
                df[['Year', 'Month', 'Day']].astype(int).astype(str).agg('-'.join, axis=1),
                errors='coerce'
            ).dt.strftime('%Y%m%d')  # Format as YYYYMMDD

            # Drop rows where 'Date' could not be constructed
            df.dropna(subset=['Date'], inplace=True)

        except Exception as err:
            raise Exception(f"Error processing 'Year', 'Month', 'Day' columns: {err}")
    else:
        raise KeyError("The input file must contain 'Year', 'Month', and 'Day' columns.")

    # Set option to disable silent downcasting warnings globally
    pd.set_option('future.no_silent_downcasting', True)

    # Replace missing values
    df = df.replace(missing_values, np.nan)

    # Clean duplicate columns
    df_cleaned = clean_duplicate_columns(df)

    # Apply quality control
    df_cleaned = apply_quality_control(df_cleaned)

    # Remove rows where all weather variables and Pr are missing
    df_cleaned.dropna(subset=weather_columns_with_pr, how='all', inplace=True)

    return df_cleaned

# Process both files
df1_cleaned = process_file(file1_path)
df2_cleaned = process_file(file2_path)

# Combine the two datasets and handle duplicate dates
combined_df = pd.concat([df1_cleaned, df2_cleaned], ignore_index=True)

# Resolve duplicate dates as per initial script
def resolve_duplicate_dates(df, key_column='Date', important_columns=columns_to_keep):
    grouped = df.groupby(key_column)
    resolved_rows = []

    for date, group in grouped:
        if len(group) == 1:
            resolved_rows.append(group.iloc[0])  # Single occurrence, no duplicates
        else:
            # Find the row with the least missing values
            most_complete_row = group.loc[group[important_columns].isnull().sum(axis=1).idxmin()]
            resolved_rows.append(most_complete_row)

    return pd.DataFrame(resolved_rows)

# Resolve duplicate dates in the combined dataset
final_df = resolve_duplicate_dates(combined_df)

# Ensure Longitude, Latitude, Altitude, and Date are non-missing
if final_df[['Longitude', 'Latitude', 'Altitude', 'Date']].isnull().any().any():
    raise ValueError("Longitude, Latitude, Altitude, or Date contains missing values.")

# Retain only the specified columns for the final output
try:
    final_df_output = final_df[columns_to_keep]
except KeyError as e:
    missing_columns = [col for col in columns_to_keep if col not in final_df.columns]
    raise KeyError(f"The following required columns are missing: {missing_columns}") from e

# Round specific columns to 2 decimal places
columns_to_round = ['WSmean', 'SR', 'ETo']
for col in columns_to_round:
    if col in final_df_output.columns:
        final_df_output.loc[:, col] = final_df_output[col].round(2)

# Save the final output file
try:
    final_df_output.to_excel(output_path, index=False, na_rep='NaN')  # Write missing as 'NaN'
    print(f"Combined data successfully cleaned and written to {output_file_name}")
except Exception as err:
    raise Exception(f"Error saving the file: {err}")

