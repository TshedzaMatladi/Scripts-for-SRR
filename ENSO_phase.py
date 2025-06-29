
"""
This script applies ENSO classification by identifying all 3-month seasons that are part of any 5 consecutive overlapping 3-month periods
meeting the ONI threshold (≥ +0.5°C for El Niño, ≤ –0.5°C for La Niña). Each season is classified as EN or LN if it lies within any such window.
Classification relies solely on ONI values (rounded to 1 decimal to align with CPC visual summaries) and does not consider atmospheric coupling.
ONI data was downloaded from: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

Created April 2025
Updated: Dual-directional + inclusion logic + CPC-style rounding
@author: SteynAS@ufs.ac.za
"""

import pandas as pd

# Step 1: Load the ONI Data
file_path = 'C:/ENSO/ONI_cpc.xlsx'
oni_df = pd.read_excel(file_path, sheet_name='ONI')

# Step 2: Add a Classification Column (rounded to 1 decimal)
def classify_enso_inclusive(oni_series):
    oni_series = oni_series.round(1)  # Match CPC rounding logic
    classification = ['N'] * len(oni_series)
    for i in range(len(oni_series) - 4):
        window = oni_series[i:i+5]
        if all(window >= 0.5):
            for j in range(i, i+5):
                classification[j] = 'EN'
        elif all(window <= -0.5):
            for j in range(i, i+5):
                classification[j] = 'LN'
    return classification

# Step 3: Apply classification
df = oni_df.copy()
df['Classification'] = classify_enso_inclusive(df['ANOM'])

# Step 4: Save the classified DataFrame
output_path = 'C:/ENSO/ONI_classification_results.xlsx'
df.to_excel(output_path, index=False)
print(f"ENSO classification with inclusive 5-season window saved to: {output_path}")