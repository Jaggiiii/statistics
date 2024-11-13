import pandas as pd
import numpy as np
from scipy.stats import norm
import re  

file_path = '/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx'
df = pd.read_excel(file_path)

# Apply the lambda function to clean and convert relevant columns (in this case 'Temperature')
# We will assume you want to calculate the sample size for the 'Temperature' column
# If other columns are needed, they can be processed similarly
df['Temperature'] = df['Temperature'].apply(lambda x: float(re.sub(r'[^\d.]', '', str(x))) if pd.notnull(x) else np.nan)

# Drop any rows with missing values in the 'Temperature' column
df.dropna(subset=['Temperature'], inplace=True)

# Now the 'Temperature' column contains the data you want to analyze
observed_frequencies = df['Temperature'].to_numpy()

# Standard deviation (S) for the sample
S = np.std(observed_frequencies, ddof=1)

# Set alpha for the confidence level
alpha = 0.05

# Z value for a two-tailed test (1 - alpha / 2)
Z = norm.ppf(1 - alpha / 2)

# Desired margin of error (d)
d = 1.0

# Sample size calculation using the formula
n = (Z**2 * S**2) / (d**2)

# Display the results
print(f"\nCalculated Sample Size for Estimating a Single Mean:")
print(f"| {'Sample Size (n)':<20} | {n:<10.4f} |")
print(f"| {'Z Tabulated (Z)':<20} | {Z:<10.4f} |")
print(f"| {'Standard Deviation (S)':<20} | {S:<10.4f} |")
print(f"| {'Margin of Error (d)':<20} | {d:<10.4f} |")
print("+-------------------------+------------+")
