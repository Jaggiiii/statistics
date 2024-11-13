import pandas as pd
import numpy as np
from scipy.stats import norm
import re  

file_path = '/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx'
df = pd.read_excel(file_path)

# Assuming 'Temperature' is the column we want to analyze
df['Temperature'] = df['Temperature'].apply(lambda x: float(re.sub(r'[^\d.]', '', str(x))) if pd.notnull(x) else np.nan)
df.dropna(subset=['Temperature'], inplace=True)

# Creating two groups: one from even-indexed rows and one from odd-indexed rows
group1 = df['Temperature'].iloc[::2]  
group2 = df['Temperature'].iloc[1::2]

# Calculate the pooled standard deviation
sigma = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)  

# Given values
alpha = 0.05  
Z_alpha = norm.ppf(1 - alpha / 2)  # Two-tailed z-score for 95% confidence
Z_beta = 0.842  # Z score for 80% power
d = 5.0  # Margin of error (e.g., temperature difference)

def calculate_sample_size_two_means(sigma, Z_alpha, Z_beta, d):
    n = (4 * sigma**2 * (Z_alpha + Z_beta)**2) / (d**2)
    return n

# Calculate the required sample size
n = calculate_sample_size_two_means(sigma, Z_alpha, Z_beta, d)

# Output the results
print(f"\nCalculated Sample Size for Comparing Two Means:")
print(f"| {'Sample Size (n)':<20} | {n:<10.4f} |")
print(f"| {'Sigma (σ)':<20} | {sigma:<10.4f} |")
print(f"| {'Z Alpha (Zα)':<20} | {Z_alpha:<10.4f} |")
print(f"| {'Z Beta (Zβ)':<20} | {Z_beta:<10.4f} |")
print(f"| {'Margin of Error (d)':<20} | {d:<10.4f} |")
print("+-------------------------+------------+")
