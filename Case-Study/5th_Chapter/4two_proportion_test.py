import pandas as pd
import numpy as np
from scipy.stats import norm

# File path to the dataset
file_path = '/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx'

# Read the data from the Excel file
df = pd.read_excel(file_path)

# Define the column of interest, e.g., Temperature
data = df['Temperature']

# Calculate the proportions of data points within two temperature ranges
rate_between_0_25 = ((data >= 0) & (data <= 25)).sum()  # Temperature between 0 and 25
rate_between_25_45 = ((data > 25) & (data <= 45)).sum()  # Temperature between 25 and 45
total_data_points = len(data)

# Proportions for both ranges
p_1 = rate_between_0_25 / total_data_points
q_1 = 1 - p_1
p_2 = rate_between_25_45 / total_data_points
q_2 = 1 - p_2

# Given parameters for statistical calculation
alpha = 0.05  # Significance level
z_alpha = 1.960  # Z-score for 95% confidence level
beta = 0.80  # Power of the test
z_beta = 1.282  # Z-score for 80% power
error = p_1 - p_2  # Difference between the two proportions

# Sample size calculation using the formula
sample_size = 4 * ((z_alpha + z_beta) ** 2) * (((p_1 + p_2) / 2) * (1 - (p_1 + p_2) / 2)) / (error ** 2)

# Print results
print(f"=> Proportion of temperature between 0 and 25: {p_1: .2f}")
print(f"=> q_1 = 1 - p_1 = {q_1: .2f}")
print(f"\n=> Proportion of temperature between 25 and 45: {p_2: .2f}")
print(f"=> q_2 = 1 - p_2 = {q_2: .2f}")
print(f"=> Calculated sample size: {sample_size: .2f}")
print(f"=> Estimated sample size can be taken: {sample_size // 2}")
print(f"=> Size of each sample can be: {sample_size // 2}\n")
