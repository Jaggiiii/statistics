import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Define file path
file_path = '/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx'

# Read the Excel file
df = pd.read_excel(file_path)

# Let's take the first 2 rows of the dataset for testing (you can modify this)
df = df.head(2)

# We are selecting the numerical columns like Temperature, CO2 Emissions, Sea Level Rise, etc.
df = df[['Temperature', 'CO2 Emissions', 'Sea Level Rise', 'Precipitation', 'Humidity', 'Wind Speed']] 

print("Data Summary:")
print(df.head())

print("\nStatistics for Numerical Columns:")
print(df.describe())

# Create a contingency table for numerical columns
contingency_table = df.select_dtypes(include=[np.number])

# Calculate row totals, column totals, and grand total
row_totals = contingency_table.sum(axis=1)
column_totals = contingency_table.sum(axis=0)
grand_total = contingency_table.values.sum()

# Calculate expected frequencies
expected_frequencies = np.outer(row_totals, column_totals) / grand_total

# Get observed frequencies (actual values)
observed_frequencies = contingency_table.values

# Calculate the chi-square statistic
chi_square_statistic = ((observed_frequencies - expected_frequencies) ** 2 / expected_frequencies).sum()

# Degrees of freedom (df) for the chi-square test
dof = (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)

# Set significance level
alpha = 0.05

# Get critical chi-square value (chi-tab value)
chi_tab_value = chi2.ppf(1 - alpha, dof)
print(f"chi_tab_value:{chi_tab_value}")

# Decision: Accept or Reject
if chi_square_statistic > chi_tab_value:
    result = "Reject the null hypothesis (H0): The observed frequencies are not homogeneous."
else:
    result = "Accept the null hypothesis (H0): The observed frequencies are homogeneous."

print("\nManual Calculation of Chi-Square Test for Homogeneity:")
print(f"Chi-Square Statistic: {chi_square_statistic:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"Chi-Tab Value (Critical Value at alpha = {alpha}): {chi_tab_value:.4f}")
print(f"Result: {result}")

# Plotting the Chi-Square Distribution and Critical Region
x = np.linspace(0, chi_square_statistic + 10, 1000)
y = chi2.pdf(x, dof)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Chi-Square Distribution', color='blue')
plt.fill_between(x, 0, y, where=(x >= chi_square_statistic), color='red', alpha=0.5, label='Right Tail')
plt.axvline(chi_square_statistic, color='green', linestyle='--', label=f'Chi-Square Statistic ({chi_square_statistic:.2f})')
plt.axvline(chi_tab_value, color='red', linestyle='--', label=f'Chi-Tab Value ({chi_tab_value:.2f})')

# Adding labels and title
plt.xlabel('Chi-Square Value')
plt.ylabel('Probability Density')
plt.title('Chi-Square Distribution for Test of Homogeneity')
plt.legend()
plt.show()
