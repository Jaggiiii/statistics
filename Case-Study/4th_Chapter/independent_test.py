import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Load the data from the Excel file
file_path = '/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx'
df = pd.read_excel(file_path)
df = df.head(50)
# Select the relevant columns (Temperature and another variable for analysis)
df = df[['Temperature', 'CO2 Emissions', 'Sea Level Rise', 'Precipitation', 'Humidity', 'Wind Speed']]

# Display Data Summary
print("Data Summary:")
print(df.head())

# Display basic statistics for the numerical columns
print("\nStatistics for Numerical Columns:")
print(df.describe())

# For Chi-Square, let's create a contingency table from two numeric columns.
# We'll convert continuous values into categories (e.g., low, medium, high).
# Here, we'll bin the temperature and CO2 emissions into categories.

df['Temp_Category'] = pd.cut(df['Temperature'], bins=[-np.inf, 10, 20, 30, np.inf], labels=['Low', 'Medium', 'High', 'Very High'])
df['CO2_Category'] = pd.cut(df['CO2 Emissions'], bins=[-np.inf, 380, 420, 460, np.inf], labels=['Low', 'Medium', 'High', 'Very High'])

# Create a contingency table for the categorical data
contingency_table = pd.crosstab(df['Temp_Category'], df['CO2_Category'])

# Calculate the Chi-Square statistic manually
row_totals = contingency_table.sum(axis=1)
column_totals = contingency_table.sum(axis=0)
grand_total = contingency_table.values.sum()
expected_frequencies = np.outer(row_totals, column_totals) / grand_total
observed_frequencies = contingency_table.values
chi_square_statistic = ((observed_frequencies - expected_frequencies) ** 2 / expected_frequencies).sum()

# Calculate degrees of freedom
dof = (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)

# Define significance level
alpha = 0.05

# Find the critical value from the Chi-Square distribution
chi_tab_value = chi2.ppf(1 - alpha, dof)
print(f"chi_tab_value:{chi_tab_value}")

# Print the results
print("\nChi-Square Test for Independent Results:")
print(f"+-------------------------+-----------------+-------------------------+")
print(f"| {'Chi-Square Statistic':<23} | {'Degrees of Freedom':<17} | {'Result':<25} |")
print(f"+-------------------------+-----------------+-------------------------+")
print(f"| {chi_square_statistic:<23.4f} | {dof:<17} | {'Reject H0' if chi_square_statistic > chi_tab_value else 'Accept H0':<25} |")
print(f"+-------------------------+-----------------+-------------------------+")

# Plot the Chi-Square distribution and the test statistic
x = np.linspace(0, chi_square_statistic + 10, 1000)
y = chi2.pdf(x, dof)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Chi-Square Distribution', color='blue')
plt.fill_between(x, 0, y, where=(x >= chi_square_statistic), color='red', alpha=0.5, label='Right Tail')
plt.axvline(chi_square_statistic, color='green', linestyle='--', label=f'Chi-Square Statistic ({chi_square_statistic:.2f})')
plt.axvline(chi_tab_value, color='red', linestyle='--', label=f'Chi-Tab Value ({chi_tab_value:.2f})')
plt.xlabel('Chi-Square Value')
plt.ylabel('Probability Density')
plt.title('Chi-Square Distribution for Test of Independence')
plt.legend()
plt.show()
