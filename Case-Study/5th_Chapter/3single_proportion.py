import pandas as pd
from scipy.stats import norm

file_path = '/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx'
df = pd.read_excel(file_path)

# Define the data you want to analyze (e.g., Temperature)
data = df['Temperature']

# Calculate the proportion of values in a specified range (e.g., between 20 and 30 degrees)
proportion = ((data >= 20) & (data <= 30)).sum()
total = len(data)
p = proportion / total
q = 1 - p

# Set alpha for significance level and calculate the z-value for a two-tailed test
alpha = 0.05  
Z = norm.ppf(1 - alpha / 2)  

# Set the margin of error
d = 0.05

# Calculate the required sample size
n = ((Z ** 2) * p * q) / (d ** 2)

# Store the results in a dictionary
results = {
    "Sample Size (n)": round(n, 4),
    "Z Tabulated (Z)": round(Z, 4),
    "Estimated Proportion": round(p, 4),
    "Margin of Error (d)": round(d, 4)
}

# Print each key-value pair on a new line
for key, value in results.items():
    print(f"{key}: {value}")
