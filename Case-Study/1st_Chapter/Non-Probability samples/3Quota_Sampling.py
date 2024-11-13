import openpyxl
from collections import defaultdict

# Load the workbook and select the active sheet
workbook = openpyxl.load_workbook(r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx")
sheet = workbook.active

# Initialize data storage
data = []
for row in sheet.iter_rows(values_only=True):
    data.append(row)

# Extract header and data rows
header = data.pop(0)

# Define quota limits based on CO₂ Emissions
quota_limits = {
    'CO₂ Emissions > 400': 3, 
    '300 < CO₂ Emissions <= 400': 3,  
    '200 < CO₂ Emissions <= 300': 3  
}
quota_groups = defaultdict(list)

# Process rows based on CO₂ Emissions criteria
for row in data:
    co2_emissions = row[4]  # Assuming "CO₂ Emissions" is the 5th column
    if isinstance(co2_emissions, (int, float)):
        if co2_emissions > 400:
            quota_groups['CO₂ Emissions > 400'].append(row)
        elif 300 < co2_emissions <= 4000:
            quota_groups['300 < CO₂ Emissions <= 400'].append(row)
        elif 200 < co2_emissions <= 3000:
            quota_groups['200 < CO₂ Emissions <= 300'].append(row)

# Collect quota samples based on the defined limits
quota_samples = []
for quota_name, limit in quota_limits.items():
    quota_samples.extend(quota_groups[quota_name][:limit])

# Output results
print("Header:", header)
print("Quota Samples:")
for sample in quota_samples:
    print(sample)
