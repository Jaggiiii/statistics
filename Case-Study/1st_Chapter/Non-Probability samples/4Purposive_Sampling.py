import openpyxl

# Load the workbook and select the active sheet
workbook = openpyxl.load_workbook(r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx")
sheet = workbook.active

# Initialize data storage
data = []
for row in sheet.iter_rows(values_only=True):
    data.append(row)

# Extract header and data rows
header = data.pop(0)

# Define temperature threshold
temperature_threshold = 20

# Filter samples based on temperature threshold
purposive_samples = [row for row in data if isinstance(row[3], (int, float)) and row[3] > temperature_threshold]

# Output results
print("Header:", header)
print(f"Purposive Samples (Temperature > {temperature_threshold}):")
for sample in purposive_samples:
    print(sample)
