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

# Set thresholds for initial and snowball sampling
initial_threshold = 20  # Starting threshold for snowball sampling
snowball_threshold = 33  # Maximum difference allowed to include in sampling

# Initialize snowball samples list
snowball_samples = []

# Find the first sample that meets the initial threshold
initial_sample = [row for row in data if isinstance(row[3], (int, float)) and row[3] > initial_threshold]

# Start snowball sampling if an initial sample is found
if initial_sample:
    # Select the first initial sample and add it to snowball samples
    current_sample = initial_sample[0]
    snowball_samples.append(current_sample)

    # Perform snowball sampling based on threshold criteria
    while current_sample:
        next_sample = None
        for row in data:
            if (
                isinstance(row[3], (int, float)) and
                abs(row[3] - current_sample[3]) <= snowball_threshold and
                row[3] > initial_threshold and
                row not in snowball_samples
            ):
                next_sample = row
                break

        if next_sample:
            snowball_samples.append(next_sample)
            current_sample = next_sample
        else:
            # Stop if no next sample is found within the threshold
            break

# Output results
print("Header:", header)
print(f"Snowball Samples starting from Temperature > {initial_threshold}:")
for sample in snowball_samples:
    print(sample)
