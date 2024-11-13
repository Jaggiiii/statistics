import pandas as pd
import random
import numpy as np

file_path = r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx"
data = pd.read_excel(file_path)

# Extract the 'Temperature' column
column_data = data['Temperature'].dropna().tolist()  
random_sample = random.sample(column_data, 20)

print(f'Length of sample: {len(random_sample)}')

# Sort the sample and find the median value
median_value = sorted(random_sample)[len(random_sample) // 2]
x = []
a_count = 0
b_count = 0

# Classify each value as 'A' (less than median), 'B' (greater than median), or '0' (equal to median)
for elem in random_sample:
    if elem < median_value:
        x.append("A")
        a_count += 1
    elif elem > median_value:
        x.append("B")
        b_count += 1
    else:
        x.append("0")  

# Count the number of runs in the sequence
runs = 0
for i in range(1, len(x)):
    if x[i] != x[i - 1] and x[i] != '0':
        runs += 1
runs += 1

# Output the results
print("Selected Sample:", random_sample)
print("Median Value:", median_value)
print(f"A/B Array: {x}")
print(f"Count of 'A': {a_count}")
print(f"Count of 'B': {b_count}")
print("Total Runs:", runs)

# Critical value range for the runs test
critical_value_1 = 1
critical_value_2 = 9
print(f"Critical Range: {critical_value_1}-{critical_value_2}")

# Hypothesis test
if critical_value_1 <= runs <= critical_value_2:
    print("Fail to reject the null hypothesis. The sequence appears to be random.")
else:
    print("Reject the null hypothesis. The sequence does not appear to be random.")
