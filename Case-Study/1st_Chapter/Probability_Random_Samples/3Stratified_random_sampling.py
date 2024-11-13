import openpyxl
import random
from collections import defaultdict
workbook = openpyxl.load_workbook(r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx")
sheet = workbook.active
data = []
for row in sheet.iter_rows(values_only=True):
    data.append(row)
header = data.pop(0)
n = 5  #number of elements for each strata
strata = defaultdict(list)
for row in data:
    average = row[4] #mdns -->co2
    if isinstance(average, (int, float)):  
        if average < 19:
            strata['<19'].append(row)
        elif 19 <= average < 30: 
            strata['19-29'].append(row)
        else:
            strata['>=30'].append(row)
stratified_samples = []
for stratum, rows in strata.items():
    selected_samples = random.sample(rows, min(n, len(rows)))  
    stratified_samples.extend(selected_samples)
print("Header:", header)
print("Stratified Samples:")
for sample in stratified_samples:
    print(sample)
