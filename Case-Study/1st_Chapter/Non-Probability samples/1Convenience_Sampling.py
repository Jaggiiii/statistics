import openpyxl
import random
workbook = openpyxl.load_workbook(r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx")
sheet = workbook.active
data = []
for row in sheet.iter_rows(values_only=True):
    data.append(row)
header = data.pop(0)
n = 10 

convenience_samples = data[:n] #selecting first 10 members
print("Header:", header)
print(f"Convenience Samples (first {n} rows from data):")
for sample in convenience_samples:
    print(sample)
