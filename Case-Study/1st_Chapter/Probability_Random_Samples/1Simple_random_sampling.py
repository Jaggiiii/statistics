import openpyxl
import random
workbook = openpyxl.load_workbook(r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx")
s = workbook.active 
data = []
for row in s.iter_rows(values_only=True):
    data.append(row)
header = data.pop(0)
ns = 5 
samples = random.sample(data, ns)
print("Header:", header)
print("Samples:")
for sample in samples:
    print(sample)
