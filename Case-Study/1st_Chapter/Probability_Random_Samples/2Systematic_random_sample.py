import openpyxl
import random
workbook = openpyxl.load_workbook(r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx")
sheet = workbook.active
data = []
for row in sheet.iter_rows(values_only=True):
    data.append(row)
header = data.pop(0)
num_samples = 5
total_elements = len(data)
interval = total_elements // num_samples
if interval < 1:
    interval = 1
start = random.randint(0, interval - 1)
samples = [data[i] for i in range(start, total_elements, interval)][:num_samples]
# Print the samples
print("Header:", header)
print("Samples:")
for sample in samples:
    print(sample)
