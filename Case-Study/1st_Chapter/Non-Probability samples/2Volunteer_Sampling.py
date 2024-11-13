import openpyxl

workbook = openpyxl.load_workbook(r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx")
sheet = workbook.active
data = []
for row in sheet.iter_rows(values_only=True):
    data.append(row)
header = data.pop(0)
co2_avg = 35

volunteer = []
for row in data:
    try:
        if isinstance(row[4], (int, float)) and row[4] > co2_avg:
            volunteer.append(row)
    except (TypeError, ValueError):
        continue

print("Header:", header)
print(f"Volunteer Samples (Batting avg > {co2_avg}):")
for sample in volunteer:
    print(sample)
