import openpyxl
import random
workbook = openpyxl.load_workbook(r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx")
sheet = workbook.active
data = []
for row in sheet.iter_rows(values_only=True):
    data.append(row)
header = data.pop(0)
runs_column_index = 1
data.sort(key=lambda x: x[runs_column_index])  
elements_per_cluster = 15
clusters = [data[i:i + elements_per_cluster] for i in range(0, len(data), elements_per_cluster)]
# Print the number of clusters
print(f"Total number of clusters created: {len(clusters)}")
num_clusters_to_display = 1
selected_clusters = random.sample(clusters, min(num_clusters_to_display, len(clusters)))
print("Header:", header)
print("Data from randomly selected clusters:")
for idx, cluster in enumerate(selected_clusters):
    print(f"\nCluster {idx + 1}:")
    for sample in cluster:
        print(sample)
