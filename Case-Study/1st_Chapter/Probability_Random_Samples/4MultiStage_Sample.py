import openpyxl
import random
from collections import defaultdict
workbook = openpyxl.load_workbook(r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx")
sheet = workbook.active
data = []
for row in sheet.iter_rows(values_only=True):
    data.append(row)
header = data.pop(0)
num_clusters_to_sample = 20  # Stage 1
num_strata_to_sample = 10     # Stage 2

clusters = defaultdict(list)
for row in data:
    cluster_id = row[0]  
    clusters[cluster_id].append(row)
selected_clusters = random.sample(list(clusters.keys()), min(num_clusters_to_sample, len(clusters)))
final_samples = []
for cluster in selected_clusters:
    individuals = clusters[cluster]
    strata = defaultdict(list)
    for individual in individuals:
        stratum = individual[4]  
        strata[stratum].append(individual)
    selected_strata = random.sample(list(strata.keys()), min(num_strata_to_sample, len(strata)))
    for stratum in selected_strata:
        sampled_individuals = random.sample(strata[stratum], min(len(strata[stratum]), num_strata_to_sample))
        final_samples.extend(sampled_individuals)
print("Header:", header)
print("Multistage Random Samples:")
for sample in final_samples:
    print(sample)
