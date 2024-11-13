import openpyxl
import random
from collections import defaultdict
workbook = openpyxl.load_workbook(r"/home/jaggi/Downloads/Case-Study/Case-Study/data.xlsx")
sheet = workbook.active
data = []
for row in sheet.iter_rows(values_only=True):
    data.append(row)
header = data.pop(0)
num_clusters_to_sample = 20  # No.of clusters to sample in Phase 1
num_strata_to_sample = 10     # No.of strata to sample in Phase 2
num_individuals_per_stratum = 2  # No.of individuals to sample from each selected stratum in Phase 3
clusters = defaultdict(list)
for row in data:
    cluster_id = row[0]  
    clusters[cluster_id].append(row)
# Phase 1: Randomly select clusters
selected_clusters = random.sample(list(clusters.keys()), min(num_clusters_to_sample, len(clusters)))
final_samples = []

# Phase 2 & 3: For each selected cluster, perform stratified sampling
for cluster in selected_clusters:
    individuals = clusters[cluster]
    strata = defaultdict(list)
    for individual in individuals:
        stratum = individual[4]  
        strata[stratum].append(individual)

    # Phase 3: Randomly select strata from each cluster
    selected_strata = random.sample(list(strata.keys()), min(num_strata_to_sample, len(strata)))

    # Phase 4: Sample individuals within each selected stratum
    for stratum in selected_strata:
      
        sampled_individuals = random.sample(strata[stratum], min(num_individuals_per_stratum, len(strata[stratum])))
        final_samples.extend(sampled_individuals)
print("Header:", header)
print("Multiphase Random Samples:")
for sample in final_samples:
    print(sample)
