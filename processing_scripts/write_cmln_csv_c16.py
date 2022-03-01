import os
import csv

normal_names = os.listdir('/data1/camelyon_data/CAMELYON16/training_patch/normal/patches')
tumor_names = os.listdir('/data1/camelyon_data/CAMELYON16/training_patch/tumor/patches')
te_names = os.listdir('/data1/camelyon_data/CAMELYON16/testing_patch/images/patches')

te_info = []
with open('/data1/camelyon_data/CAMELYON16/testing/reference.csv') as f:
	reader = csv.reader(f)

	for row in reader:
		te_info.append(row)

print(normal_names)
print(tumor_names)
print(te_names)
print(te_info)