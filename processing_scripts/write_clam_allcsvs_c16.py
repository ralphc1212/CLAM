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

all_cases = [['case_id', 'slide_id', 'label']]
splits_0_bool = [['', 'train', 'val', 'test']]
splits_0_descriptor = [['', 'train', 'val', 'test']]
splits_0 = [['', 'train', 'val', 'test']]

patient_counter = 0
for name in normal_names:
	all_cases.append(['parient_'+str(patient_counter), name[:-3], 'normal'])
	patient_counter += 1

for name in tumor_names:
	all_cases.append(['parient_'+str(patient_counter), name[:-3], 'tumor'])
	patient_counter += 1


def get_label_by_test_id(id):
	for elem in te_info:
		if elemp[0] is id:
			return elem[1]
for name in te_names:
	all_cases.append(['parient_'+str(patient_counter), name[:-3], get_label_by_test_id(name[:-3])])
	patient_counter += 1

print(all_cases)