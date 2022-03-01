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

with open('./transmil_val_id.txt') as f:
	val_names = f.readlines()

val_names = [val_name[:-1] for val_name in val_names]

all_cases = [['case_id', 'slide_id', 'label']]
splits_0_bool = [['', 'train', 'val', 'test']]
splits_0_descriptor = [['', 'train', 'val', 'test']]
splits_0 = [['', 'train', 'val', 'test']]

patient_counter = 0
normal_counter = [0, 0, 0]
tumor_counter = [0, 0, 0]

for name in normal_names:
	all_cases.append(['parient_'+str(patient_counter), name[:-3], 'normal'])
	if name not in val_names:
		splits_0_bool.append([name[:-3], 'TRUE', 'FALSE', 'FALSE'])
		normal_counter[0] += 1
	else:
		splits_0_bool.append([name[:-3], 'FALSE', 'TRUE', 'FALSE'])
		normal_counter[1] += 1
	patient_counter += 1

for name in tumor_names:
	all_cases.append(['parient_'+str(patient_counter), name[:-3], 'tumor'])
	if name not in val_names:
		splits_0_bool.append([name[:-3], 'TRUE', 'FALSE', 'FALSE'])
		tumor_counter[0] += 1
	else:
		splits_0_bool.append([name[:-3], 'FALSE', 'TRUE', 'FALSE'])
		tumor_counter[1] += 1

	patient_counter += 1

def get_label_by_test_id(id):
	for elem in te_info:
		if elem[0] == id:
			return elem[1]

for name in te_names:
	label = get_label_by_test_id(name[:-3])
	if label == 'normal':
		normal_counter[2] += 1
	elif label == 'tumor':
		tumor_counter[2] += 1

	all_cases.append(['parient_'+str(patient_counter), name[:-3], label])

	splits_0_bool.append([name[:-3], 'FALSE', 'FALSE', 'TRUE'])

	patient_counter += 1

splits_0_descriptor.append(normal_counter)
splits_0_descriptor.append(tumor_counter)

print(all_cases)
print(splits_0_bool)
print(splits_0_descriptor)
