import os

post_fix = 'h5'

training_names = os.listdir('/media/hdd/CAMELYON16/feats_resnet50/training/' + post_fix + '_files')

rows = []
# rows = [['case_id', 'slide_id', 'label']]

count = 0
for name in training_names:
	type_, post = name.split('_')
	num, _ = post.split('.')

	num_int = int(num) - 1 if type_ == 'normal' else int(num)+159
	print(type_, 'slide_' + str(num_int) + '.' + post_fix)

	rows.append(['patient_' + str(count), 'slide_' + str(num_int), type_, num_int])
	count += 1

# def takeSecond(elem):
    # return elem[3]

rows.sort(key=lambda x: x[3])

print(rows)