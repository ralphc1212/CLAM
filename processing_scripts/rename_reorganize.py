import os

post_fix = 'h5'

training_names = os.listdir('/media/hdd/CAMELYON16/feats_resnet50/training/' + post_fix + '_files')

for name in training_names:
	type_, post = name.split('_')
	num, _ = post.split('.')

	if type_ == 'normal':
		print(type_, 'slide_' + num + '.' + post_fix)
	else:
		print(type_, 'slide_' + str(int(num)+160) + '.' + post_fix)

