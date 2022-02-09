import os

post_fix = 'h5'

training_names = os.listdir('/media/hdd/CAMELYON16/feats_resnet50/training/' + post_fix + '_files')

for name in training_names:
	print(name.split('_'))
	print(name.split('_')[1].split('.'))
	print('------')