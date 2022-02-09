import os

normal_names = os.listdir('/media/hdd/CAMELYON16/training_patched/normal/patches')
tumor_names = os.listdir('/media/hdd/CAMELYON16/training_patched/tumor/patches')

normal_names = [name[:-3] for name in normal_names]
tumor_names = [name[:-3] for name in tumor_names]

print(normal_names)
print(tumor_names)