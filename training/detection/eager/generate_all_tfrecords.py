import os
import sys

import pandas as pd

sys.path.append('../../../')
from dnn.tfrecord import generate_tfrecord
from dnn.utils import load_pbtxt

label_map = load_pbtxt('./label_map.pbtxt')
csv = pd.read_csv('all_detections_eval.csv')
csv['class'] = csv['class_id'].apply(lambda x: 'car' if x == 3 else 'person')
csv['time'] = csv.filename.apply(lambda x: x.split('/')[0])
print(csv['time'])

csv_morning = csv[csv['time'] == 'eval_morning']
csv_night = csv[csv['time'] == 'eval_night']
csv.to_csv('./eval_day.csv', sep=',', index=False)
csv_morning.to_csv('./eval_morning.csv', sep=',', index=False)
csv_night.to_csv('./eval_night.csv', sep=',', index=False)

generate_tfrecord(f'./eval_morning.record', './', f'eval_morning.csv', label_map)
generate_tfrecord(f'./eval_night.record', './', f'eval_night.csv', label_map)
generate_tfrecord(f'./eval_day.record', './', f'eval_day.csv', label_map)

sys.exit()
datasets = ['morning', 'night', 'day']
ds_sizes = [25, 50, 100]

# csv = pd.read_csv('all_detections.csv')
csv['class'] = csv['class_id'].apply(lambda x: 'car' if x == 3 else 'person')
csv['img_file'] = csv['filename'].apply(lambda x: x.split('/')[1])
csv.to_csv('./annotations.csv', sep=',', index=False)

subsets = {}
subsets_eval = {}
# for ds in datasets:
#     for imgs in ds_sizes:
#         subsets[f'{ds}-{imgs}'] = csv[csv['img_file'].isin(os.listdir(f'{ds}/{imgs}'))]
#         # subsets[f'{ds}-{imgs}'].to_csv(f'{ds}-{imgs}.csv', sep=',', index=False)
# 
#         subsets_eval[f'{ds}-{imgs}'] = csv[~csv['img_file'].isin(os.listdir(f'{ds}/{imgs}'))]
#         subsets_eval[f'{ds}-{imgs}'].to_csv(f'{ds}-{imgs}_eval.csv', sep=',', index=False)



# for ds in datasets:
#     for imgs in ds_sizes:
#         generate_tfrecord(f'{ds}/{imgs}/train.record', './', f'{ds}-{imgs}.csv', label_map)
#         generate_tfrecord(f'{ds}/{imgs}/test.record', './', f'{ds}-{imgs}_eval.csv', label_map)
        
# generate_tfrecord(f'./test.record', './', f'annotations.csv', label_map)
