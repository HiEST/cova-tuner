#!/bin/env python3

import os
from pathlib import Path
import sys

import pandas as pd
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from tqdm import tqdm

path = sys.argv[1]
event_files = [f for f in Path(path).glob('*_lr*/**/events.out.tfevents*')]
# num_event_files = len([f for f in Path(path).glob('*_lr/**/events.out.tfevents*')])
# pbar = tqdm(total=num_event_files)

exps = [exp for exp in os.listdir(path) if '_lr' in exp]
 
all_df = pd.DataFrame([], columns=['experiment', 'step', 'metric', 'value'])
event_dirs = ['eval_day', 'eval_morning', 'eval_night', 'train']

for event_file in tqdm(event_files):
    _, exp, dataset, eval_train, *_ = event_file.parts
    ds_time, ds_size = dataset.split('_')

    metrics = []
    events = [ev for ev in summary_iterator(str(event_file))]
    for ev in events:
        if not hasattr(ev, 'wall_time'):
            print(f'{ev.summary.value[0].tag} in {str(event_file)} has no wall_time')
        if not hasattr(ev, 'step'):
            print(f'{ev.summary.value[0].tag} in {str(event_file)} has no step')
        wall_time = getattr(ev, 'wall_time', -1)
        step = getattr(ev, 'step', -1)
        for v in ev.summary.value:
            if 'eval_side_by_side' not in v.tag and 'image' not in v.tag:
                metrics.append([v.tag, tf.make_ndarray(v.tensor).item(), wall_time, step])

    # step = events[-1].step
    exp_df = pd.DataFrame(metrics, columns=['metric', 'value', 'wall_time', 'step'])
    # exp_df['step'] = step
    exp_df['experiment'] = exp
    exp_df['learning_rate'] = [lr for lr in exp.split('-') if '_lr' in lr][0].replace('_lr', '')
    exp_df['dataset'] = ds_time
    exp_df['dataset_size'] = ds_size
    exp_df['train_eval'] = eval_train

    all_df = all_df.append(exp_df, ignore_index=True)

all_df = all_df.sort_values(['experiment', 'learning_rate', 'dataset', 'dataset_size', 'train_eval', 'wall_time', 'step'])
all_df.to_csv('experiments_.csv', sep=',', index=False)

sys.exit(0)
for exp in exps:
    datasets = [ds for ds in os.listdir(f'{path}/{exp}')]
    for ds in datasets:
        ds_time, ds_size = ds.split('_')

        for event_dir in event_dirs:
            event_dir = f'{event_dir}/eval' if 'eval' in event_dir else event_dir
            event_files = [e for e in os.listdir(f'{path}/{exp}/{ds}/{event_dir}') if 'events.out.tfevents' in e]
            for f in event_files:
                metrics = []
                events = [ev for ev in summary_iterator(f'{path}/{exp}/{ds}/{event_dir}/{f}')]
                for ev in events:
                    for v in ev.summary.value:
                        if 'eval_side_by_side' not in v.tag and 'image' not in v.tag:
                            metrics.append([v.tag, tf.make_ndarray(v.tensor).item()])
                            # print(f'{v.tag}: {metrics[-1]}')
                        # else:
                        #     print(f'Skipping {v.tag}')
        #             else:
        #                 print(f'{v.tag} has length 0')

                step = events[-1].step
                exp_df = pd.DataFrame(metrics, columns=['metric', 'value'])
                exp_df['step'] = step
                exp_df['experiment'] = exp
                exp_df['dataset'] = ds_time
                exp_df['dataset_size'] = ds_size
                exp_df['train_eval'] = event_dir

                all_df = all_df.append(exp_df, ignore_index=True)

all_df = all_df.sort_values(['experiment', 'dataset', 'dataset_size', 'train_eval', 'step'])
all_df.to_csv('experiments.csv', sep=',', index=False)
