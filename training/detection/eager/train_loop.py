#!/bin/env python3

import os
import re
import sys

sys.path.append('../../../')
from dnn.tftrain import train_loop

dataset = sys.argv[1]

os.makedirs(f'trained_models/{dataset}', exist_ok=True)
print(f'Training on dataset {dataset}')

pipeline_config = f'trained_models/{dataset}/pipeline.config'
if not os.path.isfile(pipeline_config):
    template = f'pipeline.config'
    lines = [re.sub('TRAIN_DATASET', f'{dataset}', l) for l in open(template, 'r').readlines()]
    with open(pipeline_config, 'w') as f:
        for l in lines:
            f.write(l)

train_loop(
       pipeline_config=pipeline_config,
       model_dir=f'trained_models/{dataset}',
       num_train_steps=2000,
       checkpoint_every_n=200,
       record_summaries=True
)
