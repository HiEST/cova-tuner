#!/bin/env python3

import os
import re
import sys
sys.path.append('../../../')
from dnn.tftrain import eval_continuously

dataset = sys.argv[1]
eval_ds = sys.argv[2]

os.makedirs(f'trained_models/{dataset}', exist_ok=True)
os.makedirs(f'trained_models/{dataset}/eval_{eval_ds}', exist_ok=True)
print(f'Evaluating {dataset} on {eval_ds}')

pipeline_config = f'trained_models/{dataset}/eval_{eval_ds}/pipeline.config'
if not os.path.isfile(pipeline_config):
    template = 'pipeline.config'
    lines = [re.sub('TEST_DATASET', f'eval_{eval_ds}', l) for l in open(template, 'r').readlines()]
    with open(pipeline_config, 'w') as f:
        for l in lines:
            f.write(l)

eval_continuously(
        pipeline_config_path=pipeline_config,
        model_dir=f'trained_models/{dataset}/eval_{eval_ds}',
        checkpoint_dir=f'trained_models/{dataset}',
        num_train_steps=2000,
        wait_interval=10,
        eval_timeout=120
)
