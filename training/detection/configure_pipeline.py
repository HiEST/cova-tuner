#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", required=True, default=None, type=str, help="Path to config's template")
    args.add_argument("-o", "--output", default="pipeline.config", type=str, help="Where to save the generated config file")
    args.add_argument("-k", "--checkpoint", required=True, default=None, type=str, help="Path to model's checkpoint")
    args.add_argument("-b", "--batch-size", default=64, type=int, help="Training batch size")
    args.add_argument("--classes", default=None, type=str, nargs='+', help="Classes to detect")
    args.add_argument("-t", "--tfrecord", default="data", type=str, help="Directory with tfrecord files")

    config = args.parse_args()

    # Generate label_map.pbtxt
    label_map_entries = [
        'item {\n'
            f'\tname: "{c}",\n'
            f'\tid: {i+1}\n'
        '}'#.format(c, i)
        for i, c in enumerate(config.classes)
    ]

    label_map = '\n'.join(label_map_entries)
    num_classes = len(config.classes)
    print(label_map)

    with open('data/label_map.pbtxt', 'w') as f:
        f.write(label_map)

    pipeline_params = {
        'NUM_CLASSES': str(num_classes),
        'LABEL_MAP': 'data/label_map.pbtxt',
        'BATCH_SIZE': str(config.batch_size),
        'CHECKPOINT': config.checkpoint,
        'TRAIN_TFRECORD': '{}/train.record'.format(config.tfrecord),
        'EVAL_TFRECORD': '{}/test.record'.format(config.tfrecord),
    }

    with open(config.config, 'r') as f:
        lines = f.readlines()

    with open(config.output, 'w') as output:
        for l in lines:
            match = [k for k in pipeline_params.keys() if k in l]
            if len(match) > 0:
                l = l.replace(match[0], pipeline_params[match[0]])

            output.write(l)


if __name__ == '__main__':
    main()
