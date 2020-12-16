#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re


def configure_pipeline(template, output, checkpoint, data_dir, classes, batch_size) 
    # Generate label_map.pbtxt
    label_map_entries = [
        'item {\n'
            f'\tname: "{c}",\n'
            f'\tid: {i+1}\n'
        '}'#.format(c, i)
        for i, c in enumerate(classes)
    ]

    label_map = '\n'.join(label_map_entries)
    num_classes = len(classes)
    print(label_map)

    with open('data/label_map.pbtxt', 'w') as f:
        f.write(label_map)

    pipeline_params = {
        'NUM_CLASSES': str(num_classes),
        'LABEL_MAP': '{}/label_map.pbtxt'.format(data_dir),
        'BATCH_SIZE': str(batch_size),
        'CHECKPOINT': checkpoint,
        'TRAIN_TFRECORD': '{}/train.record'.format(data_dir),
        'EVAL_TFRECORD': '{}/test.record'.format(data_dir),
    }

    with open(template, 'r') as f:
        lines = f.readlines()

    with open(output, 'w') as output:
        for l in lines:
            match = [k for k in pipeline_params.keys() if k in l]
            if len(match) > 0:
                l = l.replace(match[0], pipeline_params[match[0]])

            output.write(l)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--template", required=True, default=None, type=str, help="Path to config's template")
    args.add_argument("-o", "--output", default="pipeline.config", type=str, help="Where to save the generated config file")
    args.add_argument("-k", "--checkpoint", required=True, default=None, type=str, help="Path to model's checkpoint")
    args.add_argument("-b", "--batch-size", default=32, type=int, help="Training batch size")
    args.add_argument("-c", "--classes", default=None, type=str, nargs='+', help="Classes to detect")
    args.add_argument("-d", "--data-dir", default="data", type=str, help="Data directory with csv, tfrecord, and pbtxtfiles")

    config = args.parse_args()

    configure_pipeline(
        template=config.template,
        output=config.output,
        checkpoint=config.checkpoint,
        data_dir=config.data_dir,
        classes=config.classes,
        batch_size=config.batch_size
    )


if __name__ == '__main__':
    main()
