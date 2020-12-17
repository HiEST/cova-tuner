#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import sys

sys.path.append('../../')
from dnn.utils import detect_video, annotate_video, generate_label_map
from dnn.tfrecord import generate_tfrecord
from dnn.tftrain import train_loop, export_trained_model
from training.detection.configure_pipeline import configure_pipeline
from dnn.utils import detect_video

# Global vars
datasets_dir = 'datasets'
train_dir = 'training'
saved_models_dir = 'trained_models'
gt_dir = ''

template_config = 'pipeline.config.temp'

classes = ['car', 'person', 'traffic light', 'motorcycle', 'truck', 'bus']
label_map = generate_label_map(valid_classes)

num_train_steps = 5000


# PRE A: Set EVAL dataset:
    # i. one whole day never used for training
    # ii. all videos used in previous checkpoints
    # iii. same video used for current training iteration

test_dataset_path = [
    'test/0',
    'test/1'
]

test_dataset = [
    [str(v) for v in Path(test_dataset_path[0]).glob('*.mkv')]
    [],  # Videos previously used for training. Empty at first
]

# PRE B: Get input videos. 
# There will be one training (num_steps=1000/5000) per video. 
# After training, it'll be moved to eval.


# For each video:
for video in train_dataset:
    
    # Experiment vars
    video_id = Path(video).stem
    data_dir = '{}/{}/data'.format(datasets_dir, video_id)
    images_dir = '{}/{}/images'.format(datasets_dir, video_id)
    model_dir = '{}/{}'.format(train_dir, video_id)
    output_model_dir = '{}/{}'.format(saved_models_dir, video_id)

    # TODO: Check if folder exists or, at least, is empty
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)


    # 1. Annotate video.
    annotate_video(
        filename=video,
        groundtruth='{}/{}.pkl'.format(gt_dir, video_id)
        images_dir=images_dir,
        data_dir=data_dir,
        val_ratio=0.2,
        reshape=(),
        min_score=0.,
        max_boxes=10,
        img_format='jpg'
    )
    

    # 2. Generate csv with train.{csv,record}, test.{csv,record}
    for dataset in ['train', 'test']:
        generate_tfrecord(
            '{}/{}.record'.format(data_dir, dataset),
            '{}/{}'.format(images_dir, dataset),
            '{}/{}.csv'.format(data_dir, dataset),
            label_map
        ) 


    # 3. Configure Pipeline
    # i. Find latest checkpoint in the training folder
    latest_checkpoint = str(sorted(Path(train_dir).glob('**/ckpt-*.index'),
                                   key=os.path.getmtime,
                                   reverse=True)[0]).replace('.index', '')

    pipeline_config_file = '{}/{}/pipeline.config'.format(train_dir, video_id)

    # ii. Create training subfolder
    os.makedirs(model_dir, exist_ok=True)

    # iii. Generate pipeline.config file base on template
    configure_pipeline(
        template=template_config,
        output=pipeline_config_file,
        checkpoint=latest_checkpoint,
        data_dir=data_dir,
        classes=classes,
        batch_size=batch_size
    )

    # 4. Launch training with latest checkpoint
    train_loop(
        pipeline_config=pipeline_config_file,
        model_dir=model_dir,
        num_train_steps=num_train_steps
    )

    # 5. Export trained model
    export_trained_model(
        pipeline_config_path=pipeline_config_file,
        trained_checkpoint_dir=model_dir,
        output_dir=saved_model_dir
    )


    # 6. Test new model against all eval dataset.
    for test_ds in [test_dataset[0], test_dataset[1], [video]]:
    
        # i. Run detections using new model on all videos of the test_ds
        detect_video(
            filename=video,
            detector=detector,
            min_score=min_score,
            max_boxes=max_boxes,
            label_map=label_map,
            save_frames_to=images_dir,
            resize=True
        )


