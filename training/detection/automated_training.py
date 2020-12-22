#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import configparser
import os
from pathlib import Path
import shlex  # split str commandfor sp
import shutil
import subprocess as sp
import sys

import pandas as pd

sys.path.append('../../')
from dnn.utils import detect_video, annotate_video, \
                    generate_label_map, generate_detection_files, \
                    configure_pipeline, read_pickle
from dnn.tfrecord import generate_tfrecord
from dnn.tftrain import train_loop, export_trained_model


def automated_training_loop(config):
    # Global vars
    paths = config['paths']
    datasets_dir = paths.get('dataset_dir', 'datasets')
    train_dir = paths.get('train_dir', 'training')
    saved_models_dir = paths.get('saved_models_dir', 'trained_models')
    gt_dir = paths.get('gt_dir', '../../ground_truths/bcn/ref')
    detection_files_path = paths.get('detetion_files_path', 'tmp')
    pascalvoc = paths.get('pascalvoc', '../../accuracy-metrics/pascalvoc.py')

    train_dataset_path = paths.get('train_dataset_path', 'videos/train')
    test_dataset_path = [
        '{}/0'.format(paths.get('test_dataset_path', 'videos/test')),
        '{}/1'.format(paths.get('test_dataset_path', 'videos/test'))
    ]

    annotation = config['annotation']
    classes = annotation.get('classes', 'car,person,traffic light,motorcycle,truck,bus').split(',')
    groundtruth_threshold = annotation.getfloat('groundtruth_threshold', 0.5)

    train_config = config['train']
    template_config = train_config.get('template_config', 'pipeline_template.config')
    num_train_steps = train_config.getint('num_train_steps', 5000)
    batch_size = train_config.getint('batch_size', 32)
    
    # Make sure all directories exist and create those that don't contain required data
    if not os.path.exists(gt_dir):
        print('Directory with groundtruth files ({}) does not exist but is required.'.format(gt_dir))
        sys.exit(1)

    if not os.path.exists(train_dataset_path):
        print('Directory with training videos ({}) does not exist but is required.'.format(train_dataset_path))
        sys.exit(1)
        if len(os.listdir(train_dataset_path)) == 0:
            print('Directory with training videos ({}) is empty.'.format(train_dataset_path))
            sys.exit(1)

    for test_dir in test_dataset_path:
        if not os.path.exists(test_dir):
            print('Directory with test videos ({}) does not exist but is required.'.format(test_dir))
            sys.exit(1)
        
    if len(os.listdir(test_dataset_path[0])) == 0:
        print('Directory with test videos ({}) is empty.'.format(test_dataset_path[0]))
        sys.exit(1)

    if not os.path.exists(template_config):
        print('Template pipeline config file ({}) does not exist but is required.'.format(template_config))
        sys.exit(1)

    if not os.path.exists(datasets_dir):
        print('datasets_dir ({}) does not exist. Creating...'.format(datasets_dir))
        os.makedirs(datasets_dir)

    if not os.path.exists(train_dir):
        print('train_dir ({}) does not exist. Creating...'.format(train_dir))
        os.makedirs(train_dir)

    if not os.path.exists(detection_files_path):
        print('Directory to store temporary detection files ({}) does not exist. Creating...'.format(detection_files_path))
        os.makedirs(detection_files_path)

    # PRE A: Set train and test datasets:
        # i. one whole day never used for training
        # ii. all videos used in previous checkpoints. Will be moved from train_dataset
        # iii. same video used for current training iteration
    train_dataset = [v for v in Path(train_dataset_path).glob('*.mkv')]
    test_dataset = [
        [v for v in Path(test_dataset_path[0]).glob('*.mkv')],
        [],  # Videos previously used for training. Empty at first
    ]
    all_videos_dataset = train_dataset + test_dataset[0]


    # PRE B: Generate label_map with valid classes used to annotate/train 
    label_map = generate_label_map(classes)


    # PRE C: Generate detection files for all training and test videos using groundtruths
    # Before start to generate files, check that every train/test video has its own groundtruth
    gt_missing = []
    for test_video in all_videos_dataset:
        test_video_id = test_video.stem
        pickle_path = '{}/{}.pkl'.format(gt_dir, test_video_id)
        if not os.path.exists(pickle_path):
            gt_missing.append(pickle_path)

    if len(gt_missing) > 0:
        print('Some groundtruth files are missing:')
        for gt in gt_missing:
            print('\t{}'.format(gt))
        sys.exit(1)

    for test_video in all_videos_dataset:
        test_video_id = test_video.stem
        pickle_path = '{}/{}.pkl'.format(gt_dir, test_video_id)
        output_dir = '{}/{}/groundtruths'.format(detection_files_path, test_video_id)
        if os.path.exists('{}/{}_0.txt'.format(output_dir, test_video_id)):
            continue
        os.makedirs(output_dir, exist_ok=True)

        detections = read_pickle(pickle_path) 
        generate_detection_files(
            detections=detections,
            output_dir=output_dir,
            prefix=test_video_id,
            label_map=None,  # with label_map=None, mscoco is used
            groundtruth=True,
            threshold=groundtruth_threshold
        )


    # For each video:
    for i, train_video in enumerate(train_dataset):
        print('Processing training video {}/{}'.format(i, len(train_dataset)))
        
        # Experiment vars
        train_video_id = train_video.stem
        data_dir = '{}/{}/data'.format(datasets_dir, train_video_id)
        images_dir = '{}/{}/images'.format(datasets_dir, train_video_id)
        train_model_dir = '{}/{}'.format(train_dir, train_video_id)
        exported_model_dir = '{}/{}'.format(saved_models_dir, train_video_id)

        # TODO: Check if folder exists or, at least, is empty
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        # 1. Annotate video.
        annotate_video(
            filename=str(train_video),
            groundtruth='{}/{}.pkl'.format(gt_dir, train_video_id),
            images_dir=images_dir,
            data_dir=data_dir,
            valid_classes=classes,
            label_map=None,  # groundtruths are annotated with mscoco label_map
            val_ratio=0.2,
            reshape=[320],
            min_score=0.,
            max_boxes=10,
            img_format='jpg'
        )
        

        # 2. Generate train.record and test.record using csv with annotations
        for dataset in ['train', 'test']:
            generate_tfrecord(
                '{}/{}.record'.format(data_dir, dataset),
                '{}/{}'.format(images_dir, dataset),
                '{}/{}_label.csv'.format(data_dir, dataset),
                label_map
            ) 

        # 3. Configure Pipeline
        # i. Find latest checkpoint in the training folder
        latest_checkpoint = str(sorted(Path(train_dir).glob('**/ckpt-*.index'),
                                       key=os.path.getmtime,
                                       reverse=True)[0]).replace('.index', '')

        pipeline_config_file = '{}/{}/pipeline.config'.format(train_dir, train_video_id)

        # ii. Create training subfolder
        os.makedirs(train_model_dir, exist_ok=True)

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
            model_dir=train_model_dir,
            num_train_steps=num_train_steps
        )

        # 5. Export trained model
        export_trained_model(
            pipeline_config_path=pipeline_config_file,
            trained_checkpoint_dir=train_model_dir,
            output_dir=exported_model_dir
        )

        # 6. Test new model against all eval dataset.
        test_accuracy = None  # DataFrame to store accuracies of all three test datasets
        for test_ds_id, test_ds in enumerate([test_dataset[0], test_dataset[1], [train_video]]):
            if len(test_ds) == 0:
                continue
        
            # i. Run detections using new model on all videos of the test_ds
            # FIXME: Should the test input data be resized?
            dataset_accuracy = {
                c: [] for c in classes
            }
            dataset_accuracy['test_video'] = []
            dataset_accuracy['mAP'] = []
            for test_video in test_ds:
                test_results = detect_video(
                    filename=str(test_video),
                    detector=detector,
                    min_score=min_score,
                    max_boxes=max_boxes,
                    label_map=label_map,
                    save_frames_to=None,
                    resize=None
                )

                # ii. Compute mAP for the tested video using pascalvoc.py
                # TODO: Integrate accuracy-metrics as a library

                # ii.1 Generate detection files from detections of the newly processed video
                test_video_id = test_video.stem
                output_dir = '{}/{}/detections_{}'.format(detection_files_path, train_video_id, test_video_id)
                os.makedirs(output_dir, exist_ok=True)

                # Save test_results detections to pickle for debugging/post-processing purposes
                to_pickle(test_results, '{}/{}.pkl'.format(output_dir, test_video_id))

                generate_detection_files(
                    detections=detections,
                    output_dir=output_dir,
                    prefix=test_video_id,
                    label_map=label_map,
                    groundtruth=False,
                    threshold=.0  # FIXME: Should we set a threshold or use every single detection
                )

                gt_detections_path = '{}/{}/groundtruths'.format(detection_files_path, test_video_id)

                cmdline_str = f'python {pascalvoc} --det {output_dir} -detformay xyrb' +\
                              f'-gt {gt_detections_path} -gtformat xyrb -np' +\
                              f'--classes {",".join(classes)} -sp {output_dir}/results'
                cmdline = shlex.split(cmdline_str)
                proc = sp.Popen(cmdline, stdout=sp.PIPE, stderr=sp.PIPE)
                out, err = proc.communicate()

                # iii. Get AP & mAP metrics from results.txt
                with open('{}/results/results.txt'.format(output_dir), 'r') as f:
                    lines = f.readlines()

                ap_class = None
                dataset_accuracy['test_video'].append(test_video_id)
                for l in lines:
                    if ap_class is not None:
                        ap = float(l.split(' ')[1].replace('%', ''))
                        dataset_accuracy[ap_class].append(ap)
                        ap_class = None

                    if 'Class: ' in l:
                        ap_class = l.split(' ')[1]
                    elif 'mAP' in l:
                        mAP = l.split(' ')[1].replace('%', '')
                        dataset_accuracy['mAP'].append(mAP)
                
            # iv. Save csv with dataset accuracies and add averages to the global
            dataset_accuracy = pd.DataFrame.from_dict(dataset_accuracy)
            dataset_accuracy.to_csv('{}/dataset_{}_accuracy.csv'.format(exported_model_dir, test_ds_id), ignore_index=True)

            ds_mean = dataset_accuracy.mean() 
            ds_mean['test_dataset'] = test_ds_id
            if test_accuracy is None:
                test_accuracy = pd.DataFrame([], columns=ds_mean.keys())
            test_accuracy = test_accuracy.append(ds_mean, ignore_index=True)

        test_accuracy.to_csv('{}/test_accuracy.csv'.format(exported_model_dir), ignore_index=True)

        # 7. Move the training video to test_dataset[1]
        # i. Move video file
        shutil.move(train_video, test_dataset_path[1])
        # ii. Add video to the test_dataset list
        test_dataset[1].append(train_video)

def main():
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", required=True, help="File containing training config")
    args = args.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    automated_training_loop(config)

if __name__ == '__main__':
    main()
