#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser
from copy import deepcopy
from glob import glob
import multiprocessing as mp
import os
from pathlib import Path
import shlex  # split str commandfor sp
import shutil
import subprocess as sp
import sys

import pandas as pd
from tqdm import tqdm, trange

sys.path.append('../../')
from dnn.utils import detect_video, annotate_video, load_pbtxt, \
                    generate_label_map, generate_detection_files, \
                    configure_pipeline, read_pickle, to_pickle
from dnn.tfrecord import generate_tfrecord_from_csv
from dnn.tftrain import train_loop, eval_continuously, \
                    export_trained_model, load_saved_model, \
                    set_gpu_config


def eval_detector(args):
    test_video = str(args['test_video'])
    test_video_id = args['test_video'].stem
    pascalvoc = args['pascalvoc']
    root_dir = args['root_dir']
    detection_files_path = args['detection_files_path']
    groundtruth_files_path = args['groundtruth_files_path']
    output_dir = args['output_dir']

    detector = args['detector']
    min_score = args['min_score']
    max_boxes = args['max_boxes']
    label_map = args['label_map'] 
    classes = args['classes']

    test_results = None
    pkl_file = '{}/{}.pkl'.format(output_dir, test_video_id)
    if os.path.exists(pkl_file):
        test_results = read_pickle(pkl_file)
        if len(test_results) == 0:
            print(f'Recovered {pkl_file} but it has no detections. Trying again.')
            os.remove(pkl_file)

    if not os.path.exists(pkl_file):
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
        os.makedirs(output_dir, exist_ok=True)

        # Save test_results detections to pickle for debugging/post-processing purposes
        to_pickle(test_results, '{}/{}.pkl'.format(output_dir, test_video_id))
    else:
        print(f'Skipping detection. {pkl_file} already exists.')

    if not os.path.exists('{}/{}_0.txt'.format(output_dir, test_video_id)):
        generate_detection_files(
            detections=test_results,
            output_dir=output_dir,
            prefix=test_video_id,
            label_map=label_map,
            groundtruth=False,
            threshold=.0  # FIXME: Should we set a threshold or use every single detection
        )
        
        gt_detections_path = '{}/{}/groundtruths'.format(groundtruth_files_path, test_video_id)
        if os.path.exists('{}/results'.format(output_dir)):  # pascalvoc.py will stall if the folder is not empty
            print('{}/results already exists. Must be emptied before calling pascalvoc or it\'ll stall.'.format(output_dir))
            shutil.rmtree('{}/results'.format(output_dir))
        os.makedirs('{}/results'.format(output_dir), exist_ok=False)

        cmdline_str = f'python {root_dir}/{pascalvoc} --det {root_dir}/{output_dir} -detformat xyrb ' +\
                      f'-gt {root_dir}/{gt_detections_path} -gtformat xyrb -np ' +\
                      f'--classes "{",".join(classes)}" -sp {root_dir}/{output_dir}/results'
        # print(f'Launching pascalvoc.py: {cmdline_str}')
        cmdline = shlex.split(cmdline_str)
        proc = sp.Popen(cmdline, stdout=sp.PIPE, stderr=sp.PIPE)
        out, err = proc.communicate()

    # Initialize accuracy of all classes because some result files don't contain all of them
    accuracy = {
        c: [0]
        for c in classes
    }
    accuracy['test_video'] = [test_video_id]
    accuracy['mAP'] = [0]

    # iii. Get AP & mAP metrics from results.txt
    with open('{}/results/results.txt'.format(output_dir), 'r') as f:
        lines = f.readlines()

    label = None
    for l in lines:
        if label is not None:
            ap = float(l.split(' ')[1].replace('%', ''))
            accuracy[label][0] = float(ap)
            label = None

        if 'Class: ' in l:
            label = l.split(' ')[1].replace('\n', '')
        elif 'mAP' in l:
            mAP = l.split(' ')[1].replace('%', '')
            accuracy['mAP'][0] = float(mAP)

    # pbar.update(1)
    return accuracy


def annotate_videos_parallel(args):
    test_video = str(args['test_video'])
    test_video_id = args['test_video'].stem
    
    gt_dir = args['gt_dir']
    datasets_dir = args['datasets_dir']
    data_dir = '{}/{}/data'.format(datasets_dir, test_video_id)
    images_dir = '{}/{}/images'.format(datasets_dir, test_video_id)
    if os.path.exists('{}/annotations.csv'.format(data_dir)):
        print(f'{test_video_id} has been already annotated.')
        return

    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    min_annotation_score = args['min_annotation_score']
    classes = args['classes']

    annotate_video(
                filename=str(test_video),
                groundtruth='{}/{}.pkl'.format(gt_dir, test_video_id),
                images_dir=images_dir,
                data_dir=data_dir,
                valid_classes=classes,
                label_map=None,  # groundtruths are annotated with mscoco label_map
                reshape=[320],
                min_score=min_annotation_score,
                max_boxes=10,
                img_format='jpg'
            )


def automated_training_loop(config):
    # Global vars
    paths = config['paths']
    root_dir = paths['root_dir']
    gt_dir = paths.get('gt_dir', '../../ground_truths/bcn/ref')
    pascalvoc = paths.get('pascalvoc', '../../accuracy-metrics/pascalvoc.py')

    datasets_dir = paths.get('dataset_dir', 'datasets')
    groundtruth_files_path = paths.get('groundtruth_files_path', 'groundtruths')
    workspace_dir = paths.get('workspace', '.')

    # Paths relative to workspace
    train_dir = os.path.join(workspace_dir, paths.get('train_dir', 'training'))
    saved_models_dir = os.path.join(workspace_dir, paths.get('saved_models_dir', 'trained_models'))
    detection_files_path = os.path.join(workspace_dir, paths.get('detetion_files_path', 'tmp'))

    train_dataset_path = os.path.join(workspace_dir, paths.get('train_dataset_path', 'videos/train'))
    test_dataset_path = [
        '{}/{}/0'.format(workspace_dir, paths.get('test_dataset_path', 'videos/test')),
        '{}/{}/1'.format(workspace_dir, paths.get('test_dataset_path', 'videos/test'))
    ]

    annotation = config['annotation']
    classes = annotation.get('classes', 'car,person,traffic light,motorcycle,truck,bus').split(',')
    min_annotation_score = annotation.getfloat('min_annotation_score', 0.5)
    frame_skipping = annotation.getint('frame_skipping', 5)
    use_only_classes_on_dataset = annotation.getboolean('use_only_classes_on_dataset', True)

    dataset_style = {}
    dataset_ratio = {}
    dataset_style['train'] = annotation.get('train_dataset_style', "current")
    dataset_ratio['train'] = annotation.getfloat('train_dataset_ratio', 1.0)
    dataset_style['test'] = annotation.get('test_dataset_style', "current")
    dataset_ratio['test'] = annotation.getfloat('test_dataset_ratio', 1.0)

    train_config = config['train']
    template_config = train_config.get('template_config', 'pipeline_template.config')
    num_train_steps = train_config.getint('num_train_steps', 5000)
    batch_size = train_config.getint('batch_size', 32)
    base_ckpt = train_config.get('base_ckpt', '')
    checkpoint_every_n = train_config.getint('checkpoint_every_n', 1000 if num_train_steps > 1000 else num_train_steps)
   
    test_config = config['test']
    skip_testing = test_config.getboolean('skip_testing', False)
    min_test_score = test_config.getfloat('min_test_score', 0.0)
    max_test_boxes = test_config.getint('max_test_boxes', 10)
    min_test_score_gt = test_config.getfloat('min_test_score_gt', 0.5)
    max_test_workers = test_config.getint('max_workers', 1)

    set_gpu_config()

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

    if not os.path.exists(groundtruth_files_path):
        print('Directory to store groundtruth files ({}) does not exist. Creating...'.format(groundtruth_files_path))
        os.makedirs(groundtruth_files_path)

    if not os.path.exists(detection_files_path):
        print('Directory to store temporary detection files ({}) does not exist. Creating...'.format(detection_files_path))
        os.makedirs(detection_files_path)

    # PRE A: Set train and test datasets:
        # i. one whole day never used for training
        # ii. all videos used in previous checkpoints. Will be moved from train_dataset
        # iii. same video used for current training iteration
    train_dataset = sorted([Path(v) for v in glob('{}/**/*.mkv'.format(train_dataset_path), recursive=True)])
    test_dataset = [
        sorted([Path(v) for v in glob('{}/**/*.mkv'.format(test_dataset_path[0]), recursive=True)]),
        sorted([Path(v) for v in glob('{}/**/*.mkv'.format(test_dataset_path[1]), recursive=True)])
    ]
    all_videos_dataset = train_dataset + test_dataset[0] + test_dataset[1]
    trained_videos = [v.stem for v in test_dataset[1]]  # All videos in test_dataset[1] have been already used to train

    # PRE B: Generate label_map with valid classes used to annotate/train 
    if use_only_classes_on_dataset:
        label_map = {}
        dataset_classes = []
    else:
        label_map = generate_label_map(classes)
        dataset_classes = classes


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

    for i in trange(len(all_videos_dataset),
            desc="Generating all groundtruth files in advance",
            file=sys.__stderr__):
        test_video = all_videos_dataset[i]
        test_video_id = test_video.stem
        pickle_path = '{}/{}.pkl'.format(gt_dir, test_video_id)
        output_dir = '{}/{}/groundtruths'.format(groundtruth_files_path, test_video_id)
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
            threshold=min_test_score_gt
        )

    annotate_args = [{
        'test_video': test_video,
        'datasets_dir': datasets_dir,
        'gt_dir': gt_dir,
        'classes': classes,
        'min_annotation_score': min_annotation_score
    } for test_video in all_videos_dataset]

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(
            executor.map(annotate_videos_parallel, annotate_args),
            total=len(annotate_args),
            desc='Annotating videos',
            file=sys.__stdout__))

    # Description for each step of the pipeline
    pipeline_desc = [
        '[Step 1] Annotating training video',
        '[Step 2] Generating .record files',
        '[Step 3] Configuring pipeline.config',
        '[Step 4] Training from latest checkpoint',
        '[Step 5] Exporting trained model',
        '[Step 6] Evaluating newly trained model',
        '[Step 7] Updating datasets'
    ]

    # For each video:
    for train_video in tqdm(train_dataset,
                            total=len(train_dataset),
                            desc='Processing training video',
                            file=sys.__stdout__):
        # print('Processing training video {}/{}'.format(i, len(train_dataset)))
        # Progress bar to show step of the training pipeline
        pbar = tqdm(pipeline_desc, total=len(pipeline_desc), file=sys.__stdout__) 
        pbar.set_description(pipeline_desc[0])
        
        # Experiment vars
        train_video_id = train_video.stem
        data_dir = '{}/{}/data'.format(datasets_dir, train_video_id)
        images_dir = '{}/{}/images'.format(datasets_dir, train_video_id)
        train_model_dir = '{}/{}'.format(train_dir, train_video_id)
        exported_model_dir = '{}/{}'.format(saved_models_dir, train_video_id)

        # TODO: Check if folder exists or, at least, is empty
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        # ii. Create training subfolder
        os.makedirs(train_model_dir, exist_ok=True)

        # 1. Annotate video.
        # If the video was previously annotated during another experiment, don't repeat
        if not os.path.exists('{}/annotations.csv'.format(data_dir)):
            # import pdb; pdb.set_trace()
            annotations = annotate_video(
                filename=str(train_video),
                groundtruth='{}/{}.pkl'.format(gt_dir, train_video_id),
                images_dir=images_dir,
                data_dir=data_dir,
                valid_classes=classes,
                label_map=None,  # groundtruths are annotated with mscoco label_map
                val_ratio=0.2,
                reshape=[320],
                min_score=min_annotation_score,
                max_boxes=10,
                img_format='jpg'
            )

            if not all(annotations):
                # The video didn't yeld enough detections, skip its training
                continue
        else:
            print(f'Skipping annotation of {train_video_id} because it has been found.')
        
        pbar.set_description(pipeline_desc[1])
        pbar.update(1)
        # 2. Generate train.record and test.record using csv with annotations
        # 2.1 First, add new classes found to label_map, if any.
        # for dataset in ['train', 'test']:

        # 2.2 If incremental_dataset is True, create a new joint tfrecord with all previous datasets
        train_size = 0
        if not os.path.exists('{}/label_map.pbtxt'.format(train_model_dir)):
            for dataset in ['train', 'test']:
                # if dataset_style[dataset] in ['current', 'incremental']:
                #     print(f'FIXME')
                #     sys.exit(1)

                #     # Read csv with annotations from train_video
                #     classes_in_train_video = pd.read_csv('{}/{}_label.csv'.format(data_dir, dataset))['class'].unique()
                #     new_classes = [c 
                #                    for c in classes 
                #                    if c in classes_in_train_video and c not in dataset_classes
                #     ]
                #     # Add to dataset_classes at the end
                #     if len(new_classes) > 0:
                #         dataset_classes += new_classes
                #         label_map = generate_label_map(dataset_classes)
                #         print(f'Added {len(new_classes)} new classes to dataset classes ({dataset_classes})')
                
                if dataset_style[dataset] == "current":
                    csv_files = ['{}/{}/annotations.csv'.format(datasets_dir, train_video_id)]
                    raise Exception  # FIXME
                elif dataset_style[dataset] == "incremental":
                    # FIXME: incremental is as of now broken as test dataset may contain train images.

                    # Get paths to all annotations.csv
                    annotations_path = Path(datasets_dir).glob('**/annotations.csv')
                    csv_files = [
                            str(csv) 
                            for csv in annotations_path
                            if any([v in str(csv) for v in trained_videos+[train_video_id]])]
                    valid_classes = classes if dataset == 'train' else dataset_classes
                    dataset_tfrecord = generate_tfrecord_from_csv(
                        '{}/{}.record'.format(train_model_dir, dataset),
                        csv_files,
                        ratio=dataset_ratio[dataset],
                        valid_classes=valid_classes,
                        frame_skipping=frame_skipping
                    )
                elif dataset_style[dataset] == 'all':
                    # Get paths to all annotations.csv
                    annotations_path = Path(datasets_dir).glob('**/annotations.csv')
                    ratio = dataset_ratio[dataset]
                    if dataset == 'train':
                        dataset_videos = [v.stem for v in train_dataset]
                    else:
                        dataset_videos = [v.stem for v in test_dataset[0]]
                        if dataset_ratio['test'] < 1:  # test ds size is set wrt train size
                            ratio = dataset_ratio['test'] * train_size

                    csv_files = [
                            str(csv) 
                            for csv in annotations_path
                            if any([v in str(csv) for v in dataset_videos])]
                    valid_classes = classes if dataset == 'train' else dataset_classes
                    dataset_tfrecord = generate_tfrecord_from_csv(
                        '{}/{}.record'.format(train_model_dir, dataset),
                        csv_files,
                        ratio=ratio,
                        valid_classes=valid_classes,
                        frame_skipping=frame_skipping
                    )

                if dataset == 'train':
                    train_size = len(dataset_tfrecord)
                    print(f'Training dataset size: {train_size} with {len(dataset_tfrecord.filename.unique())} images.')
                    # Get classes in dataset in the same order as in the config file
                    dataset_classes = [
                            c
                            for c in classes 
                            if c in dataset_tfrecord['class'].unique()]
                    label_map = generate_label_map(dataset_classes)
        
        else:  # Load label_map.pbtxt
            label_map = load_pbtxt('{}/label_map.pbtxt'.format(train_model_dir))
            dataset_classes = [c['name'] for c in label_map.values()]

        pbar.set_description(pipeline_desc[2])
        pbar.update(1)
        # 3. Configure Pipeline
        # i. Find latest checkpoint in the training folder
        checkpoints = sorted(Path(train_dir).glob('**/ckpt-*.index'),
                                       key=os.path.getmtime,
                                       reverse=True)
                                       
        # If no checkpoints found 
        if len(checkpoints) == 0:
            # latest_checkpoint = os.path.join(root_dir, base_ckpt)
            latest_checkpoint = base_ckpt
        else:
            # If there is a model trained with train_video, get that one.
            latest_checkpoint = None
            for ckpt in checkpoints:
                if train_video_id in str(ckpt):
                    latest_checkpoint = str(ckpt)
                    print(f'Found checkpoint with train_video_id. Using it ({latest_checkpoint}).')
                    break
            if latest_checkpoint is None:
                latest_checkpoint = str(checkpoints[0])
            
            latest_checkpoint = latest_checkpoint.replace('.index', '')

        print(f'latest checkpoint found: {latest_checkpoint}')

        pipeline_config_file = '{}/{}/pipeline.config'.format(train_dir, train_video_id)

        # ii. Generate pipeline.config file base on template
        if not os.path.exists(pipeline_config_file):
            configure_pipeline(
                template=template_config,
                output=pipeline_config_file,
                checkpoint=latest_checkpoint,
                tfrecord_dir=train_model_dir,
                data_dir=data_dir,
                model_dir=train_model_dir,
                classes=dataset_classes,
                batch_size=batch_size,
            )

        pbar.set_description(pipeline_desc[3])
        pbar.update(1)

        if train_video_id in latest_checkpoint:  
        # Means the latest checkpoint corresponds to the current training video
            eval_proc = None
        else:

            # 4.1 Launch a separate process to monitor evaluation
            eval_proc = mp.Process(target=eval_continuously,
                                   args=(
                                       pipeline_config_file,
                                       train_model_dir,  # model_dir
                                       train_model_dir,  # checkpoint_dir
                                       num_train_steps,
                                       20,  # wait_interval
                                       180
                                   ))
            eval_proc.start()
            print(f'evaluation process started with pid {eval_proc.pid}')
            if eval_proc.exitcode is not None:
                print(f'evaluation process exited after start with code {eval_proc.exitcode}')

            # 4.2 Launch training with latest checkpoint
            # checkpoint_every_n = num_train_steps if num_train_steps < 1000 else 1000
            train_loop(
                pipeline_config=pipeline_config_file,
                model_dir=train_model_dir,
                num_train_steps=num_train_steps,
                checkpoint_every_n=checkpoint_every_n
            )

        # 5. Export trained model
        pbar.set_description(pipeline_desc[4])
        pbar.update(1)
        if not os.path.exists('{}/saved_model/saved_model.pb'.format(exported_model_dir)):  
            export_trained_model(
                pipeline_config_path=pipeline_config_file,
                trained_checkpoint_dir=train_model_dir,
                output_dir=exported_model_dir
            )

            # 4.3 stop eval_proc (we place it after export to give it more time to process last checkpoint)
            if eval_proc is not None:
                eval_proc.join(100)
                if eval_proc.exitcode is None:
                    eval_proc.terminate()
                    print(f'evaluation process has been terminated.')
                else:
                    print(f'evaluation process terminated correctly.')
                # eval_proc.close()

        pbar.set_description(pipeline_desc[5])
        pbar.update(1)
        # 6. Test new model against all eval dataset.
        # First, load the recently exported model
        detector = load_saved_model('{}/saved_model/'.format(exported_model_dir))
        test_accuracy = None  # DataFrame to store accuracies of all three test datasets
        
        if dataset_style['train'] == 'all':
            train_videos_to_test = train_dataset
        else:
            train_videos_to_test = [train_video]

        if not skip_testing:
            for test_ds_id, test_ds in enumerate([test_dataset[0], test_dataset[1], train_videos_to_test]):
                if len(test_ds) == 0:
                    continue
                if test_ds_id > 0:
                    continue
            
                # i. Run detections using new model on all videos of the test_ds
                # FIXME: Should the test input data be resized?
                try:
                    test_args = [
                        {
                            'test_video': test_video,
                            'pascalvoc': pascalvoc,
                            'root_dir': root_dir,
                            'detection_files_path': detection_files_path,
                            'groundtruth_files_path': groundtruth_files_path,
                            'output_dir': '{}/{}/detections_{}'.format(
                                                        detection_files_path,
                                                        train_video_id,
                                                        test_video.stem
                                                    ),
                            # 'detector': deepcopy(detector),
                            'classes': dataset_classes,
                            'detector': detector,
                            'min_score': min_test_score,
                            'max_boxes': max_test_boxes,
                            # 'label_map': deepcopy(label_map),
                            'label_map': label_map,
                            # 'save_frame_to': None,
                            # 'resize': None
                        }
                        for test_video in test_ds
                    ]
                except Exception as e:
                    import pdb; pdb.set_trace()
                    print(str(e))

                dataset_accuracy = None
                with ThreadPoolExecutor(max_workers=max_test_workers) as executor:
                    results = list(tqdm(
                        executor.map(eval_detector, test_args),
                        total=len(test_args),
                        desc='Evaluating videos',
                        file=sys.__stdout__))
                    for result in results:

                        video_accuracy = result
                        df = pd.DataFrame.from_dict(video_accuracy)
                        if dataset_accuracy is None:
                            dataset_accuracy = df 
                        else:
                            dataset_accuracy = dataset_accuracy.append(df, ignore_index=True)

                # iv. Save csv with dataset accuracies and add averages to the global
                dataset_accuracy.to_csv('{}/dataset_{}_accuracy.csv'.format(exported_model_dir, test_ds_id), index=False)

                ds_mean = dataset_accuracy.mean() 
                ds_mean['test_dataset'] = test_ds_id
                if test_accuracy is None:
                    test_accuracy = pd.DataFrame([], columns=ds_mean.keys())
                test_accuracy = test_accuracy.append(ds_mean, ignore_index=True)

            test_accuracy.to_csv('{}/test_accuracy.csv'.format(exported_model_dir), index=False)

            trained_videos.append(train_video_id)
        
        # if we used all train videos for the dataset, we're done
        if dataset_style['train'] == 'all':
            break

        pbar.set_description(pipeline_desc[6])
        pbar.update(1)
        # 7. Move the training video to test_dataset[1]
        # i. Move video file
        shutil.move(str(train_video), str(test_dataset_path[1]))
        # ii. Add video to the test_dataset list
        new_video_path = '{}/{}'.format(test_dataset_path[1], train_video.name)
        test_dataset[1].append(Path(new_video_path))
        


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", required=True, help="File containing training config")
    args = args.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    automated_training_loop(config)

if __name__ == '__main__':
    main()
