import argparse
import configparser
from typing import Tuple
import os
import sys
import time

from datetime import datetime
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tqdm

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker import ModelPackage
from sagemaker.tensorflow import TensorFlow
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput


from edge_autotune.api.client import AWSClient
from edge_autotune.motion import motion_detector as motion

from model_package_arns import ModelPackageArnProvider


def load_config(args):
    config = configparser.ConfigParser(
        converters={'list': lambda x: [i.strip() for i in x.split(',')]})
    config.read(args['config'])

    required_fields = {
        'app': {
            'streams': 'list',
            's3_prefix_key': 'str',
            'dataset_name': 'str',
            'valid_classes': 'list',
        },
        'aws': {
            'role': 'str',
        },
        'ecr': {
            'tfrecord': 'str',
            'tftrain': 'str',
        },
    }
    
    config = {s:dict(config.items(s)) for s in config.sections()}
    # Merge with config with args, having this higher priority
    config['app'] = {**config['app'], **args}
    
    for field, params in required_fields.items():
        try:
            assert config.get(field, None)
        except Exception as e:
            print(f'Missing field {field} in config file.')
            raise e
        for param, valtype in params.items():
            try:
                assert config[field].get(param, None)
            except Exception as e:
                print(f"Missing parameter {param} on field {field} in config file")
                raise e
            if valtype == 'list' and isinstance(config[field][param], str):
                config[field][param] = json.loads(config[field].get(param))
    
    return config


def capture_aws(
    streams_list: Tuple[str],
    bucket: str,
    key_prefix: str,
    resize: Tuple[int,int] = (1280, 720),
    crop_motion: bool = False,
    framerate: int = 1,
    max_images: int = 1000,
    min_images: int = 100,
    min_area: int = 1000,
    timeout: int = 0,
    no_show: bool = True,
):
    """Capture and annotate images from stream and generate dataset.

    Args:
        stream (Tuple[str]): List of input streams from which to capture images.
        valid_classes (str, optional): Comma-separated list of classes to detect. If None, all classes will be considered during annotation. Defaults to None
        background_method (int, optional): Method to generate background image. Defaults to 6 (MOG).
        frame_skip (int, optional): Frame skipping value. Defaults to 25.
        min_score (float, optional): Minimum score to accept groundtruth model's predictions as valid. Defaults to 0.5.
        max_images (int, optional): Stop when maximum is reached. Defaults to 1000.
        min_images (int, optional): Prevents timeout to stop execution if the minimum of images has not been reached. Used only if timeout > 0. Defaults to 0.
        min_area (int, optional): Minimum area for countours to be considered as actual movement. Defaults to 1000.
        timeout (int, optional): timeout for capture. When reached, capture stops unless there is a minimum of images enforced. Defaults to 0.
        tmp_dir (str, optional): Path to temporary directory where intermediate results are written. Defaults to '/tmp'.
        first_frame_background (bool, optional): If True, first frame of the stream is chosen as background and never changed. Defaults to False.
        flush (bool, optional): If True, new frames and annotations are written to the output TFRecord as soon as received. Defaults to True.
    """
    max_boxes = 100
    background = motion.BackgroundCV()
    motionDetector = motion.MotionDetector(background=background, min_area_contour=min_area)
    num_selected_images = 0

    motion_warmup_secs = 5
    client = AWSClient(bucket=bucket, key_prefix=key_prefix)

    for stream in tqdm.tqdm(streams_list):
        print(f'capturing images from {stream}')
        cap = cv2.VideoCapture(stream)
        next_frame = 0
        stream_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f'stream fps: {stream_fps}')
        frame_skip = round(stream_fps/framerate)
        motion_warmup_frames = round(motion_warmup_secs*stream_fps)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_skip)
        progress_cap = tqdm.tqdm(range(int(num_frames)), unit=' decoded frames')

        print(f'frameskip of {frame_skip} frames.')
        print(f'Warmup motion: {next_frame} < {motion_warmup_frames}')
        ret = True
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            progress_cap.update(1)
            print(f'next_frame: {next_frame}')

            next_frame += frame_skip
            cap.set(1, next_frame)
            
            regions_proposed, areas = motionDetector.detect(frame)

            if next_frame < motion_warmup_frames:
                print(f'Warmup motion: {next_frame} < {motion_warmup_frames}')
                continue

            imgs_to_upload = []
            if crop_motion:
                for _, roi in enumerate(regions_proposed):
                    imgs_to_upload.append(np.array(frame[roi[1]:roi[3], roi[0]:roi[2]]))
                
            elif len(regions_proposed):
                img = cv2.resize(frame.copy(), resize)
                imgs_to_upload = [img]
                if not no_show:
                    for roi_id, roi in enumerate(regions_proposed):
                        area = areas[roi_id]
                        cv2.rectangle(frame, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (0, 0, 255), 2)
                        cv2.putText(frame, str(area), (int(roi[0]), int(roi[1])-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if len(imgs_to_upload):
                num_selected_images += 1

            if max_images > 0 and num_selected_images >= max_images:
                break

            if not no_show:
                if len(imgs_to_upload):
                    cv2.imshow('frame', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            client.extend(imgs_to_upload)
        
        client.upload_all()

        print(f'Uploaded {num_selected_images} from stream {Path(stream).stem}')


## Auxiliary Functions
def deploy_model(role, num_instances, model_arn, instance_type, model_name, output_path, max_concurrent_transforms=2):
    model = ModelPackage(
        role=role, model_package_arn=model_arn, sagemaker_session=sagemaker.Session()
    )
    # model.deploy(num_instances, instance_type, endpoint_name=model_name)
    transformer = model.transformer(
        instance_count=num_instances,
        instance_type=instance_type,
        output_path=output_path,
        max_concurrent_transforms=max_concurrent_transforms)

    return model, transformer


def batch_transform(data, transformer, batch_output, content_type):
    ts0 = time.time()
    transformer.transform(
        data=data,
        data_type="S3Prefix",
        content_type=content_type,
        input_filter="$",
        join_source= "None",
        output_filter="$",
    )
    ts_create = time.time() - ts0

    ts0 = time.time()
    transformer.wait()
    ts_exec = time.time() - ts0
    print(f'Batch Transform job created in {ts_create:.2f} seconds and executed in {ts_exec:.2f} seconds.')

    assert batch_output == transformer.output_path
    output = transformer.output_path

    return output


def invoke_DL_endpoint(image_path, runtime, endpoint_name, content_type="image/png", bounding_box="no"):
    img = open(image_path, "rb").read()

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=bytearray(img),
        ContentType=content_type,
        CustomAttributes='{"threshold": 0.2}',
        Accept="json",
    )
    result = json.loads(response["Body"].read().decode("utf-8"))
    return result 


def annotate(bucket, imgs_prefix, output_prefix, role):
    instance_type = "ml.m4.xlarge"
    instance_count = 1
    max_concurrent_transforms = 4 if instance_type == "ml.m4.xlarge" else 2
    content_type = "image/png"
    region = "eu-west-1"

    endpoint_name = "yolov3-gt-endpoint"

    model_arn = ModelPackageArnProvider.get_yolov3_model_package_arn(region)

    # The location of the test dataset
    batch_input = 's3://{}/{}'.format(bucket, imgs_prefix)

    # The location to store the results of the batch transform job
    batch_output = 's3://{}/{}'.format(bucket, output_prefix)

    ts0 = time.time()
    model, batch = deploy_model(
        role=role,
        num_instances=instance_count,
        model_arn=model_arn,
        instance_type=instance_type,
        model_name=endpoint_name,
        output_path=batch_output,
        max_concurrent_transforms=max_concurrent_transforms,
    )
    ts1 = time.time()

    print(f'Model deployed successfuly after {ts1-ts0:.2f} seconds.')

    ts0 = time.time()
    output_path = batch_transform(batch_input, batch, batch_output, content_type)
    ts1 = time.time()
    print(f'Images successfuly annotated in {ts1-ts0:.2f} seconds. Results stored in {output_path}')
    return output_path


def generate_manifest(bucket, imgs_prefix, results_prefix, dataset_name, valid_classes):
    min_score = 0.3
    timestamp = datetime.now().isoformat(timespec="milliseconds")

    batch_input = 's3://{}/{}'.format(bucket, imgs_prefix)

    manifest_entries = []
    s3 = boto3.client('s3')
    s3_objects = s3.list_objects_v2(Bucket=bucket, Prefix=results_prefix)['Contents']
    for obj in s3_objects:
        print(obj)
        path, filename = os.path.split(obj['Key'])
        # with open(f'/tmp/{filename}', 'wb') as f:
        with io.BytesIO() as f:
            s3.download_fileobj(bucket, obj['Key'], f)
            f.seek(0)

            annotations = json.load(f)
            img_dict = {'source-ref':f'{batch_input}/{Path(filename).stem}', dataset_name: {}}
            img_dict[dataset_name]['annotations'] = []
            
            for ann in annotations:
                ann_dict = {}

                score = float(ann['score'])
                if score < min_score:
                    continue

                label = ann['id']
                try:
                    class_id = valid_classes.index(label)
                except ValueError:
                    continue

                ann_dict['class_id'] = class_id
                ann_dict['top'] = ann['top']
                ann_dict['left'] = ann['left']
                ann_dict['width'] = ann['right'] - ann['left']
                ann_dict['height'] = ann['top'] - ann['bottom']

                img_dict[dataset_name]['annotations'].append(ann_dict)

            img_dict[f'{dataset_name}-metadata'] = {
                "class-map": {
                    str(class_id):label for class_id,label in enumerate(valid_classes)
                },
                "human-annotated": "no",
                "creation-date": timestamp,
                "type": "groundtruth/object-detection"
            }

            img_json = json.dumps(img_dict)
            manifest_entries.append(img_json)

    manifest_str = '\n'.join(manifest_entries)
    with open('/tmp/manifest.json', 'w') as f:
        f.write(manifest_str)


def generate_tfrecord(bucket, s3_manifest, output_prefix, valid_classes, container, role):
    ts0 = time.time()
    data_processor = Processor(
        role=role, 
        image_uri=container, 
        instance_count=1, 
        instance_type="ml.m4.xlarge",
        volume_size_in_gb=30, 
        max_runtime_in_seconds=1200,
        base_job_name='tf2-object-detection'
    )
    ts1 = time.time()
    print(f'Took {ts1-ts0:.2f} seconds to create data Processor.')

    input_folder = '/opt/ml/processing/input'
    ground_truth_manifest = '/opt/ml/processing/input/manifest.json'
    label_map = {
        str(i): c
        for i, c in enumerate(valid_classes)
    }
    label_map = json.dumps(label_map)
    output_folder = '/opt/ml/processing/output'

    ts0 = time.time()
    data_processor.run(
        arguments= [
            f'--input={input_folder}',
            f'--ground_truth_manifest={ground_truth_manifest}',
            f'--label_map={label_map}',
            f'--output={output_folder}'
        ],
        inputs = [
            ProcessingInput(
                input_name='input',
                source=s3_manifest,
                destination=input_folder
            )
        ],
        outputs= [
            ProcessingOutput(
                output_name='tfrecords',
                source=output_folder,
                destination=f's3://{bucket}/{output_prefix}'
            )
        ]
    )
    ts1 = time.time()
    print(f'Took {ts1-ts0:.2f} seconds to execute data Processor.')


def train(train_input, train_data, eval_data, output_prefix, tensorboard_prefix, container, role):
    
    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path=tensorboard_prefix,
        # container_local_output_path='/opt/training/'
    )

    hyperparameters = {
        "model_dir":"/opt/ml/model",
        "pipeline_config_path": "pipeline.config",
        "checkpoint_dir": "checkpoint/",
        "num_train_steps": 1000,    
        "sample_1_of_n_eval_examples": 1
    }

    estimator = TensorFlow(
        entry_point='train.py', role=role, 
        instance_count=1, instance_type='ml.g4dn.xlarge', 
        source_dir='source_dir',
        output_path=output_prefix, image_uri=container,
        hyperparameters=hyperparameters,
        tensorboard_output_config=tensorboard_output_config)

    
    # job_artifacts_path = estimator.latest_job_tensorboard_artifacts_path()
    # tensorboard_s3_output_path = f'{job_artifacts_path}/train' 

    #We make sure to specify wait=False, so our notebook is not waiting for the training job to finish.
    inputs = {'train': train_input}

    estimator.fit(inputs)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--streams", nargs='+', default=None, help="List of streams to process")
    args.add_argument("--s3-prefix-key", default=None, type=str, help="S3 key prefix where data will be stored")
    args.add_argument("--dataset-name", default=None, type=str, help="Dataset name")
    args.add_argument("--valid-classes", nargs='+', default=None, help="List of valid classes")
    args.add_argument("--show", action='store_true', default=False, help="Show window with results")
    args.add_argument("-f", "--file", default='config.ini', type=str, help="Path to the .ini config file")

    config = vars(args.parse_args())
    config = load_config(config)

    # The session remembers our connection parameters to Amazon SageMaker. We'll use it to perform all of our Amazon SageMaker operations.
    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()

    prefix_key = config['app']['s3_prefix_key']
    imgs_prefix = prefix_key + '/images'
    output_prefix = prefix_key + '/annotations'


    capture_aws(
        streams_list=config['app']['streams'],
        bucket=bucket,
        key_prefix=imgs_prefix,
        no_show=not config['show'],
    )

    annotate(
        bucket=bucket,
        imgs_prefix=imgs_prefix,
        output_prefix=output_prefix,
        role=config['role'],
    )

    generate_manifest(
        bucket=bucket,
        imgs_prefix=imgs_prefix,
        results_prefix=output_prefix,
        dataset_name=config['dataset_name'],
        valid_classes=config['valid_classes'],
    )

    # FIXME: Create two ProcessInput for images and manifest
    s3_manifest = 's3://{}/{}'.format(bucket, imgs_prefix)
    tfrecord_prefix = 's3://{}/{}/{}'.format(bucket, prefix_key, 'tfrecord')
    generate_tfrecord(
        bucket,
        s3_manifest,
        output_prefix=tfrecord_prefix,
        valid_classes=config['app']['valid_classes'],
        container=config['ecr']['tfrecord'],
        role=config['aws']['role'],
        )

    train_data = tfrecord_prefix + '/train.records'
    eval_data = tfrecord_prefix + '/validation.records'
    train_output = 's3://{}/{}/{}'.format(bucket, prefix_key, 'train')
    tensorboard_prefix = 's3://{}/{}/{}'.format(bucket, prefix_key, 'tensorboard')
    train(
        train_input=tfrecord_prefix,
        train_data='train.records',
        eval_data='validation.records',
        output_prefix=train_output,
        tensorboard_prefix=tensorboard_prefix,
        container=config['ecr']['train'],
        role=config['aws']['role'])


if __name__ == '__main__':
    main()