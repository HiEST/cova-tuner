#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os

from google.protobuf import text_format
import tensorflow.compat.v2 as tf
from object_detection import inputs
from object_detection import model_lib_v2
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2


def set_gpu_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def train_loop(pipeline_config: str,
               model_dir: str,
               num_train_steps: int,
               checkpoint_every_n: int = 1000,
               record_summaries: bool = True):

    tf.config.set_soft_device_placement(True)

    strategy = tf.compat.v2.distribute.MirroredStrategy()
    with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path=pipeline_config,
            model_dir=model_dir,
            train_steps=num_train_steps,
            use_tpu=False,
            checkpoint_every_n=checkpoint_every_n,
            record_summaries=record_summaries,
            checkpoint_max_to_keep=int(num_train_steps/checkpoint_every_n)
        )


def export_trained_model(
    pipeline_config_path: str,
    trained_checkpoint_dir: str,
    output_dir: str,
    input_type: str = 'image_tensor'
):
    tf.enable_v2_behavior()
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    config_override = ''
    text_format.Merge(config_override, pipeline_config)

    exporter_lib_v2.export_inference_graph(
        input_type=input_type,
        pipeline_config=pipeline_config,
        trained_checkpoint_dir=trained_checkpoint_dir,
        output_directory=output_dir,
        use_side_inputs=False,
        side_input_shapes='',
        side_input_types='',
        side_input_names='')

