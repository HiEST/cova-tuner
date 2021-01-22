#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2
from google.protobuf import text_format
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2


def load_saved_model(path_to_model):
    return tf.saved_model.load(path_to_model)


def set_gpu_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def train_loop(pipeline_config,
               model_dir,
               num_train_steps,
               checkpoint_every_n=1000,
               record_summaries=True):

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    # tf.config.experimental.set_memory_growth(gpus[1], True)
    tf.config.set_soft_device_placement(True)

    strategy = tf.compat.v2.distribute.MirroredStrategy()
    with strategy.scope():
        model_lib_v2.train_loop(
            pipeline_config_path=pipeline_config,
            model_dir=model_dir,
            train_steps=num_train_steps,
            use_tpu=False,
            checkpoint_every_n=checkpoint_every_n,
            record_summaries=record_summaries
        )


def eval_continuously(
    pipeline_config_path,
    model_dir,
    checkpoint_dir,
    num_train_steps,
    wait_interval=180,
    eval_timeout=3600):

    tf.config.set_soft_device_placement(True)
    set_gpu_config()
    print(f'[EVAL] Strating evaluation on {checkpoint_dir}')
    model_lib_v2.eval_continuously(
        pipeline_config_path=pipeline_config_path,
        model_dir=model_dir,
        train_steps=num_train_steps,
        sample_1_of_n_eval_examples=None,
        sample_1_of_n_eval_on_train_examples=(5),
        checkpoint_dir=checkpoint_dir,
        wait_interval=wait_interval,
        timeout=eval_timeout)
    print(f'[EVAL] Finished evaluation')


def export_trained_model(
    pipeline_config_path,
    trained_checkpoint_dir,
    output_dir,
    input_type='image_tensor'
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
