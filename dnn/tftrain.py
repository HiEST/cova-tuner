#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os

import tensorflow.compat.v2 as tf

from object_detection import inputs
from object_detection import model_lib_v2
from object_detection.utils import config_util
from object_detection.builders import model_builder

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
            record_summaries=record_summaries,
            checkpoint_max_to_keep=int(num_train_steps/checkpoint_every_n)
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


def eager_eval_loop(
    pipeline_config_path,
    eval_dataset,
    model_dir=None,
    label_map=None,
    ckpt_id=None):
    """Run continuous evaluation of a detection model eagerly.

    This method builds the model, and continously restores it from the most
    recent training checkpoint in the checkpoint directory & evaluates it
    on the evaluation data.

    Args:
        pipeline_config_path: A path to a pipeline config file.
        model_dir: Directory to output resulting evaluation summaries to.
  """
    sample_1_of_n_eval_on_train_examples=1
    use_tpu=False
    postprocess_on_cpu=False
    eval_index=0

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    train_input_config = configs['train_input_config']
    eval_config = configs['eval_config']
    eval_input_config = configs['eval_input_config']
    eval_on_train_input_config = copy.deepcopy(train_input_config)
    eval_on_train_input_config.sample_1_of_n_examples = (
      sample_1_of_n_eval_on_train_examples)

    eval_on_train_input_config.num_epochs = 1

    if label_map is None:
        eval_input_config.label_map_path = '/tmp/label_map.pbtxt'
    else:
        eval_input_config.label_map_path = f'/tf/workspace/edge_autotune/training/detection/eager/{eval_input_config.label_map_path}'
    eval_input_config.tf_record_input_reader.input_path[0] = eval_dataset
    strategy = tf.compat.v2.distribute.get_strategy()
    with strategy.scope():
        detection_model = model_builder.build(model_config=model_config, is_training=False)

    eval_input = strategy.experimental_distribute_dataset(
        inputs.eval_input(
            eval_config=eval_config,
            eval_input_config=eval_input_config,
            model_config=model_config,
            model=detection_model))

  
    global_step = tf.compat.v2.Variable(0, trainable=False, dtype=tf.compat.v2.dtypes.int64)

    ckpt = tf.train.Checkpoint(model=detection_model, step=global_step)
    manager = tf.train.CheckpointManager(ckpt, directory=model_dir, max_to_keep=10)
    ckpt_to_load = manager.latest_checkpoint
    
    if ckpt_id is not None:
        ckpt_to_load_ = [c for c in manager.checkpoints if ckpt_id in c]
        if len(ckpt_to_load_) == 1:
            ckpt_to_load = ckpt_to_load_[0]
        else:
            return None
            print(f'{ckpt_id} not found in {manager.checkpoints}')

    ckpt.restore(ckpt_to_load).expect_partial()

    summary_writer = tf.compat.v2.summary.create_file_writer(
            os.path.join(model_dir, 'eval', eval_input_config.name))
    
    with summary_writer.as_default():
        model_lib_v2.eager_eval_loop(
                detection_model,
                configs,
                eval_input,
                use_tpu=use_tpu,
                postprocess_on_cpu=postprocess_on_cpu,
                global_step=global_step)
