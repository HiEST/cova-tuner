#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os

import tensorflow.compat.v2 as tf
from object_detection import inputs
from object_detection import model_lib_v2
from object_detection.utils import config_util
from object_detection.builders import model_builder


def eval_continuously(
    pipeline_config_path: str,
    model_dir: str,
    checkpoint_dir: str,
    num_train_steps: int,
    wait_interval: int = 180,
    eval_timeout: int = 3600):

    tf.config.set_soft_device_placement(True)
    
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


def eager_eval_loop(
    pipeline_config_path: str,
    eval_dataset: str,
    model_dir: str,
    label_map: dict = None,
    ckpt_id: str = None):
    """Run continuous evaluation of a detection model eagerly.

    This method builds the model, and continously restores it from the most
    recent training checkpoint (or, if specified, the checkpoint id) in the 
    checkpoint directory & evaluates it on the evaluation dataset.

    Args:
        pipeline_config_path (str): A path to a pipeline config file.
        eval_dataset (str): A path to the .record file with the evaluation dataset.
        model_dir (str): Directory to output resulting evaluation summaries to.
        label_map (dict, optional): Dictionary containing label map to show labels of the detections. Defaults to None.
        ckpt_id (str, optional): checkpoint id to load. Defaults to None.

    Returns:
        [type]: [description]
    """
    sample_1_of_n_eval_on_train_examples=1
    use_tpu=False
    postprocess_on_cpu=False

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
            print(f'{ckpt_id} not found in {manager.checkpoints}')
            return None

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
