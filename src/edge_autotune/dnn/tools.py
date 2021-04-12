#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements diverse tools useful to work with DNN models and their whereabouts"""

from functools import partial
import os

import tensorflow as tf

def load_saved_model(model: str):
    """Load exported pb saved model.

    Args:
        model (str): Path to the directory that contains the saved model pb. 

    Returns:
        TODO: TensorFlow model.
    """
    
    detection_model = tf.saved_model.load(model)

    def detect_fn(self, batch):
        # input_tensor = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
        # input_tensor = tf.cast(batch, dtype=tf.float32)
        result = self(batch)
        result = {key:value.numpy() for key,value in result.items()}
        return result

    detection_model.detect = partial(detect_fn, detection_model)
    detection_model.from_checkpoint = False
    return detection_model 


def load_checkpoint_model(
    checkpoint_dir: str,
    pipeline_config: str,
    ckpt_id: str = None):
    """Load checkpoint model.

    Args:
        checkpoint_dir (str): Directory containing checkpoint to load.
        pipeline_config (str): Path to the pipeline.config to load the checkpoint.
        ckpt_id (str, optional): Checkpoint to load. Defaults to None.

    Returns:
        TODO: Loaded checkpoint model.
    """

    try:
        from object_detection.builders import model_builder
        from object_detection.utils import config_util
    except Exception:
        print('No module object_detection. Checkpoint cannot be loaded.')
        return None

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config, is_training=False)

    ckpt = tf.train.Checkpoint(model=detection_model)
    manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, max_to_keep=10)
    ckpt_to_load = manager.latest_checkpoint
    if ckpt_id is not None:
        ckpt_to_load_ = [c for c in manager.checkpoints if ckpt_id in c]
        if len(ckpt_to_load_) == 1:
            ckpt_to_load = ckpt_to_load_[0]
            print(f'Loaded checkpoint {ckpt_to_load}')
        else:
            print(f'{ckpt_id} not found in {manager.checkpoints}')
            return None

    ckpt.restore(ckpt_to_load).expect_partial()

    def detect_fn(self, batch):
        input_tensor = tf.cast(batch, dtype=tf.float32)
        preprocessed_image, shapes = self.preprocess(input_tensor)
        prediction_dict = self.predict(preprocessed_image, shapes)
    
        result = self.postprocess(prediction_dict, shapes)
        result = {key:value.numpy() for key,value in result.items()}
        # +1 to detected classes as we start counting at 1
        for b in range(len(batch)):
            result['detection_classes'][b] += 1
        return result

    detection_model.detect = partial(detect_fn, detection_model)
    detection_model.from_checkpoint = True
    return detection_model


def load_model(model_dir: str, ckpt_id: str = None):
    """Load model, either saved_model.pb or checkpoint.

    Args:
        model_dir (str): Path to directory containing either saved_model or checkpoint.
        ckpt_id (str, optional): Id of the checkpoint to load. 
            Used only if loading a checkpoint. If None, latest checkpoint will be loaded.
            Defaults to None.

    Raises:
        Exception: Model type could not be detected (not saved_model dir or checkpoint dir without pipeline.config)

    Returns:
        tf.Model: Loaded model.
    """
    if 'saved_model' in model_dir:
        return load_saved_model(model_dir)
    elif os.path.isfile(f'{model_dir}/pipeline.config'):
        return load_checkpoint_model(model_dir, f'{model_dir}/pipeline.config', ckpt_id)
    else:
        raise Exception(f'Model type could not be detected for {model_dir}')


def label_to_id_map(label_map: dict):
    """Convert label_map to id_map.

    Args:
        label_map (dict): label_map (from pbtxt format).

    Returns:
        dict: id_map created from the label_map.
    """
    id_map = {
        c['name']: int(c['id'])
        for c in label_map.values()
    }
    return id_map


def load_pbtxt(filename):
    label_map = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    label = ''
    for l in lines:
        if 'name' in l:
            label = l.split('"')[1]
        elif 'id' in l:
            class_id = int(l.split(':')[1])
            label_map[class_id] = {
                'name': label,
                'id': class_id
            }

    return label_map