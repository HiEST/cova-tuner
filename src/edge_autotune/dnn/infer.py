#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Methods related to the execution of DNN Models"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except:
    logging.warning('TensorFlow module could not be loaded.'
                    'Ignore if not using TF models.')
    pass

try:
    from openvino.inference_engine import IECore
except:
    logging.warning('IECore could not be loaded.'
                    'Ignore if not using OpenVINO models.')
    pass

import cv2

from edge_autotune.dnn.tools import load_model, load_pbtxt
from edge_autotune.dnn.dataset import get_dataset_labels


# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


class Model:
    def __init__(
        self,
        model_dir: str,
        max_boxes: int = 100,
        min_score: float = 0,
        iou_threshold: float = 0,
        label_map: str = None):
        """Load Model from model_dir and initializes parameters"""   

        self.detector = load_model(model_dir)
        self.from_checkpoint = self.detector.from_checkpoint

        self.max_boxes = max_boxes
        self.nms = (iou_threshold > 0)
        self.iou_threshold = iou_threshold
        self.min_score = min_score
        self.label_map = None
        
        if label_map:
            if os.path.isfile(label_map):
                self.label_map = load_pbtxt(label_map)
            else:
                raise Exception(f'label map file ({label_map}) not found.')
        else:
            self.label_map = get_dataset_labels('mscoco')
            assert self.label_map
        

    def run(self, batch: list):
        """Run inference on the batch of images.

        Args:
            batch (list): Batch of images to process.

        Returns:
            list: list with results, one per image in the batch. Results are dicts with ['boxes', scores', 'class_ids']
        """
        batch_size = len(batch)
        batch_np = np.stack([img for img in batch], axis=0)
        
        if self.from_checkpoint:
            input_tensor = tf.cast(batch_np, dtype=tf.float32)
        else:
            input_tensor = tf.image.convert_image_dtype(batch_np, tf.uint8)
        results = self.detector.detect(input_tensor)

        batch_results = []

        for batch_id in range(batch_size):
            boxes = results['detection_boxes'][batch_id]
            scores = results['detection_scores'][batch_id]
            class_ids = results['detection_classes'][batch_id]
            
            if self.nms:
                selected_indices = tf.image.non_max_suppression(
                        boxes=boxes, scores=scores, 
                        max_output_size=self.max_boxes,
                        iou_threshold=self.iou_threshold,
                        score_threshold=max(self.min_score, 0.05))
            
                boxes = tf.gather(boxes, selected_indices).numpy()
                scores = tf.gather(scores, selected_indices).numpy()
                class_ids = tf.gather(class_ids, selected_indices).numpy()

            labels = []
            if self.label_map:
                for c in class_ids:
                    label = self.label_map[int(c)]['name']
                    labels.append(label)

            batch_results.append({
                'boxes': boxes.tolist(),
                'scores': scores.tolist(),
                'class_ids': [int(c) for c in class_ids.tolist()],
                'labels': labels,
            })

        return batch_results

class ModelIE:
    def __init__(
        self,
        model_dir: str,
        device: str = 'CPU',
        max_boxes: int = 100,
        min_score: float = 0.,
        iou_threshold: float = 0.,
        label_map: str = None,
    ):
        self.max_boxes = max_boxes
        self.nms = (iou_threshold > 0)
        self.iou_threshold = iou_threshold
        self.min_score = min_score
        self.label_map = None
        
        if label_map:
            if os.path.isfile(label_map):
                self.label_map = load_pbtxt(label_map)
            else:
                raise Exception(f'label map file ({label_map}) not found.')
        else:
            self.label_map = get_dataset_labels('mscoco')
            assert self.label_map

        self.ie = IECore()
        supported_extensions = ['.xml', '.bin', '.onnx']
        if os.path.isfile(model_dir):
            if Path(model_dir).suffix in supported_extensions:
                self.model = model_dir
            else:
                raise Exception(f'Model format not supported (supported: {", ".join(supported_extensions)}).')
        else:
            try:
                self.model = [
                    fn for fn in os.listdir(model_dir)
                    if Path(fn).suffix in supported_extensions
                ][-1]
            except:
                raise Exception(f'{model_dir} does not contain a supported file ({", ".join(supported_extensions)}).')
    
        logging.info(f'Readin the network: {self.model}')
        self.net = self.ie.read_network(model=self.model)

        if len(self.net.input_info) != 1:
            logging.error('Only single input topologies are supported')
            return -1

        if len(self.net.outputs) != 1 and not ('boxes' in self.net.outputs or 'labels' in self.net.outputs):
            logging.error('Only models with 1 output or with 2 with the names "boxes" and "labels" are supported')
            return -1

        logging.info('Configuring input and output blobs')
        # Get name of input blob
        self.input_blob = next(iter(self.net.input_info))

        # Set input and output precision manually
        self.net.input_info[self.input_blob].precision = 'U8'

        if len(self.net.outputs) == 1:
            self.output_blob = next(iter(self.net.outputs))
            self.net.outputs[self.output_blob].precision = 'FP32'
        else:
            self.net.outputs['boxes'].precision = 'FP32'
            self.net.outputs['labels'].precision = 'U16'

        logging.info('Loading the model to the plugin')
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)

        _, _, self.net_h, self.net_w = self.net.input_info[self.input_blob].input_data.shape

    
    def run(self, batch: list):
        """Run inference on the batch of images.

        Args:
            batch (list): Batch of images to process.

        Returns:
            list: list with results, one per image in the batch. Results are dicts with ['boxes', scores', 'class_ids']
        """

        batch_results = []
        for img in batch:
            if img.shape[:-1] != (self.net_h, self.net_w):
                img = cv2.resize(img, (self.net_w, self.net_h))

            # Change data layout from HWC to CHW
            img = img.transpose((2, 0, 1))
            # Add N dimension to transform to NCHW
            img = np.expand_dims(img, axis=0)

            logging.info('Starting inference in synchronous mode')
            results = self.exec_net.infer(inputs={self.input_blob: img})

            
            if len(self.net.outputs) == 1:
                results = results[self.output_blob]
                # Change a shape of a numpy.ndarray with results ([1, 1, N, 7]) to get another one ([N, 7]),
                # where N is the number of detected bounding boxes
                detections = results.reshape(-1, 7).tolist()
            else:
                detections = results['boxes']
                labels = results['labels']
                assert False

            boxes = []
            scores = []
            class_ids = []
            for i, detection in enumerate(detections):
                _, class_id, score, xmin, ymin, xmax, ymax = detection

                if score < self.min_score:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                scores.append(score)
                class_ids.append(class_id)

            labels = []
            if self.label_map:
                for c in class_ids:
                    label = self.label_map[int(c+1)]['name']
                    labels.append(label)

            batch_results.append({
                'boxes': boxes,
                'scores': scores,
                'class_ids': class_ids,
                'labels': labels,
            })

        return batch_results

