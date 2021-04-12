#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Methods related to the execution of DNN Models"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

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