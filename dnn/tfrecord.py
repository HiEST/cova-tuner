#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import io
import os
import random
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple


def label_to_id_map(label_map):
    id_map = {
        c['name']: int(c['id'])
        for c in label_map.values()
    }
    return id_map


def classes_to_id_map(valid_classes):
    id_map = {
        c: i
        for i, c in enumerate(valid_classes)}

    return id_map


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, id_map):
    print(group.filename)
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(id_map[row['class']])

    assert max(xmins + xmaxs + ymins + ymaxs) <= 1.1
    assert min(xmins + xmaxs + ymins + ymaxs) >= 0.0

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def generate_tfrecord(output_path, images_dir, csv_input, label_map):
    writer = tf.python_io.TFRecordWriter(output_path)
    path = os.path.join(images_dir)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    id_map = label_to_id_map(label_map)
    for group in grouped:
        tf_example = create_tf_example(group, path, id_map)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


def generate_joint_tfrecord(output_path, images_dirs, csv_inputs, label_map=None):
    writer = tf.python_io.TFRecordWriter(output_path)
    id_map = label_to_id_map(label_map)
    for img_dir, csv in zip(images_dirs, csv_inputs):
        examples = pd.read_csv(csv)
        grouped = split(examples, 'filename')
        path = os.path.join(img_dir)
        for group in grouped:
            tf_example = create_tf_example(group, path, id_map)
            writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


def generate_tfrecord_from_csv(output_path, csv_inputs, ratio=1.0, valid_classes=None, write_test=False): 
    print('Generating tfrecord')
    writer = tf.python_io.TFRecordWriter(output_path)
    all_annotations = None
    for csv in csv_inputs:
        imgs_dir = '{}/images'.format('/'.join(csv.split('/')[:-2]))
        df = pd.read_csv(csv)
        df['filename'] = df['filename'].apply(lambda x: os.path.join(imgs_dir, x)) 
        if all_annotations is None:
            all_annotations = df
        else:
            all_annotations = all_annotations.append(df, ignore_index=True) 

    if valid_classes is None:
        valid_annotations = all_annotations
    else:
        valid_annotations = all_annotations[all_annotations['class'].isin(valid_classes)]

    valid_ratio = len(valid_annotations) / len(all_annotations)
    if valid_ratio <= ratio:
        ratio = 1
    else:
        ratio = (ratio / valid_ratio)

    # FIXME: ratio works now wrt number of annotations, not images/frames
    valid_filenames = valid_annotations.filename.unique()
    selected_imgs, _ = train_test_split(valid_filenames, test_size=1-ratio)
    selected_annotations = valid_annotations[valid_annotations.filename.isin(selected_imgs)]
    # since we used train_test_split over filenames, selected_annotations is not shuffled.
    # shuffle selected_annotations or they'll be ordered in tfrecord.
    selected_annotations = selected_annotations.sample(frac=1).reset_index(drop=True)
    selected_classes = selected_annotations['class'].unique()
    id_map = classes_to_id_map(selected_classes)
    grouped = split(selected_annotations, 'filename')
    random.shuffle(grouped)
    for group in grouped:
        tf_example = create_tf_example(group, '', id_map)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)

    print_tfrecord_summary(valid_annotations, selected_annotations)

    return selected_annotations


def print_tfrecord_summary(annotations, picked):
    num_entries = len(annotations)
    labels = annotations['class'].unique()
    entries_per_label = {}
    for label in labels:
        df = annotations[annotations['class'] == label]
        entries_per_label[label] = len(df)

    print('Entries: {} - {} ({:.2f}%)'.format(
        num_entries,
        len(picked),
        len(picked)/num_entries*100))

    labels = picked['class'].unique()
    print('Labels: {} - {} ({:.2f}%)'.format(
        len(entries_per_label.keys()),
        len(labels),
        len(labels)/len(entries_per_label)*100))

    for label in labels:
        df = picked[picked['class'] == label]
        print('\tclass {}: {} - {} ({:.2f}% - {:.2f}%)'.format(
            label,
            entries_per_label[label],
            len(df),
            entries_per_label[label] / num_entries*100,
            len(df) / len(picked)*100))
