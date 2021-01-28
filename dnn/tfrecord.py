#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import io
import os
from pathlib import Path
import random
from shutil import copyfile
import sys

import numpy as np
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


def frame_skip(filenames, fs):
    prev_selected = 0
    if len(filenames) == 0:
        return []
    selected_fn = [filenames[0]]
    for i, fn in enumerate(filenames[1:]):
        if int(Path(fn).stem) >= (prev_selected + fs):
            selected_fn.append(fn)
            prev_selected = int(Path(fn).stem)

    return selected_fn


def generate_tfrecord_from_csv(
        output_path,
        csv_inputs,
        imgs_subdir='images',
        ratio=1.0,
        valid_classes=None,
        frame_skipping=1,
        write_test=False,
        force_max_representation=True): 
    writer = tf.python_io.TFRecordWriter(output_path)
    all_annotations = None
    random.shuffle(csv_inputs)
    for csv in csv_inputs:
        imgs_dir = '{}/{}'.format('/'.join(csv.split('/')[:-2]), imgs_subdir)
        df = pd.read_csv(csv)
        if frame_skipping > 1:
            if valid_classes is None:
                filenames = df.filename.unique()
            else:
                filenames = df[df['class'].isin(valid_classes)].filename.unique()

            selected_fn = frame_skip(filenames, frame_skipping)
            # selected_fn = [
            #         fn
            #         for i, fn in enumerate(filenames)
            #         # if i%frame_skipping == 0
            #         # Use frame_id in the name
            #         if i == 0 or \
            #                 int(Path(fn).stem) >= int(Path(filenames[i-1]).stem) + frame_skipping
            # ]
            # print(f'selected frames from {csv}:\n\t{selected_fn}')
            # print(f'Originally had {len(df.filename.unique())}:\n\t{df.filename.unique()}')
            # for i, fn in enumerate(filenames):
            #     if i == 0:
            #         print(f'[{i}] selected')
            #     elif int(Path(fn).stem) >= int(Path(filenames[i-1]).stem) + frame_skipping:
            #         print(f'[{i}] selected')
            #     else:
            #         print(f'[{i}] discarded')
            #         print(f'\tframe_id: {int(Path(fn).stem)}')
            #         print(f'\tprev frame_id: {int(Path(filenames[i-1]).stem)}')
            #         print(f'\t+fs: {int(Path(filenames[i-1]).stem) + frame_skipping}')
            df = df[df.filename.isin(selected_fn)]

        df['filename'] = df['filename'].apply(lambda x: os.path.join(imgs_dir, x)) 
        df['csv'] = csv
        if all_annotations is None:
            all_annotations = df
        else:
            all_annotations = all_annotations.append(df, ignore_index=True) 

    if valid_classes is None:
        valid_annotations = all_annotations
    else:
        valid_annotations = all_annotations[all_annotations['class'].isin(valid_classes)]

    valid_ratio = len(valid_annotations.filename.unique()) / len(all_annotations.filename.unique())

    # If ratio > 1, it represents the maximum number of images in tfrecord.
    min_num_files = 0
    if ratio > 1:
        num_files = len(valid_annotations.filename.unique())
        if num_files > ratio:
            print(f'num files > ratio: {num_files} > {ratio}')
            min_num_files = int(ratio)
            max_num_files = int(ratio)
            ratio = ratio / num_files
        else:
            ratio = 1
    elif valid_ratio <= ratio:
        ratio = 1
    else:
        ratio = (ratio / valid_ratio)

    # import pdb; pdb.set_trace()
    if ratio < 1:
        if force_max_representation:
            valid_datasets = valid_annotations.csv.unique()
            selected_imgs = np.array([])
            for csv in valid_datasets:
                csv_df = valid_annotations[valid_annotations.csv == csv]
                csv_imgs = csv_df.filename.unique()
                print(f'train size: {ratio*len(csv_imgs)} (ratio={ratio})')
                if ratio*len(csv_imgs) < 1:
                    random.shuffle(csv_imgs)
                    selected_csv_imgs = [csv_imgs[0]]
                    # tmp_ratio = 1/len(csv_imgs)
                    # selected_csv_imgs, _ = train_test_split(csv_imgs, test_size=1-tmp_ratio)
                else:
                    selected_csv_imgs, _ = train_test_split(csv_imgs, test_size=1-ratio)
                selected_imgs = np.append(selected_imgs, selected_csv_imgs)
        else:
            valid_imgs = valid_annotations.filename.unique()
            selected_imgs, _ = train_test_split(valid_imgs, test_size=1-ratio)
            
        if len(selected_imgs) < min_num_files:
            not_yet_selected = valid_annotations[~valid_annotations.filename.isin(selected_imgs)].filename.unique()
            left_to_select = int(min_num_files - len(selected_imgs))
            print(f'{left_to_select} images left to select.')

            random.shuffle(not_yet_selected)
            selected_imgs = np.append(selected_imgs, not_yet_selected[:left_to_select])
            print(f'now {len(selected_imgs)} images selected.')

        elif len(selected_imgs) > max_num_files:
            random.shuffle(selected_imgs)
            selected_imgs = selected_imgs[:max_num_files]
        # selected_imgs, _ = train_test_split(valid_filenames, test_size=1-ratio)
    else:
        valid_filenames = valid_annotations.filename.unique()
        selected_imgs = valid_filenames

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

    # Copy images to output_dir 
    output_dir = Path(output_path).parent
    dataset = 'train' if 'train' in Path(output_path).stem else 'test'
    imgs_dir = '{}/images/{}/'.format(output_dir, dataset)
    os.makedirs(imgs_dir, exist_ok=True)
    for img in selected_annotations.filename.unique():
        img_src = [p for p in img.split('/') if '2020' in str(p)][0]
        print(f'img src {img_src}')
        img_id = Path(img).stem
        output_img = '{}/{}_{}.jpg'.format(imgs_dir, img_src, img_id)
        print(f'Copying {img} to {output_img}')
        copyfile(img, output_img)
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
