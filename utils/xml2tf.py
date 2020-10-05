""" Sample TensorFlow XML-to-TFRecord converter

usage: generate_tfrecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -w WORKSPACE_PATH, --workspace WORKSPACE_PATH
                        Path to the workspace with the expected directory structure.
  -c, --csv
                        write csv file.
"""

import os
import sys
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse
import random
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow XML-to-TFRecord converter")
parser.add_argument("-w",
                    "--workspace",
                    help="Path to the folder where the input .xml files are stored.",
                    type=str)
parser.add_argument("-c",
                    "--csv",
                    help="Write csv file.",
                    default=False, action="store_true")

args = parser.parse_args()

label_map = label_map_util.load_labelmap(f'{args.workspace}/annotations/label_map.pbtxt')
label_map_dict = label_map_util.get_label_map_dict(label_map)


def xml_to_csv(workspace, train_eval_ratio=0.7):
    """Iterates through all .xml files (generated by labelImg) in a given directory and combines
    them in a single Pandas dataframe.

    Parameters:
    ----------
    workspace : str
        The path to the workspace
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    xml_path = f'{workspace}/annotations/xmls'
    xml_list = {'train': [], 'eval': []}
    xml_files = glob.glob(xml_path + '/*.xml')
    random.shuffle(xml_files)

    train_len = int(len(xml_files) * train_eval_ratio)
    train_files = random.sample(xml_files, train_len)

    imgs_path = f'{workspace}/images'
    os.makedirs(f'{imgs_path}/train', exist_ok=True)
    os.makedirs(f'{imgs_path}/eval', exist_ok=True)

    for i, xml_file in enumerate(xml_files):
        # print(xml_file)
        
        if xml_file in train_files:
            dataset = 'train'
        else:
            dataset = 'eval'

        tree = ET.parse(xml_file)
        root = tree.getroot()
        img_path = None
        for member in root.findall('object'):
            # print(member[0].text)
            # print(root.find('filename').text)
            img_filename = root.find('filename').text
            value = (f'images/{dataset}/{img_filename}',
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     float(member[2][0].text),
                     float(member[2][1].text),
                     float(member[2][2].text),
                     float(member[2][3].text)
                     )
            xml_list[dataset].append(value)

            if not img_path:
                img_path = f"{root.find('path').text}"

        # print(f'moving {img_path} to {imgs_path}/{dataset}/{img_filename}')
        shutil.move(img_path, f'{imgs_path}/{dataset}/{img_filename}')
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df_train = pd.DataFrame(xml_list['train'], columns=column_name)
    xml_df_eval = pd.DataFrame(xml_list['eval'], columns=column_name)
    return xml_df_train, xml_df_eval


def class_text_to_int(row_label):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    # print(f'Creating tf example: {group.filename}')
    with tf.gfile.GFile(os.path.join(path, group.filename), 'rb') as fid:
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
        classes.append(class_text_to_int(row['class']))

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


def main(_):
    # imgs_path = os.path.join(args.image_dir)
    # workspace should contain the expected directory structure:
    # workspace/
    #   annotations/
    #       annotations/xmls/
    #       annotations/label_map.pbtxt
    #   dataset/
    #
    # After executing the script, the following directories will be created
    # workspace/
    #   images/
    #       images/train
    #       images/eval
    #   tf_record/
    #       train.record
    #       eval.record

    workspace = os.path.join(args.workspace)

    train_examples, eval_examples = xml_to_csv(workspace)
    examples = {'train': train_examples, 'eval': eval_examples}
    
    os.makedirs(f'{workspace}/tf_record', exist_ok=True)
    for dataset in ['train', 'eval']:
        output_tf = f'{workspace}/tf_record/{dataset}.record'
        writer = tf.python_io.TFRecordWriter(output_tf)
        grouped = split(examples[dataset], 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, workspace)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecord file: {}'.format(output_tf))
    
        if args.csv:
            output_csv = f'{workspace}/{dataset}.csv'
            examples[dataset].to_csv(output_csv, index=None)
            print('Successfully created the CSV file: {}'.format(output_csv))


if __name__ == '__main__':
    tf.app.run()

