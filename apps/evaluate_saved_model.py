# import the necessary packages
import argparse
from functools import partial
import json
import os
from pathlib import Path
import shlex
import subprocess as sp
import sys

import numpy as np
import pandas as pd
from PIL import Image
from six import BytesIO
from tqdm import tqdm

# Tensorflow
import tensorflow as tf
import tensorflow_hub as hub

from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2
import imutils

sys.path.append('../')
# Auxiliary functions
from dnn.utils import generate_detection_files
from utils.detector import run_detector
from utils.datasets import MSCOCO


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


def load_saved_model(model):
    detection_model = tf.saved_model.load(model)

    def detect_fn(self, batch):
        # input_tensor = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
        # input_tensor = tf.cast(batch, dtype=tf.float32)
        result = self(batch)
        result = {key:value.numpy() for key,value in result.items()}
        return result

    detection_model.detect = partial(detect_fn, detection_model)
    return detection_model 


def load_checkpoint_model(checkpoint_dir, pipeline_config, ckpt_id=None):
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
        else:
            return None
            print(f'{ckpt_id} not found in {manager.checkpoints}')

    ckpt.restore(ckpt_to_load).expect_partial()

    def detect_fn(self, batch):
        # print(img.shape)
        # img = imutils.resize(img, width=300, height=300)
        # img = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
        # input_tensor = tf.convert_to_tensor(img.numpy(), dtype=tf.float32)
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
    return detection_model


def load_model(model):
    if 'saved_model' in model:
        return load_saved_model(model)
    # elif 'checkpoint' in model:
    else:
        return load_checkpoint_model(model, f'{model}/pipeline.config', "ckpt-6")
    # else:
    #     print(f'Model type could not be detected for {model}')
    #     raise Exception


def tf_parse(eg):
    example = tf.io.parse_example(
            eg[tf.newaxis], {
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/height': tf.io.FixedLenFeature([], tf.int64),
                'image/width': tf.io.FixedLenFeature([], tf.int64),
                'image/object/class/label': tf.io.VarLenFeature(tf.int64),
                'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32)
            })

    return tf.image.decode_jpeg(example['image/encoded'][0]), \
            [example['image/height'][0], example['image/width'][0]], \
            example['image/object/class/label'].values, \
            [example['image/object/bbox/ymin'].values, example['image/object/bbox/xmin'].values, \
            example['image/object/bbox/ymax'].values, example['image/object/bbox/xmax'].values]


def load_dataset_map(tfrecord):
    dataset = tf.data.TFRecordDataset(tfrecord)
    decoded = dataset.map(tf_parse)
    return decoded


def resize(image, img_shape, label, box, width=300, height=300):
    image = tf.image.resize(image, [width, height])
    image = tf.image.convert_image_dtype(image, tf.uint8)[tf.newaxis, ...]
    # box = box * img_shape.numpy()[0]
    # box = tf.cast(box, tf.int64)

    return image, img_shape, label, box


def transpose(image, img_shape, label, box):
    label = tf.transpose(label)
    box = tf.transpose(box)
    return image, img_shape, label, box


def inputs(tfrecord, batch_size=8, num_epochs=1):
    dataset = tf.data.TFRecordDataset(tfrecord)
    # dataset.repeat(num_epochs)

    dataset = dataset.map(tf_parse)
    dataset = dataset.map(transpose)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.map(resize)

    return dataset#.__iter__()


# Read dataset from tfrecord
def load_dataset_from_tfrecord(tfrecord):
    dataset = tf.data.TFRecordDataset(tfrecord)
    images_np = []
    gt_boxes = []
    gt_labels = []

    for raw_example in tqdm(dataset, total=len(dataset)):
        parsed = tf.train.Example.FromString(raw_example.numpy())
        feature = parsed.features.feature

        raw_img = feature['image/encoded'].bytes_list.value[0]
        img = Image.open(BytesIO(raw_img))
        (im_width, im_height) = img.size
        img_np = np.array(img.getdata()).reshape(im_height, im_width, 3).astype(np.uint8)
        images_np.append(img_np)
        labels = [l for l in feature['image/object/class/label'].int64_list.value]

        boxes = [[] for _ in labels]
        coords = ['ymin', 'xmin', 'ymax', 'xmax']
        for coord in coords:
            for i, c in enumerate(feature[f'image/object/bbox/{coord}'].float_list.value):
                boxes[i].append(c)

        gt_boxes.append(np.array([box for box in boxes], dtype=np.float32))
        gt_labels.append(np.array([l for l in labels], dtype=np.int32))

    return [images_np,
            gt_boxes,
            gt_labels]


def main():
    # construct the argument parser and parse the arguments
    args = argparse.ArgumentParser()

    args. add_argument("-d", "--dataset", nargs='+', default=None, help="Path to the dataset to evaluate.")
    args. add_argument("-o", "--output", default=None, help="Path to the output dir.")

    # Detection/Classification
    args.add_argument("-m", "--model", nargs='+', default=None, help="Model for image classification")
    args.add_argument("-l", "--label-map", default=None, help="Label map for the model")
    args.add_argument("--min-score", type=float, default=0, help="minimum score for detections")
    
    args.add_argument("--batch-size", type=int, default=1, help="batch size for inferences")
    args.add_argument("--show", action="store_true", default=False, help="show detections")

    config = args.parse_args()
    min_score = config.min_score

    output_dir = '/tmp/detections'
    if config.output is not None:
        output_dir = config.output

    exist_ok = os.path.isfile(f'{output_dir}/results.csv')
    os.makedirs(output_dir, exist_ok=exist_ok)

    if config.label_map is None:
        label_map = MSCOCO
    else:
        label_map = load_pbtxt(config.label_map)

    print(config.model)
    # print(config.dataset)
    # print(label_map)

    cols = ['model', 'exp', 'train_scene', 'eval_scene', 'eval_ds']
    all_results = pd.DataFrame([], columns=cols)
    if exist_ok:
        all_results = pd.read_csv(f'{output_dir}/results.csv')

    for model in config.model:
        model_nn = 'edge' if 'edge' in model else 'ref'
        model_type = 'saved' if 'saved_model' in model else 'checkpoint'
        if 'base_model' in model:
            exp = 'base_model'
            model_scene = 'None'
        else:
            exp = [p for p in model.split('/') if any([prefix in p for prefix in [model_nn, 'frozen', 'augmented']])][0]
            model_scene = [p for p in model.split('/') if 'scene' in p][0]

        if len(all_results[all_results['exp'] == exp]) == len(config.dataset):
            print(all_results[all_results['exp'] == exp])
            print(config.dataset)
            print(f'{exp} already exists')
            continue

        detector = load_model(model)
        if detector is None:
            continue
        model_dir = output_dir + '/' + model_nn + '/' + exp + '/' + model_scene
        for ds in config.dataset:
            eval_scene = None
            if 'scene' in ds:
                eval_scene = [p for p in ds.split('/') if 'scene' in p][0]
            if 'n-fold' in model:
                if eval_scene != model_scene:
                    continue
                print(f'{model} eval on {eval_scene}')
            
            if len(all_results[(all_results['exp'] == exp) & (all_results['eval_scene'] == eval_scene)]) > 0:
                print(f'{exp} on eval {ds} already exists')
                continue

            eval_scene = '' if eval_scene is None else eval_scene

            # else:
            #     assert False # TODO
            
            eval_ds = [p for p in Path(ds).parts if 'dataset' in p][0]
            results_dir = model_dir + '/' + eval_ds + '/' + eval_scene
            print(results_dir)
            results = evaluate(detector, label_map, ds, results_dir, 
                               min_score=config.min_score, show=config.show)

            df = pd.DataFrame([[model_nn, exp, model_scene, eval_scene, eval_ds]], columns=cols)
            for k, v in results.items():
                df[k] = v
            all_results = all_results.append(df, ignore_index=True)
            print(df)
            print(all_results)
        all_results.to_csv(f'{output_dir}/results.csv', sep=',', index=False)


def evaluate(detector, label_map, dataset, output_dir, min_score=0, batch_size=1, show=False):
    pascalvoc = '../accuracy-metrics/pascalvoc.py'

    detections_dir = f'{output_dir}/detections'
    gt_dir = '{}/groundtruths'.format(output_dir)
    generate_gt = True
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs('{}/results'.format(output_dir), exist_ok=True)

    detections = []
    gt_detections = []

    img_id = 0
    for images, shapes, gt_labels, gt_boxes in inputs(dataset, batch_size): 
        results = detector.detect(images) 

        images = images.numpy()
        gt_labels = gt_labels.numpy()
        gt_boxes = gt_boxes.numpy()
        for batch_id in range(batch_size):
            boxes = results['detection_boxes'][batch_id]
            scores = results['detection_scores'][batch_id]
            class_ids = results['detection_classes'][batch_id]

            img = images[batch_id]
            gt_label = gt_labels[batch_id]
            gt_box = gt_boxes[batch_id]

            if show:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            while True:
                if show:
                    img = img_bgr.copy()
                for i in range(len(boxes)):
                    if scores[i] >= min_score:
                        ymin, xmin, ymax, xmax = tuple(boxes[i])
                        (xmin, xmax, ymin, ymax) = (
                                int(xmin * img.shape[1]), 
                                int(xmax * img.shape[1]),
                                int(ymin * img.shape[0]), 
                                int(ymax * img.shape[0])
                            )
                        det = [img_id, int(class_ids[i]), scores[i],
                           xmin, ymin, xmax, ymax]
                        detections.append(det)
                        
                        if show:
                            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 1)
                            cv2.putText(img, f'{class_ids[i]}: {scores[i]*100:.2f}%', (int(xmin), int(ymin)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if generate_gt:
                    for box, label in zip(gt_box, gt_label):
                        ymin, xmin, ymax, xmax = tuple(box)
                        (xmin, xmax, ymin, ymax) = (
                                int(xmin * img.shape[1]), 
                                int(xmax * img.shape[1]),
                                int(ymin * img.shape[0]), 
                                int(ymax * img.shape[0])
                            )

                        gt_det = [img_id, label, 1,
                                xmin, ymin, xmax, ymax]
                        gt_detections.append(gt_det)

                        if show:
                            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1)
                            cv2.putText(img, str(label), (int(xmin), int(ymin)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if show:
                    cv2.imshow("Detections", img)

                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        sys.exit()
                        break
                    elif key == ord("a"):
                        min_score = min(min_score+0.05, 1)
                    elif key == ord("s"):
                        min_score = max(min_score-0.05, 0)
                    elif key == ord("c"):
                        break
                else:
                    break

            img_id += 1

    # classes = [c['name'] for c in label_map.values()]
    columns = ['frame', 'class_id', 'score',
                'xmin', 'ymin', 'xmax', 'ymax']

    detections = pd.DataFrame(detections, columns=columns)
    detections.to_csv(f'{output_dir}/detections.csv', sep=',', index=False)
    
    if generate_gt:
        gt_detections = pd.DataFrame(gt_detections, columns=columns)
        gt_detections.to_csv(f'{output_dir}/groundtruths.csv', sep=',', index=False)

        ret = generate_detection_files(gt_detections, gt_dir, "detections", 
                label_map=label_map, groundtruth=True) 
        assert ret
        classes_in_gt = gt_detections['class_id'].unique()
        classes = [c['name'] for c in label_map.values() if c['id'] in classes_in_gt]

    ret = generate_detection_files(detections, detections_dir, "detections", 
            label_map=label_map, groundtruth=False, threshold=min_score) 
    assert ret

    cmdline_str = f'python {pascalvoc} --det {detections_dir} -detformat xyrb ' +\
                  f'-gt {gt_dir} -gtformat xyrb -np ' +\
                  f'--classes "{",".join(classes)}" -sp {output_dir}/results'
    cmdline = shlex.split(cmdline_str)
    proc = sp.Popen(cmdline, stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = proc.communicate()

    results_file = f'{output_dir}/results/results.txt'
    results = {c['name']:0 for c in label_map.values()}
    with open(results_file, 'r') as f:
        lines = f.readlines()
        label = ''
        for l in lines:
            if 'Class' in l:
                label = l.split(' ')[1].replace('\n', '')
            elif 'AP: ' in l:
                ap = float(l.split(' ')[1].replace('%', ''))
                if label != '':
                    results[label] = ap
                    label = ''
                else:
                    results['mAP'] = ap

    print(results)
    with open(f'{output_dir}/mAP.txt', 'w') as f:
        json.dump(results, f)
        
    return results

if __name__ == "__main__":
    main()
