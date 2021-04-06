# import the necessary packages
import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
import shlex
import subprocess as sp
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from six import BytesIO
from tqdm import tqdm

# Tensorflow
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

try:
    from object_detection.builders import model_builder
    from object_detection.utils import config_util
except Exception:
    print('No module object_detection.')
    pass

import cv2
import imutils

sys.path.append('../')
# Auxiliary functions
from dnn.utils import generate_detection_files
from dnn.tftrain import eager_eval_loop
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
    print(f'Loading checkpoint {ckpt_to_load}')
    if ckpt_id is not None:
        ckpt_to_load_ = [c for c in manager.checkpoints if ckpt_id in c]
        if len(ckpt_to_load_) == 1:
            ckpt_to_load = ckpt_to_load_[0]
            print(f'Loaded checkpoint {ckpt_to_load}')
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


def load_model(model, ckpt_id=None):
    if 'saved_model' in model:
        return load_saved_model(model)
    # elif 'checkpoint' in model:
    else:
        return load_checkpoint_model(model, f'{model}/pipeline.config', ckpt_id)
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


def inputs(tfrecord, batch_size=1, num_epochs=1):
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
    args.add_argument("--ckpt-id", default=None, help="Id of the ckpt to load")
    args.add_argument("-l", "--label-map", default=None, help="Label map for the model")
    args.add_argument("--min-score", type=float, default=0, help="minimum score for detections")
    args.add_argument("--roi-size", nargs='+', default=None, help="Size of the ROI to crop images")
    args.add_argument("--use-api", action="store_true", default=False, help="Evaluate using Object Detection API.")
    
    args.add_argument("--batch-size", type=int, default=1, help="batch size for inferences")
    args.add_argument("--show", action="store_true", default=False, help="show detections")

    config = args.parse_args()
    min_score = config.min_score

    if config.roi_size is not None \
            and len(config.roi_size) != 2 \
            and int(config.roi_size[0]) != -1:
        print('[ERROR] ROI Size must have exactly 2 dimensions '
              'or -1 to set it to the model\'s input size')
        sys.exit(1)

    output_dir = '/tmp/detections'
    if config.output is not None:
        output_dir = config.output

    os.makedirs(output_dir, exist_ok=True)

    if config.label_map is None:
        label_map = MSCOCO
    else:
        label_map = load_pbtxt(config.label_map)

    print(config.model)
    print(config.dataset)
    day_model = False

    cols = ['model', 'ckpt-id', 'exp', 'train_scene', 'eval_scene', 'eval_ds']
    all_results = pd.DataFrame([], columns=cols)
    if os.path.isfile(f'{output_dir}/results.csv'):
        all_results = pd.read_csv(f'{output_dir}/results.csv')

    for model in config.model:
        model_nn = [p for p in model.split('/') if any([prefix in p for prefix in ['edge', 'ref']])][0]
        model_type = 'saved' if 'saved_model' in model else 'checkpoint'
        if 'base_model' in model:
            exp = 'base_model'
            model_scene = 'None'
        elif any([p in model for p in ['day', 'night', 'morning']]):
            day_model = True
            exp = [p for p in model.split('/') if model_nn in p][0]
            model_scene = exp.split('-')[1]
        else:
            model_scene = [p for p in model.split('/') if 'scene' in p][0]
            exp = '-'.join([p for p in model.split('/') 
                    if any([prefix in p 
                        for prefix in [model_nn, model_scene, 'frozen', 'augmented']])])
            print(f'EXP: {exp}')

        if len(all_results[(all_results['exp'] == exp) & \
                (all_results['train_scene'] == model_scene) & \
                (all_results['ckpt-id'] == config.ckpt_id) & \
                (all_results['model'] == model)]) == len(config.dataset):
            print(all_results[all_results['exp'] == exp])
            print(config.dataset)
            print(f'{exp} already exists')
            continue

        detector = None
        if not config.use_api:
            detector = load_model(model, config.ckpt_id)
            if detector is None:
                continue
        model_dir = output_dir + '/' + model_nn + '/' + exp + '/' + model_scene
        for ds in config.dataset:
            eval_scene = None
            if day_model:
                eval_scene, eval_ds = Path(ds).stem.split('_')
            else:
                if 'scene' in ds:
                    eval_scene = [p for p in ds.split('/') if 'scene' in p][0]
                
                if model_scene != 'None':
                    if eval_scene != model_scene:
                        continue
                    print(f'{model} eval on {eval_scene}')
                
                eval_ds = [p for p in Path(ds).parts if 'dataset' in p][0]

            prev_results = all_results[(all_results['exp'] == exp) & \
                    (all_results['model'] == model) & \
                    (all_results['train_scene'] == model_scene) & \
                    (all_results['eval_scene'] == eval_scene) & \
                    (all_results['ckpt-id'] == config.ckpt_id) & \
                    (all_results['eval_ds'] == eval_ds)]
            if len(prev_results)  > 0:
                print(prev_results)
                print(f'{exp} on eval {ds} already exists')
                continue

            eval_scene = '' if eval_scene is None else eval_scene

            # else:
            #     assert False # TODO
            
            ckpt_id = config.ckpt_id
            if ckpt_id is None:
                ckpt_id = 'latest'
            if not config.use_api:
                results_dir = model_dir + '/' + ckpt_id + '/' + eval_ds + '/' + eval_scene

                if os.path.exists(f'{results_dir}/results'):
                    import pdb; pdb.set_trace()
                    print(f'Results dir {results_dir}/results exists but didn\'t get it :/')
                    continue
                dets, gt_dets = evaluate(detector, label_map, ds, results_dir, batch_size=config.batch_size, 
                                         min_score=config.min_score, show=config.show,
                                         single_inference=(config.roi_size is None), roi_size=config.roi_size)

                results = compute_accuracy(dets, gt_dets, results_dir, label_map, min_score=config.min_score)

            else:
                prev_results = all_results[(all_results['exp'] == exp) & \
                        (all_results['eval_ds'] == eval_ds) & \
                        (all_results['model'] == model)]
                if len(prev_results) > 0:
                    print(prev_results)
                    continue
                print(f'Evaluating {model}')
                if config.ckpt_id is not None:
                    if not os.path.exists(f'{model}/ckpt-{config.ckpt_id}.index'):
                        print(f'{model}/ckpt-{config.ckpt_id} not found. Skipping')
                        continue
                events = []
                if os.path.exists(f'{model}/eval'):
                    events = [ev for ev in os.listdir(f'{model}/eval') if 'tfevents' in ev]
                    for ev in events:
                        print(f'removing {model}/eval/{ev}')
                        os.remove(f'{model}/eval/{ev}')
                    print(events)

                if len(events) == 0 or True:
                    print(f"\teager_eval_loop(\n"
                            f"\t\tpipeline_config_path={model}/pipeline.config,\n"
                            f"\t\teval_dataset={ds},\n"
                            f"\t\tmodel_dir={model},\n"
                            f"\t\tlabel_map={config.label_map},\n"
                            f"\t\tckpt_id={config.ckpt_id})\n")

                    eager_eval_loop(
                            pipeline_config_path=f'{model}/pipeline.config',
                            eval_dataset=ds,
                            model_dir=model,
                            label_map=config.label_map,
                            ckpt_id=config.ckpt_id)
                    events = [ev for ev in os.listdir(f'{model}/eval') if 'tfevents' in ev]

                metrics = []
                events = [ev for ev in summary_iterator(f'{model}/eval/{events[0]}')]
                for ev in events:
                    for v in ev.summary.value:
                        if 'eval_side_by_side' not in v.tag and 'image' not in v.tag:
                            metrics.append([v.tag, tf.make_ndarray(v.tensor).item()])
                results = {m[0]: m[1] for m in metrics}

            df = pd.DataFrame([[model_nn, config.ckpt_id, exp, model_scene, eval_scene, eval_ds]], columns=cols)
            for k, v in results.items():
                df[k] = v
            all_results = all_results.append(df, ignore_index=True)
            print(df)
            print(all_results)
        all_results.to_csv(f'{output_dir}/results.csv', sep=',', index=False)


def evaluate(detector, label_map, dataset, output_dir, min_score=0, batch_size=1, show=False, debug=True, single_inference=False, roi_size=None):
    if debug:
        os.makedirs(f'{output_dir}/images', exist_ok=True)
    generate_gt = True

    saved_model = False
    if getattr(detector, 'detect', False):
        saved_model = True

    detections = []
    gt_detections = []

    num_boxes_per_img = 100
    img_id = 0
    assert batch_size == 1
    batch_id = 0
    print('Processing image: ', end='')
    for images, shapes, gt_labels, gt_boxes in inputs(dataset, batch_size): 
        print(img_id, end=',', flush=True)
        if not single_inference:
            # size of the image
            img_height, img_width = shapes[0].numpy()

            if len(roi_size) == 1:
                # Get input size of the model
                _, input_shape = detector.preprocess(tf.zeros([1, img_height, img_width, 3]))
                input_height, input_width, _ = input_shape[0].numpy()
            else:
                input_height = int(roi_size[0])
                input_width = int(roi_size[1])

            # compute number of columns and rows
            if img_width < input_width:
                input_height = int(input_height * img_width/input_width)
                input_width = img_width
                # print(f'new input size is {input_width}x{input_height}')
            if img_height < input_height:
                input_width = int(input_width * img_height/input_height)
                input_height = img_height
                # print(f'new input size is {input_width}x{input_height}')
            # import pdb; pdb.set_trace()

            num_columns = math.ceil(img_width / input_width)
            num_rows = math.ceil(img_height / input_height)

            new_col_every = 0 if num_columns == 1 else input_width - math.ceil(((input_width * num_columns) - img_width) / (num_columns-1))
            new_row_every = 0 if num_rows == 1 else input_height - math.ceil(((input_height * num_rows) - img_height) / (num_rows-1))

            full_img = images.numpy()[0]
            part_images = []
            boxes = []
            scores = []
            class_ids = []
            # print(f'Inferences required: {num_columns*num_rows}')
            coords_offsets = []
            for i in range(num_columns):
                for j in range(num_rows):
                    start_x = i*new_col_every
                    end_x = min(start_x+input_width, img_width)
                    # end_x = min((i+1)*new_col_every-1, img_width)
                    start_y = j*new_row_every
                    end_y = min(start_y+input_height, img_height)
                    # end_y = min((j+1)*new_row_every-1, img_height)
                    coords_offsets.append([start_y, start_x])
                    # print(f'[col: {i} - row: {j}] start: x = {start_x}, y = {start_y} - end: x = {end_x}, y = {end_y}')
                    img = full_img[start_y:end_y, start_x:end_x,]

                    # input_tensor = tf.image.convert_image_dtype(img, tf.uint8)[tf.newaxis, ...]
                    part_images.append(img)

            part_images = np.stack([p for p in part_images], axis=0)
            ts0 = time.time()
            if not saved_model:
                input_tensor = tf.cast(part_images, dtype=tf.float32)
                ts1 = time.time()
                results = detector.detect(input_tensor)
                ts2 = time.time()
                # print(f'Done with inferences in {ts2-ts0:.3f}secs ({(ts2-ts1)/(num_columns*num_rows):.3f} per inf. and {ts1-ts0:.2f} for tf.cast)')

                for batch_id in range(len(results['detection_scores'])):
                    boxes.extend(results['detection_boxes'][batch_id])
                    scores.extend(results['detection_scores'][batch_id])
                    class_ids.extend(results['detection_classes'][batch_id])

            else:
                for pimg in part_images:
                    input_img = tf.image.convert_image_dtype(pimg, tf.uint8)[tf.newaxis, ...]
                    ts1 = time.time()
                    results = detector(input_img)
                    ts2 = time.time()
                    fps = 1/(ts2-ts1)
                    print(f'fps: {fps:.2f}')
                    results = {key:value.numpy() for key,value in results.items()}

                    boxes.extend(results['detection_boxes'][batch_id])
                    scores.extend(results['detection_scores'][batch_id])
                    class_ids.extend(results['detection_classes'][batch_id])

            try:
                selected_indices = tf.image.non_max_suppression(
                        boxes=boxes, scores=scores, 
                        max_output_size=100,
                        iou_threshold=0.5,
                        score_threshold=max(min_score, 0.05))
            except Exception:
                import pdb; pdb.set_trace()
            boxes = tf.gather(boxes, selected_indices).numpy()
            scores = tf.gather(scores, selected_indices).numpy()
            class_ids = tf.gather(class_ids, selected_indices).numpy()

            selected_indices_to_img = [int(i/num_boxes_per_img) for i in selected_indices.numpy()]
            new_boxes = []
            for i in range(len(boxes)):
                batch_id = selected_indices_to_img[i]
                ymin, xmin, ymax, xmax = boxes[i]
                ymin, xmin, ymax, xmax = (
                        (ymin * img.shape[0] + coords_offsets[batch_id][0])/img_height,
                        (xmin * img.shape[1] + coords_offsets[batch_id][1])/img_width,
                        (ymax * img.shape[0] + coords_offsets[batch_id][0])/img_height,
                        (xmax * img.shape[1] + coords_offsets[batch_id][1])/img_width
                )

                new_boxes.append([ymin, xmin, ymax, xmax])

            boxes = new_boxes
            # import pdb; pdb.set_trace()

            selected_indices = tf.image.non_max_suppression(
                    boxes=boxes, scores=scores, 
                    max_output_size=100,
                    iou_threshold=0.5,
                    score_threshold=max(min_score, 0.05))
            boxes = tf.gather(boxes, selected_indices).numpy()
            scores = tf.gather(scores, selected_indices).numpy()
            class_ids = tf.gather(class_ids, selected_indices).numpy()


            # # NMS between subimages intersections
            # selected_indices_to_img = [selected_indices_to_img[i] for i in selected_indices.numpy()]
            # if False:



            for batch_id in range(len(results['detection_scores'])):
                if debug:
                    img_ = part_images[batch_id].copy()
                    for i, box in enumerate(results['detection_boxes'][batch_id][:10]):
                        ymin, xmin, ymax, xmax = box
                        ymin, xmin, ymax, xmax = (
                                int(ymin * img.shape[0]),
                                int(xmin * img.shape[1]),
                                int(ymax * img.shape[0]),
                                int(xmax * img.shape[1])
                        )

                        score = results["detection_scores"][batch_id][i]
                        class_id = int(results["detection_classes"][batch_id][i])
                        label = label_map[class_id]["name"]
                        # print(f'[img part: {batch_id}] {label}: {score*100:.2f}')

                        cv2.rectangle(img_, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 1)
                        cv2.putText(img_, f'{label}: {score*100:.2f}%', (int(xmin), int(ymin)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    img_path = f'{output_dir}/images/img_{img_id}-batch_{batch_id}.jpg'
                    # print(f'Saving image to {img_path}')
                    cv2.imwrite(img_path, img_)

            # sys.exit()
            batch_id = 0

        else:
            # import pdb; pdb.set_trace()
            ts0 = time.time()
            results = detector.detect(images) 
            ts1 = time.time()
            # print(f'Done with inferences in {ts1-ts0:.3f}secs.')
            boxes = results['detection_boxes'][batch_id]
            scores = results['detection_scores'][batch_id]
            class_ids = results['detection_classes'][batch_id]

        images = images.numpy()
        gt_labels = gt_labels.numpy()
        gt_boxes = gt_boxes.numpy()

        img = images[batch_id]
        gt_label = gt_labels[batch_id]
        gt_box = gt_boxes[batch_id]

        if show or debug:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        while True:
            if show or debug:
                img = img_bgr.copy()
                # import pdb; pdb.set_trace()
            for i in range(len(boxes)):
                if scores[i] >= min_score:
                    ymin, xmin, ymax, xmax = tuple(boxes[i])
                    if True: # single_inference:
                        # import pdb; pdb.set_trace()
                        (xmin, xmax, ymin, ymax) = (
                                int(xmin * img.shape[1]), 
                                int(xmax * img.shape[1]),
                                int(ymin * img.shape[0]), 
                                int(ymax * img.shape[0])
                            )
                    det = [img_id, int(class_ids[i]), scores[i],
                       xmin, ymin, xmax, ymax]
                    detections.append(det)
                    
                    if (show and scores[i] >= min_score) or (debug and scores[i] >= 0.5):
                        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 1)
                        cv2.putText(img, f'{label_map[int(class_ids[i])]["name"]}: {scores[i]*100:.2f}%', (int(xmin), int(ymin)-10),
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

                    if show or debug:
                        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1)
                        cv2.putText(img, str(label), (int(xmin), int(ymin)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if debug:
                cv2.imwrite(f'{output_dir}/images/detections-{img_id}.jpg', img)

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
                img_id += 1
                break

    # classes = [c['name'] for c in label_map.values()]
    columns = ['frame', 'class_id', 'score',
                'xmin', 'ymin', 'xmax', 'ymax']

    detections = pd.DataFrame(detections, columns=columns)
    detections.to_csv(f'{output_dir}/detections.csv', sep=',', index=False)

    gt_detections = pd.DataFrame(gt_detections, columns=columns)
    gt_detections.to_csv(f'{output_dir}/groundtruths.csv', sep=',', index=False)
    return detections, gt_detections


def compute_accuracy(detections, gt_detections, output_dir, label_map, min_score=0.5):
    pascalvoc = '../accuracy-metrics/pascalvoc.py'

    detections_dir = f'{output_dir}/detections'
    gt_dir = '{}/groundtruths'.format(output_dir)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs('{}/results'.format(output_dir), exist_ok=False)

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
