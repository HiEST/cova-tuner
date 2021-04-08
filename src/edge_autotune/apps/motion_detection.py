#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
# from tensorflow.image import non_max_suppression
import torch
import torchvision
from torchvision import transforms
from torchvision.models import mobilenet_v2, resnet18, resnet50, resnet101
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pandas as pd
 
# from api.client import offload_single_frame
# from api.server import get_top_torch, infer_torch, get_top_tf
# from training.binary import Net2
sys.path.append('../')
from utils.datasets import IMAGENET, MSCOCO
from utils.detector import init_detector, run_detector
from utils.motion_detection import Background, MotionDetection
from utils.nms import non_max_suppression_fast
from dnn.utils import load_pbtxt

url = 'http://localhost:5000/infer'

def get_yolov5():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).fuse().autoshape()  # for PIL/cv2/np inputs and NMS
    return model
 

def object_detection_torch(model, imgs):
    results = model(imgs, size=640)
    # import pdb; pdb.set_trace()
    return results


def get_resnet50_backbone(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
     
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model


def get_mobilenet_v2_backbone(num_classes):
  # load a pre-trained model for classification and return
  # only the features
  backbone = torchvision.models.mobilenet_v2(pretrained=True).features
  # FasterRCNN needs to know the number of
  # output channels in a backbone. For mobilenet_v2, it's 1280
  # so we need to add it here
  backbone.out_channels = 1280
  
  # let's make the RPN generate 5 x 3 anchors per spatial
  # location, with 5 different sizes and 3 different aspect
  # ratios. We have a Tuple[Tuple[int]] because each feature
  # map could potentially have different sizes and
  # aspect ratios 
  anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
  
  # let's define what are the feature maps that we will
  # use to perform the region of interest cropping, as well as
  # the size of the crop after rescaling.
  # if your backbone returns a Tensor, featmap_names is expected to
  # be [0]. More generally, the backbone should return an
  # OrderedDict[Tensor], and in featmap_names you can choose which
  # feature maps to use.
  roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                  output_size=7,
                                                  sampling_ratio=2)
  
  # put the pieces together inside a FasterRCNN model
  model = FasterRCNN(backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
  
  return model


def get_instance_segmentation_model(backbone='mobilenet_v2', num_classes=91):
    if backbone == 'mobilenet_v2':
        return get_mobilenet_v2_backbone(num_classes)
    elif backbone == 'faster_rcnn':
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    else:
        return get_resnet50_backbone(num_classes)


def scale_roi(roi, factor, frame_shape):
    roi_width = roi[2] - roi[0]
    roi_height = roi[3] - roi[1]

    new_width = roi_width * factor
    new_height = roi_height * factor
    
    diff_width = new_width - roi_width
    diff_height = new_height - roi_height

    new_roi = [
        max(roi[0] - int(diff_width/2), 0),
        max(roi[1] - int(diff_height/2), 0),
        min(roi[2] + int(diff_width/2), frame_shape[1]),
        min(roi[3] + int(diff_height/2), frame_shape[0])
    ]
    return new_roi


def main():
    args = argparse.ArgumentParser()

    # App's I/O
    args.add_argument("-v", "--video", default=None, help="path to the video file")
    args.add_argument("-o", "--output", default="./", help="path to the output dir")


    # Object Detection
    args.add_argument("-m", "--model", default=None, help="Model for object detection")
    args.add_argument("-i", "--input-size", type=int, nargs='+', default=None, help="Model's input size")
    args.add_argument("-t", "--threshold", type=float, default=0.5, help="Confidence threshold to accept predictions")
    args.add_argument("--classify-only", help="Run only image classification (with PyTorch).", action="store_true")
    args.add_argument("--offload", help="Offload inference for object detection.", action="store_true")
    args.add_argument("--no-infer", help="Skip inference.", action="store_true")

    # Motion Detection
    args.add_argument("--first-pass-bg", action="store_true", help="Give a first pass to the video to get a more accuracte background")
    args.add_argument("--no-merge-rois", action="store_true", help="Don't merge ROIs on scene")
    args.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args.add_argument("--no-average", help="use always first frame as background.", action="store_true")
    args.add_argument("--min-roi-size", type=int, nargs='+', default=(1,1), help="Model's input size")
    args.add_argument("--scale-roi", type=float, default=1, help="Factor to scale ROI")
    args.add_argument("--save-bg", type=str, default=None, help="Save background as image to recover it later.")
    args.add_argument("--load-bg", type=str, default=None, help="Path to background to load.")


    # App control
    args.add_argument("--start-after", type=int, default=0, help="Frames to skip before starting detection")
    args.add_argument("--stop-after", type=int, default=0, help="Stop after a number of frames of the video.")
    args.add_argument("--stop-after-detection", type=int, default=-1, help="Stop after a number of frames after the first successful detection.")
    args.add_argument("--no-show", help="Do not show results.", action="store_true")
    args.add_argument("--debug", help="Show debug info.", action="store_true")

    args.add_argument("--frame-skip", type=int, default=0, help="Frame skipping")

    # Dataset annotation
    args.add_argument("--save-rois", type=str, help="Save RoIs in png format.", default=None)
    args.add_argument("--overwrite", action="store_true", help="Overwrite dataset, if exists")
    args.add_argument("--append", action="store_true", help="Append new RoIs to dataset, if exists")


    config = args.parse_args()

    background = Background(
        no_average=config.no_average,
        skip=10,
        take=10,
        use_last=15,
    )

    if config.first_pass_bg:
        cap = cv2.VideoCapture(config.video)
        ret, frame = cap.read()
        frame_id = 0
        while ret:
            background.update(frame)
            if not config.no_show:
                bg = cv2.resize(background.background.copy(), (1280, 768))
                cv2.putText(bg, f'frame: {frame_id}', (10, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.imshow('Current Background', bg)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    sys.exit()
            ret, frame = cap.read()
            frame_id += 1
        cv2.destroyAllWindows()
        cap.release()
        background.freeze() 

        if config.save_bg is not None:
            bg = background.background.copy()
            cv2.imwrite(config.save_bg, bg)
    elif config.load_bg is not None:
        bg = cv2.imread(config.load_bg, cv2.IMREAD_UNCHANGED)
        background.background = bg.copy()
        background.freeze()

    input_size = config.input_size
    if input_size is None:
        input_size = (320,320)

    motionDetector = MotionDetection(
        background=background,
        min_area_contour=config.min_area,
        roi_size=config.min_roi_size,
        merge_rois=(not config.no_merge_rois),
    )

    framework = 'tf'
    if not config.offload and not config.no_infer:
        if config.classify_only:
            if config.model is not None and '.pth' in config.model:
                if os.path.isfile(config.model):
                    classifier = Net2()
                    classifier.load_state_dict(torch.load(config.model))
                    classifier.eval()
                    label_map = {
                        0: 'no_car',
                        1: 'car'
                    }
                else:
                    print(f'[ERROR] Model not found in {config.model}')
                    sys.exit(1)
            
            else:
                if config.model == 'resnet18':
                    classifier = resnet18(pretrained=True)
                elif config.model == 'resnet50':
                    classifier = resnet50(pretrained=True)
                elif config.model == 'resnet101':
                    classifier = resnet101(pretrained=True)
                else:
                    classifier = mobilenet_v2(pretrained=True)
                label_map = IMAGENET
        else:
            if 'saved_model' in config.model:
                detector = tf.saved_model.load(config.model)
                label_map_path = [str(p) for p in Path(config.model).parents if 'saved_model' not in str(p)][0]
                # label_map_path = '/'.join(label_map_path)
                label_map_path = '{}/label_map.pbtxt'.format(label_map_path)
                if os.path.isfile(label_map_path):
                    label_map = load_pbtxt(label_map_path)
                else:
                    # Use MSCOCO
                    label_map = MSCOCO
                print(f'Loading saved model {config.model} and label map {label_map_path}')
            elif config.model is not None:
                if config.model in ['ref', 'edge']:
                    detector = init_detector(config.model)
                elif config.model == 'yolov5':
                    detector = get_yolov5() 
                    framework = 'torch'
                else:
                    detector = get_instance_segmentation_model(config.model)
                    detector.eval()
                    framework = 'torch'
                label_map = MSCOCO
            else:
                detector = init_detector()
                label_map = MSCOCO

    # save_class = config.save_roi_class
    if config.save_rois is not None:
        # save_dir = f'data/{save_class}_dataset'
        rois_saved = 0
        if not os.path.exists(config.save_rois):
            os.makedirs(config.save_rois)
            os.makedirs(f'{config.save_rois}/certain')
            os.makedirs(f'{config.save_rois}/uncertain')
            # os.makedirs(f'{save_dir}/{save_class}')
            # os.makedirs(f'{save_dir}/not_{save_class}')
        else:
            if config.overwrite == config.append:
                print('[ERROR] Dataset exists but overwrite and append are either not set or equal.')
                sys.exit(1)

            if config.overwrite:
                files = [f for f in Path(config.save_rois).glob('*.png')]
                # Remove images:
                for f in files:
                    print(f'Removing image {str(f)}')
                    Path.unlink(f)
                # Remove empty directory
                dirs = [d[0] for d in os.walk(config.save_rois)]
                for d in dirs:
                    print(f'Removing empty dir {d}')
                    Path.rmdir(d)
                
            elif config.append:
                files = [int(f.stem) for f in Path(config.save_rois).rglob('*.png')]
                files.sort()
                rois_saved = 0 if len(files) == 0 else files[-1]+1
                print(f'Starting at RoI id {rois_saved}')
            

    cap = cv2.VideoCapture(config.video)
    ret, frame = cap.read()
    max_boxes = 10
    min_score = config.threshold
    frame_id = 0
    frames_since_first_detection = -1
    detections = []
    prev_valid_detections = []
    while ret:
        if config.frame_skip > 0 and frame_id % config.frame_skip != 0:
            frame_id += 1
            ret, frame = cap.read()
            continue

        bg_ts0 = time.time()
        motion_boxes, areas = motionDetector.detect(frame)
        bg_ts1 = time.time()
        infer_ts = 0
        if frame_id < config.start_after:
            motion_boxes = []
 
        # collage = np.zeros(frame.shape, np.uint8)
        # collage_limits = {
        #     'right': 0,
        #     'bottom': 0,
        # }
        collage = None
        if len(motion_boxes):
            original_frame = frame.copy()
            num_areas_to_detect = len([area for area in areas if area >= 2*config.min_area])
            for roi_id, roi in enumerate(motion_boxes):
                if areas[roi_id] < 2*config.min_area:
                    continue

                # collage_roi = [
                #     collage_limits['right'],                        # left
                #     collage_limits['bottom'],                          # top
                #     collage_limits['right'] + (roi[2] - roi[0]),    # right
                #     collage_limits['bottom'] + (roi[3] - roi[1])       # bottom  
                # ]

                # print(f'collage roi:')
                # print(f'\theight: {collage_roi[1]}:{collage_roi[3]}')
                # print(f'\twidth: {collage_roi[0]}:{collage_roi[2]}')
                # collage[collage_roi[1]:collage_roi[3], collage_roi[0]:collage_roi[2]] = np.array(original_frame[roi[1]:roi[3], roi[0]:roi[2]])
                # collage_limits['right'] = collage_limits['right'] + (roi[2] - roi[0])
                # if collage_limits['right'] >= frame.shape[1]:
                #     collage_limits['bottom'] = collage_limits['bottom'] + (roi[3] - roi[1])
                #     collage_limits['right'] = 0
                import pdb; pdb.set_trace()
                if collage is None:
                    collage = original_frame[roi[1]:roi[3], roi[0]:roi[2]]
                else:
                    cropped = original_frame[roi[1]:roi[3], roi[0]:roi[2]]
                    if collage.shape[0] > collage.shape[1]:
                        collage = cv2.hconcat([collage, cropped])
                    else:
                        collage = cv2.vconcat([collage, cropped])

                print(type(collage))


                roi = scale_roi(roi, config.scale_roi, frame.shape)
                cropped_roi = np.array(original_frame[roi[1]:roi[3], roi[0]:roi[2]])


                infer_ts0 = time.time()
                if config.no_infer:
                    boxes = []
                    scores = []
                    class_ids = []
                elif config.offload:
                    ret, preds = offload_single_frame(cropped_roi, url, 'edge', 'tf')
                    boxes = preds['boxes']
                    scores = preds['scores']
                    class_ids = preds['idxs']

                elif config.classify_only:
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]) 
                    ])
                    img = transform(cropped_roi)
                    img = img.unsqueeze(0)
                    ts0 = time.time()

                    if config.model is not None and '.pth' in config.model:
                        with torch.no_grad():
                            preds = classifier(img)
                            preds = torch.max(torch.log_softmax(preds, dim=1), dim=1).float()
                            scores = [1]
                            class_ids = [int(c) for c in preds.indices.flatten()]
                    else:
                        preds = classifier(img)
                        preds = torch.nn.functional.softmax(preds[0], dim=0)
                        preds = preds.detach().numpy()
                        top = get_top_torch(preds, topn=1)
                        scores = [float(s) for s in top['scores']]
                        class_ids = [int(c) for c in top['idxs']]
                        
                    ts1 = time.time()
                    # print(f'inference took {ts1-ts0:.3} seconds')
                    boxes = []
                    # preds = get_top_torch(preds)
                    # scores = [float(s)/100 for s in preds['scores']]
                    # class_ids = [int(c) for c in preds['idxs']]
                    # scores = [s for s in preds.values.flatten()]
                    # import pdb; pdb.set_trace()
                elif framework == 'tf':
                    preds = run_detector(detector, cropped_roi, input_size=input_size) 
                    boxes = preds['detection_boxes'][0]
                    scores = preds['detection_scores'][0]
                    class_ids = preds['detection_classes'][0]

                    selected_indices = tf.image.non_max_suppression(boxes, scores, 10, iou_threshold=0.3)
                    selected_boxes = np.array(tf.gather(boxes, selected_indices))
                    # print(f'Removed {len(boxes) - len(selected_boxes)} boxes with NMS')
                    boxes = selected_boxes
                    scores = np.array(tf.gather(scores, selected_indices))
                    class_ids = np.array(tf.gather(class_ids, selected_indices))

                else:
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((input_size[0], input_size[1])),
                        transforms.ToTensor()
                    ])
                    img = transform(cropped_roi)
                    img = img.unsqueeze(0)
                    # preds = detector(img)
                    if config.model == 'yolov5':
                        preds = object_detection_torch(detector, img)
                        if len(preds.xyxy[0]) > 0:
                            preds = preds.xyxy[0].detach().numpy()
                            boxes = [[preds[0][0], preds[0][1], preds[0][2], preds[0][3]]]
                            scores = [preds[0][4]]
                            class_ids = [int(preds[0][5])]
                        else:
                            boxes = []

                    else:
                        preds = detector(img)[0]
                        boxes = preds['boxes'].to('cpu').detach().numpy()
                        scores = preds['scores'].to('cpu').detach().numpy()
                        class_ids = preds['labels'].to('cpu').detach().numpy()
            
                infer_ts1 = time.time()
                infer_ts += infer_ts1 - infer_ts0
                                   
                cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
                cv2.putText(frame, 'ROI', (roi[0], roi[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(frame, str(areas[roi_id]), (roi[0], roi[3]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                if config.classify_only:
                    # import pdb; pdb.set_trace()
                    if scores[0] >= min_score:
                        print(f'[{class_ids[0]}]{label_map[class_ids[0]]} ({scores[0]*100}%)')
                        cv2.putText(frame, f'{label_map[class_ids[0]]} ({scores[0]*100}%)', (roi[0], roi[1]+10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

                for i in range(min(len(boxes), max_boxes)):
                    # if boxes[i] not in boxes_nms:
                    #     continue
                    class_id = int(class_ids[i])
                    class_name = label_map[class_id]['name']
                    # if scores[i] >= 0.01:
                    #     print(f'{class_name}: {scores[i]:.3f}')
                    if scores[i] >= min_score:
                        # print(f'score for class {label_map[str(int(class_ids[i]))]["name"]}: {scores[i]}')
                        if framework == 'tf':
                            ymin, xmin, ymax, xmax = tuple(boxes[i])
                            (left, right, top, bottom) = (roi[0] + xmin * cropped_roi.shape[1], roi[0] + xmax * cropped_roi.shape[1],
                                                          roi[1] + ymin * cropped_roi.shape[0], roi[1] + ymax * cropped_roi.shape[0])
                        else:
                            xmin, ymin, xmax, ymax = tuple(boxes[i])
                            (left, right, top, bottom) = (roi[0] + xmin, roi[0] + xmax,
                                                          roi[1] + ymin, roi[1] + ymax)
                        
                        label = label_map[class_id]['name']
                        det = [frame_id, class_id, scores[i], int(left), int(top), int(right), int(bottom), roi_id, num_areas_to_detect]
                        detections.append(det)

                        if not config.no_show:
                            display_str = "{}: {}%".format(label, int(100 * scores[i]))
                            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)
                            cv2.putText(frame, display_str, (int(left), int(top)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        if config.save_rois is not None:
                            print(f'saving roi from class {class_name} with score {scores[i]:.2f} at frame {frame_id}')
                            save_dir = f'{config.save_rois}' 
                            if scores[i] >= min(0.99, min_score * 1.5):
                                save_dir = f'{save_dir}/certain' 
                            else:
                                save_dir = f'{save_dir}/uncertain'
                            save_dir = f'{save_dir}/{class_name}' 
                            if not os.path.exists(save_dir):
                                os.makedirs(f'{save_dir}')
                            
                            cropped_bbox = np.array(original_frame[int(top):int(bottom), int(left):int(right)])
                            cv2.imwrite(f'{save_dir}/{rois_saved}.png', cropped_bbox)
                            rois_saved += 1
                            
                        if frames_since_first_detection == -1:
                            frames_since_first_detection = 0

        if not config.no_show:
            if collage is not None:
                frame = cv2.resize(collage, (1280, 768))
            # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert it to hsv
            # import pdb; pdb.set_trace()
            # hsv[:,:,1] = 255 
            # frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # cv2.putText(frame, f'frame: {frame_id}', (10, 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.rectangle(frame, (10, 2), (140,60), (255,255,255), -1)
                cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                cv2.putText(frame, f'Bg: {bg_ts1-bg_ts0:.3f} sec.', (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                cv2.putText(frame, f'Infer: {infer_ts:.3f} sec.', (15, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                cv2.imshow('Detections', frame)

                threshold = motionDetector.current_threshold.copy()
                threshold = cv2.resize(threshold, (800, 600))
                cv2.imshow('Threshold', threshold)

                if config.debug:
                    delta = motionDetector.current_delta.copy()
                    delta = cv2.resize(delta, (800, 600))
                    cv2.imshow('Delta', delta)

                    gray = motionDetector.current_gray.copy()
                    gray = cv2.resize(gray, (800, 600))
                    cv2.imshow('Gray', gray)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    sys.exit()

        ret, frame = cap.read()
        frame_id += 1
        if frames_since_first_detection >= 0:
            frames_since_first_detection += 1
        if config.stop_after > 0 and \
            config.stop_after <= frame_id:
            break
        # or
        if config.stop_after_detection > 0 and\
                frames_since_first_detection >= config.stop_after_detection:
            break

    columns = ['frame', 'class_id', 'score', 'xmin', 'ymin', 'xmax', 'ymax', 'num_roi', 'parallel_infers']
    detections = pd.DataFrame(detections, columns=columns)
    detections.to_pickle(f'{config.output}/{Path(config.video).stem}.pkl', 'bz2')

if __name__ == "__main__":
    main()
