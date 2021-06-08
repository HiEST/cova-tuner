#!/bin/bash

EDGE_MODEL="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
GT_MODEL="http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz"

curl $EDGE_MODEL -O .
curl $GT_MODEL -O .