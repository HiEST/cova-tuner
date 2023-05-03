#!/bin/bash

MODEL_VERSION="ssd_mobilenet_v2_320x320_coco17_tpu-8"
MODEL_URL="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"

if [ ! -d source_dir/checkpoint2 ]; then
    curl $MODEL_URL --output model.tar.gz
    tar xvfz model.tar.gz
    mv $MODEL_VERSION/checkpoint source_dir/checkpoint2
fi

docker build -t edge-cova:aws -f Dockerfile ../
