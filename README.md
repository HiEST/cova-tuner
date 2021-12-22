<h1 align="center">
  <br>
  <!--
    <img src="https://user-images.githubusercontent.com/11491836/114053034-dcac7600-988e-11eb-9227-27c92d1114b7.png" alt="Edge AutoTune" width="600">
  -->
    <img src="https://user-images.githubusercontent.com/11491836/147062336-a44da7cf-8085-4247-8e6c-08ddf18dada6.png" alt="COVA" width="600">
</h1>


  <a href='https://opensource.org/licenses/Apache-2.0'>
    <img src='https://img.shields.io/badge/License-Apache%202.0-blue.svg' alt='License'/>
  </a>

  <a href="https://zenodo.org/badge/latestdoi/placeholder">
    <img src="https://zenodo.org/badge/267315762.svg" alt="DOI">
  </a>

</p>

<p align="center">
    <b>COVA</b> automates the optimization of convolutional neural networks deployed to the edge for video analytics.
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#basic-usage">Basic Usage</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-does-it-work">How Does It Work?</a> •
  <a href="#the-cova-pipeline">The COVA pipeline</a>
</p>

# Quickstart
COVA (Contextually Optimized Video Analytics) is a framework aimed at automating the process of model specialization for edge video analytics. It provides a series of structures and tools to assist in the rapid deployment of CNN models for video analytics in edge cloud locations. It automates every step of the pipeline, from the creation of the dataset using images from the edge cameras, the _tuning_ of a generic model, all the way to the deployment of the _specialized_ model.

To start, COVA requires three key components: one _edge_ model (i.e., the model to specialize), one _groundtruth_ model (i.e., the model whose knowledge will be distilled into the _edge_ model), and a video stream from a fixed camera (either from a live camera or previously recorded footage). Then, COVA generates a training dataset using the input stream and the annotations proposed by the _groundtruth_ model. Finally, the _edge_ model is trained using the generated dataset. Consequently, the resulting _edge_ model is specialized for the specifics of the edge camera. 

Keep in mind that COVA assumes that the _groundtruth_ model has been trained on the classes we want the _edge_ model to detect, although more than one _groundtruth_ model can be used for this purpose. That is, COVA does not generate new knowledge but _distills_ part of the _groundtruth_'s model knowledge into the specialized _edge_ model.

COVA implements a series of techniques together that work best when used on images from static cameras. Nonetheless, the pipeline is fully customizable and each step can be extended independently. For more detailed information, please read <a href="#documentation">Documentation</a> section.

## Basic Usage
COVA executes a pipeline defined in a _json_ config file. A working example can be found under `examples/config.template.json`.

To start specializing with COVA, simply pass the path to the configuration file with the following command:

```console
foo@bar:~$ edge_autotune config.json
```


To show the command-line help:
```console
foo@bar:~$ edge_autotune --help
usage: edge_autotune [-h] config

This program runs a COVA pipeline defined in a json-like config file.

positional arguments:
  config      Path to the configuration file.

optional arguments:
  -h, --help  show this help message and exit
```

# Installation

The most recent code can be installed from the source on [GitHub](https://github.com/HiEST/cova-tuner) with:

```python
python -m pip install git+https://github.com/HiEST/cova-tuner.git
```

For developers, the repository can be cloned from [GitHub](https://github.com/HiEST/cova-tuner) and installed in
editable mode with:

```python
git clone https://github.com/HiEST/cova-tuner.git
cd cova-tuner
python -m pip install -e .
```

## Requirements:
COVA works with Python >=3.6 and uses the following packages:
```
- tensorflow>=2.4.1
- opencv-python
- Pillow>=8.1.1
- numpy
- pandas
- flask
- flask-restful
```

CUDA is not a requirement to run. However, for the fine-tuning step, it is recommended to use a machine with a GPU installed and CUDA support.
 
- Download edge and groundtruth model's checkpoint from Tensorflow Model Zoo. 
There is no restriction in what models to use. However, we have used the following models in our experiments:
>  **Edge**: [MobileNet V2 320x320](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz)

> **Reference**: [EfficientDet D2 768x768](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz)



# How does it work?
COVA is composed of multiple modules implementing different ideas and approaches. This project is the result of an thorough exploration to optimize video analytics in the context of resource-constrained edge nodes. However, two ideas stand out from the rest and are based on the assumption of static cameras:

1. Overfitting the context (model specialization)
2. The scene does not move, but _interesting_ objects do (motion detection).

When working with static cameras, it is common for the scene to be mostly _static_ (sic.). The first implication of this is the realization that new objects do not enter the scene _every frame_. If nothing new enters the scene, we can safely assume that there is nothing new to be detected. Therefore, inferences (either by the edge model while deployed or by the groundtruth model during annotation) are halted if no new objects are detected. This technique has been extensively used in several previous works to save computation for whenever new objects are detected. However, this only optimizes _the amount_ of work but not the quality of the work.

The key contribution of COVA is that it takes this observation and uses it to improve the prediction results on different _key_ parts of the pipeline: annotation (groundtruth model further improves the quality of the resulting training dataset), deployment (specialized model is tuned for the specific context of the camera where it is deployed).

## Overfitting the Context (model specialization)
We define the context of a camera as the set of characteristics (environmental, or technical) that have a say in the composition of the scene and will ultimately impact the model's accuracy. These are all characteristics that do not change over time or, if they do, do not change _short-term_. Otherwise, they would not be part of the context but another feature.

For example, location (images from a highway will have different composition than images captured at a mall), the type of objects seen (a camera in a Formula 1 circuit will certainly get to see more _exclusive_ cars than a camera installed at a regular toll), focal distance (how big or small objects are with respect to the background), and even height at which the camera is installed (are objects seen from above, ground-level, or below?). These are all characteristics considered to be part of a camera's context, as they all conform the composition of the scene and the way the model experiences the scene.

Deep Neural Networks have proven to be highly capable of generalizing predictions to previously unseen environments. However, there is a direct relationship between the level of generalitzation and the computational complexity of a model. Therefore, there is a trade-off to be made when choosing what model to deploy. Unfortunately, we should assume that resources in Edge Cloud locations are scarce and the compute installed in such locations is often insufficient to execute _state-of-the-art_ DNN's. 

Generalization is a _desirable_ property on deployed models for two reasons (or one implying the other). First, we can expect better _inference_ results from generalistic models, as they manage to successfully detect objects that were not seen during training. This enables the same model to be used in plenty different scenes. However, this goes beyond results, as it means that generalistic models can be deployed to multiple scenes without new training. A proper training is extremely costly. Training on new problems requires a set of curated images that are also properly annotated, which is an effort that requires time and money. Moreover, training a model from scratch requires expensive hardware and plenty of training hours.

We can break the aforementioned problems down into the following sequence of sub-problems:
- Generalization _desirable_ as it makes deployments easier but computationally expensive.
- Specialization requires new training.
- New training requires a representative and fully annotated set of training images.
- Annotation is expensive, in time and money.

COVA is devised as a means of automating the creation of an annotated training dataset and the subsequent fine-tuning phase. By assuming a static scene, COVA applies a series of techniques that boost the quality of the training dataset, and therefore the quality of the deployed model. All this withouth increasing the cost of the resulting deployed model in the slightest.

In a nutshell, COVA solves the previous problem as follows:
- Specialization is the goal. A _simpler_ model  can be effectively used, as the scope of the problem is narrowed down to the context of a single camera.
- Dataset creation is automated: 
    * _representativeness_ is assumed, as we only consider images from the same camera where the model will be later deploye.
    * _annotation_ is obtained through a form of _active learning_ in which the oracle is replaced by a _state-of-the-art_ DNN running in a hardware-accelerated server node.
    * Both _representativeness_ and _annotation_ are by assuming static cameras (which allows us to simple motion detection techniques [details](#motion-detection)).


## Motion Detection
Static cameras are usually placed meters away from the objects of interest with the intention of capturing a wide area. Therefore, objects appear small. This has two implications: first, objects of interest occupy a small part of the scene, which means that most of the frame will have little to no interest but still will be processed by the neural network hoping to find something (only increasing chances of False Positives). Second, smaller objects are more difficult to be correctly detected by neural networks. Furthermore, FullHD (1920x1080 pixels) or even 4K resolutions (3840x2160 pixels) are already common for edge cameras, while 300x300 is a common input resolution used by edge models. Therefore, image gets compressed 23 to 46 times before being passed to the neural network. With it, smaller objects are at risk of becoming just a few indistinguishable pixels.  

Thanks to region proposal based on motion detection, we are able to focus the attention of the groundtruth model on those parts of the frame that really matter.
 
After we obtain a background for the scene, we can easily compute the difference between the background and the latest decoded frame to detect motion (left), previously converted to grayscale. Then, we translate the image to grayscale. Then, we compute the delta   
<table>
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/11491836/113740606-249d9280-9701-11eb-937a-185f0372edf0.gif"  alt="Delta" width = 640px height = 360px>
    </td>
    <td> 
      <img src="https://user-images.githubusercontent.com/11491836/113740580-1cddee00-9701-11eb-89b7-7c89d1bc6886.gif" alt="Threshold" width = 640px height = 360px>
    </td>
  </tr>
</table>  

As a result, we obtain the following regions of interest:
<table>
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/11491836/113740205-ca043680-9700-11eb-9e68-8261e980cc64.gif"  alt="Region proposal based on motion detection" width = 640px height = 360px>
    </td>
  </tr>
</table>
  
Objects detected by an _off-the-shelf_ edge model. Left: the input of the neural network is the bounding box containing all detected moving objects. Right: traditional approach, i.e. full frame is passed as input to the neural network.
<table>
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/11491836/113740014-9de8b580-9700-11eb-87d4-cc4bd4d6703f.gif"  alt="Detections based on motion" width = 640px height = 360px>
    </td>
    <td> 
      <img src="https://user-images.githubusercontent.com/11491836/113850591-ec4a9280-979a-11eb-9e23-54c46a1a37ed.gif" alt="Traditional approach" width = 640px height = 360px>
    </td>
  </tr>
</table>

We can observe how using the regions proposed by the motion detection algorithm boosts accuracy of the model without any further training. 
For example, on a closer look at the detections of the previous scene:
<table>
  <tr>
    <th>Metric</th>
    <th>Motion (left)</th>
    <th>Full frame (right)</th>
  </th>
  <tr>
    <td>Average confidence</td>
    <td></td>
    <td></td>  
  </tr>
  <tr>
    <td>Top1 car detections (% of frames)</td>
    <td style="text-align:center">89%</td>
    <td style="text-align:center">16%</td>  
  </tr>
  <tr>
    <td>Most Top1 detected class</td>
    <td style="text-align:center">car (avg. score 76%)</td>
    <td style="text-align:center">person (avg. score 46%)</td>  
  </tr>
</table>

Moreover, we observe how a small square in the background is consitently and mistakenly detected as a car (avg. score 26%).

These numbers highlight the incapacity of the edge model, with an input size of 300x300x3, to distinguish objects that represent only a small portion of the total frame. Thanks to the region proposal using motion detection, the edge model does not only boost its confidence on its detections but also dramatically reduce its error rate.    


# The COVA pipeline

COVA works by executing a user-defined pipeline. Each stage of the pipeline is, ideally, independent from the others. Pipeline stages are implemented using a plugin architecture to makes it easy to extend or modify the default behaviour.

By default, COVA defines the following pipeline:

![cova-pipeline](https://user-images.githubusercontent.com/11491836/140907656-8d90d7a4-19ec-4b84-9119-a32687d19f61.png)

Therefore, it expects the following stages to be defined in the configuration file.

1. *Capture*. This stage is retrieves images. Usually from a video or stream but can be extended to use other methods, such as images from disk. It is expected to return the next image to process.
2. *Filter*. This stage filters the images captured. For example, perform motion detection on the captured images to filter static frames out. It is expected to return a list of images.
3. *Annotate*. This stage annotates the images filtered. It is expected to return a path to the list of annotations.
4. *Dataset*. This stage creates the dataset. By default, the dataset is generated in [_TFRecord_](https://www.tensorflow.org/tutorials/load_data/tfrecord) format. It is expected to return the path where the dataset was stored.
5. *Train*. This stage starts training using the dataset generated at the previous stage. It is expected to return the path where the artifacts of the trained model were stored. 


## Pipeline

The pipeline, as the sequence of stages, is defined in the _COVAPipeline_ class. This class can be inherited to re-defined the default pipeline. Each stage must inherit its corresponding abstract class.

The default plugins can be found in `src/edge_autotune/pipeline/plugins/`. However, the config file allows to pass the path where the plugin for a specific stage can be found.
All parameters of every plugin can be defined in the configuration file.


### Capture

#### 1. Dummy
Does nothing but return an 100x100x3 matrix filled with zeros.

#### 2. VideoCapture
Captures images using _OpenCV_'s _VideoCapture_ class.

Parameters:
- stream (_str_). Path or url of the stream from which images are captured.
- frameskip (_int_). Frames to skip between frame and frame. Defaults to 0.
- resize (_Tuple[int,int]_). Resize captured images to specific resolution. Defaults to None (images not resized).

### Filter

#### 1. Dummy
Does nothing. Always return input inside a list without any filtering.

#### 2. Filter Static
Filters static frames out. Performs motion detection on the input images. If motion is detected, returns the input image in a list. Otherwise, returns an empty list.

Parameters:
- warmup (_int_). Number of frames used to warmup the motion detector to build its background model.


### Annotate

#### 1. Dummy
Does nothing.

#### 2. AWSAnnotation
Uses _Amazon Web Services_ for the annotation process. Images are uploaded to S3 and annotated using a _BatchTransform_ job in _AWS_ _SageMaker_. 

Parameters:
- aws_config (_dict_). Contains the configuration required to use AWS' SageMaker service.
  The dictionary is expected to contain the following information:
    - _role_: *Required*.
    - _instance_type_: ml.m4.xlarge
    - _instance_count_: 1
    - _max_concurrent_transforms_: 4
    - _content_type_: image/png
    - _region_: eu-west-1
    - _model_name_: yolov3
    - _endpoint_name_: {model_name}-gt-endpoint

- s3_config (_dict_). Contains the configuration required to use S3.
  The dictionary is expected to contain the following information:
    - _bucket_: *Required*
    - _prefix_: *Required*
    - _images_prefix_: {_prefix_}/images
    - _annotations_prefix_: {_prefix_}/annotations


#### 3. FlaskAnnotator
Uses the built-in annotation API

### Dataset

#### 1. AWSDataset
Uses _AWS SageMaker_ to create the dataset in _TFRecord_ format.

Parameters:
- aws_config (_dict_). Contains the configuration required to use AWS' SageMaker service.
  The dictionary is expected to contain the following information:
    - _role_: *Required*.
    - _ecr_image_: *Required*
    - _instance_type_: ml.m4.xlarge
- s3_config (_dict_). Contains the configuration required to use S3.
  The dictionary is expected to contain the following information:
    - _bucket_: *Required*
    - _prefix_: *Required*
- dataset_config (_dict_). Contains the configuration required to use S3.
  The dictionary is expected to contain the following information:
   - _dataset_name_: *Required*


### Train

#### 1. Dummy 
Does nothing.


#### 2. SageMakerTrain
Uses _AWS SageMaker_ to train an object detection model.


#### 3. TFObjectDetectionAPI
Uses TensorFlow's Object Detection API to train a model locally.
Parameters:
- config (_dict_). Contains the configuration required to use AWS' SageMaker service.
  The dictionary is expected to contain the following information:
    - _role_: *Required*.


# Citation
If you use COVA for your research please cite our [preprint](http://arxiv.org/abs/2104.06826): 

> Rivas, Daniel, Francesc Guim, Jordà Polo, Pubudu M. Silva, Josep Ll Berral, and David Carrera. “Towards Automatic Model Specialization for Edge Video Analytics.” ArXiv:2104.06826 [Cs, Eess], December 13, 2021. http://arxiv.org/abs/2104.06826.
