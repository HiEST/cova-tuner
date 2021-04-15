<h1 align="center">
  <br>
  <img src="https://user-images.githubusercontent.com/11491836/114053034-dcac7600-988e-11eb-9227-27c92d1114b7.png" alt="Edge AutoTune" width="600">
</h1>

  <a href='https://opensource.org/licenses/Apache-2.0'>
    <img src='https://img.shields.io/badge/License-Apache%202.0-blue.svg' alt='License'/>
  </a>

  <a href="https://zenodo.org/badge/latestdoi/placeholder">
    <img src="https://zenodo.org/badge/267315762.svg" alt="DOI">
  </a>

</p>

<p align="center">
    <b>Edge AutoTune</b> automates the optimization of convolutional neural networks deployed to the edge for video analytics.
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#basic-usage">How To Use</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-does-it-work">How Does It Work?</a>
</p>

## Quickstart
Edge AutoTune provides a series of tools aimed at assisting with a rapid deployment of CNN models for video analytics in edge cloud locations. It automates every step of the pipeline, from the creation of the dataset using images from the edge cameras, the _tuning_ of a generic model, all the way to the deployment of the _specialized_ model.

Edge AutoTune takes one _edge_ model, one _groundtruth_ model, and a video stream as inputs and generates and _deploys_ a fine-tuned version of the _edge_ model specifically optimized for the specifics of the edge camera.

Edge AutoTune brings a series of techniques together that work best when used on images from static cameras. Moreover, it assumes that the _groundtruth_ model has been trained on the classes we want the _edge_ model to detect, although more than one _groundtruth_ model can be used for this purpose.

### Citation
If you use Edge AutoTuner for your research please cite our [preprint](http://arxiv.org/abs/2104.06826): 

> Rivas, Daniel, Francesc Guim, Jordà Polo, Josep Ll Berral, Pubudu M. Silva, and David Carrera. “Towards Unsupervised Fine-Tuning for Edge Video Analytics.” ArXiv:2104.06826 [Cs, Eess], April 14, 2021. http://arxiv.org/abs/2104.06826.

## How To Use
Edge AutoTune provides multiple tools that are accessible from the command-line interface. 
The typical flow is start server with `server`, create training dataset with `capture`, fine-tune model with `tune`, and deploy with `deploy`.

```console
foo@bar:~$ edge_autotune --help
Usage: edge_autotune [OPTIONS] COMMAND [ARGS]...

  CLI for edge_autotune.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  server    Start annotation server.
  capture   Capture and annotate images from stream and generate dataset.
  tune      Start fine-tuning from base model's checkpoint.
  deploy    Start client for inference using the tuned model.
```

## Installation

<!--
<p align="center">
  <a href="https://drug2ways.readthedocs.io/en/latest/">
    <img src="http://readthedocs.org/projects/drug2ways/badge/?version=latest"
         alt="Documentation">
  </a>
<!--
  <img src='https://img.shields.io/pypi/pyversions/drug2ways.svg' alt='Stable Supported Python Versions'/>
<!--
  <a href="https://pypi.python.org/pypi/drug2ways">
    <img src="https://img.shields.io/pypi/pyversions/drug2ways.svg"
         alt="PyPi">
  </a>
</p>
-->

<!-- The latest stable code can be installed from [Edge AutoTune](https://pypi.python.org/pypi/edge_autotuner) with:

```python
python -m pip install edge-autotuner
```
-->
The most recent code can be installed from the source on [GitHub](https://github.com/HiEST/edgeautotuner) with:

```python
python -m pip install git+https://github.com/HiEST/edgeautotuner.git
```

For developers, the repository can be cloned from [GitHub](https://github.com/HiEST/edgeautotuner) and installed in
editable mode with:

```python
git clone https://github.com/HiEST/edgeautotuner.git
cd edgeautotuner
python -m pip install -e .
```

### Requirements:
Edge AutoTune works with Python >=3.6 and uses the following packages:
```
- tensorflow>=2.4.1
- opencv-python
- Pillow>=8.1.1
- numpy
- pandas
- flask
- flask-restful
- tqdm
```

CUDA is not a requirement to run. However, for the fine-tuning step, it is recommended to use a machine with a GPU installed and CUDA support.
 
- Download edge and groundtruth model's checkpoint from Tensorflow Model Zoo. 
There is no restriction in what models to use but in our experiments we have used the following:
>  **Edge**: [MobileNet V2 320x320](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz)

> **Reference**: [EfficientDet D2 768x768](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz)


## How does it work?
Edge AutoTune is composed of multiple modules implementing different ideas and approaches. This project is the result of an thorough exploration to optimize video analytics in the context of resource-constrained edge nodes. However, two ideas stand out from the rest and are based on the assumption of static cameras:

1. Overfitting the context (model specialization)
2. The scene does not move, but _interesting_ objects do (motion detection).

When working with static cameras, it is common for the scene to be mostly _static_ (sic.). The first implication of this is the realization that new objects do not enter the scene _every frame_. If nothing new enters the scene, we can safely assume that there is nothing new to be detected. Therefore, inferences (either by the edge model while deployed or by the groundtruth model during annotation) are halted if no new objects are detected. This technique has been extensively used in several previous works to save computation for whenever new objects are detected. However, this only optimizes _the amount_ of work but not the quality of the work.

The key contribution of Edge AutoTune is that it takes this observation and uses it to improve the prediction results on different _key_ parts of the pipeline: annotation (groundtruth model further improves the quality of the resulting training dataset), deployment (specialized model is tuned for the specific context of the camera where it is deployed).

## Overfitting the Context (model specialization)
We define the context of a camera as the set of characteristics -- environmental, or technical -- that have a say in the composition of the scene, which will ultimately impact the model's accuracy. These are all characteristics that do not change over time or, if they do, do not change _short-term_. Otherwise, they would not be part of the context but another feature.

For example, location (images from a highway or a mall?), the type of objects seen (a camera in a Formula 1 circuit will certainly get to see more _exclusive_ cars than a camera installed at a regular toll), focal distance (how big or small are objects with respect to the background), and even height at which the camera is installed (are objects seen from above, ground-level, or below?). These are all characteristics considered to be part of a camera's context, as they all conform the composition of the scene and the way the model experiences the scene.

Deep Neural Networks have proven to be highly capable of generalizing predictions to previously unseen environments. However, there is a direct relationship between the level of generalitzation and the computational complexity of a model. Therefore, there is a trade-off to be made when choosing what model to deploy. Unfortunately, we should assume that resources in Edge Cloud locations are scarce and the compute installed in such locations is often insufficient to execute _state-of-the-art_ DNN's. 

Generalization is a _desirable_ property on deployed models for two reasons (or one implying the other). First, we can expect better _inference_ results from generalistic models, as they manage to successfully detect objects that were not seen during training. This enables the same model to be used in plenty different scenes. However, this goes beyond results, as it means that generalistic models can be deployed to multiple scenes without new training. A proper training is extremely costly. Training on new problems requires a set of curated images that are also properly annotated, which is an effort that requires time and money. Moreover, training a model from scratch requires expensive hardware and plenty of training hours.

We can break the aforementioned problems down into the following sequence of sub-problems:
- Generalization _desirable_ as it makes deployments easier but computationally expensive.
- Specialization requires new training.
- New training requires a representative and fully annotated set of training images.
- Annotation is expensive, in time and money.

Edge AutoTune is devised as a means of automating the creation of an annotated training dataset and the subsequent fine-tuning phase. By assuming a static scene, Edge AutoTune is able to apply a series of techniques that boost the quality of the training dataset, and therefore the quality of the deployed model. All this withouth increasing the cost of the resulting deployed model in the slightest.

In a nutshell, Edge AutoTune solves the previous problem as follows:
- Specialization is the goal. A _simpler_ model  can be effectively used, as the scope of the problem is narrowed down to the context of a single camera.
- Dataset creation is automated: 
    * _representativeness_ is assumed, as we only consider images from the same camera where the model will be later deploye.
    * _annotation_ is obtained through a form of _active learning_ in which the oracle is replaced by a _state-of-the-art_ DNN running in a hardware-accelerated server node.
    * Both _representativeness_ and _annotation_ are by assuming static cameras (which allows us to simple motion detection techniques [details](#motion-detection)).


### Motion Detection
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


