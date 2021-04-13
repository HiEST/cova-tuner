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
  <a href="#description">Description</a> •
  <a href="#basic-usage">Basic Usage</a> •
  <a href="#installation">Installation</a> •

</p>

## Description
Edge AutoTune provides a series of tools aimed at assisting with a rapid deployment of CNN models for video analytics in edge cloud locations. The framework automates every step of the pipeline, from the creation of the dataset using images from the edge cameras to the deployment of the _specialized_ model. Edge AutoTune focuses makes use of a series of techniques that work best when used on images from static cameras. 
<!-- an annotated training dataset to fine-tune neural network models using images from the same camera feed where the model is planned to be deployed.-->

### Obtaining the background

### Motion detection
When working with static cameras, it is common for the scene to be mostly _static_ (sic.). From the one side, not always there are new objects entering the scene. If nothing new enters the scene, we can safely assume that there is nothing new to be detected. Therefore, inferences (either by the edge model while deployed or by the groundtruth model during annotation) are halted if no new objects are detected. On the other side, static cameras are usually placed meters away from the objects of interest with the intention of capturing a wide area. Therefore, objects appear small. This has two implications: first, objects of interest occupy a small part of the scene, which means that most of the frame will have little to no interest but still will be processed by the neural network hoping to find something (only increasing chances of False Positives). Second, smaller objects are more difficult to be correctly detected by neural networks. Furthermore, FullHD (1920x1080 pixels) or even 4K resolutions (3840x2160 pixels) are already common for edge cameras, while 300x300 is a common input resolution used by edge models. Therefore, image gets compressed 23 to 46 times before being passed to the neural network. With it, smaller objects are at risk of becoming just a few indistinguishable pixels.  

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


## Basic Usage
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


### Citation
If you use Edge AutoTuner for your research please cite our [preprint](https://www.arxiv.org/to-be-submitted): 

> Daniel Rivas-Barragan, Francesc Guim-Bernat, Jordà Polo, Josep Ll. Berral, Pubudu M. Silva, and David Carrera (2021).
Towards Unsupervised Fine-Tuning for Edge Video Analytics. *arXiv* 2020.tbd; https://doi.org/tbd


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
- tensorflow>=2.4.1
- opencv-python
- Pillow>=8.1.1
- numpy
- pandas
- flask
- flask-restful
- tqdm

CUDA is not a requirement to run. However, for the fine-tuning step, it is recommended to use a machine with a GPU installed and CUDA support.
 
- Download edge and groundtruth model's checkpoint from Tensorflow Model Zoo. 
There is no restriction in what models to use but in our experiments we have used the following:
>  **Edge**: [MobileNet V2 320x320](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz)

> **Reference**: [EfficientDet D2 768x768](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz)

