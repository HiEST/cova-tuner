<h1 align="center">
  Edge AutoTuner
</h1>

  <a href='https://opensource.org/licenses/Apache-2.0'>
    <img src='https://img.shields.io/badge/License-Apache%202.0-blue.svg' alt='License'/>
  </a>

  <a href="https://zenodo.org/badge/latestdoi/267315762">
    <img src="https://zenodo.org/badge/267315762.svg" alt="DOI">
  </a>

</p>

<p align="center">
    <b>Edge AutoTuner</b> is a framework that automates fine-tuning of CNN models deployed to the edge for video analytics -- i.e. image recognition and object detection.
</p>

<p align="center">
  <a href="#requirements">Requirements</a> •
  <a href="#installation">Installation</a> •
  <a href="#configuration">Configuration</a>
</p>

### Citation
If you use Edge AutoTuner for your research please cite our [preprint](https://www.arxiv.org/to-be-submitted): 

> Daniel Rivas-Barragan, Francesc Guim-Bernat, Josep Ll. Berral, Jordà Polo, and David Carrera (2021).
Edge AutoTuner: Automated Fine-Tuning for Continuous Video Analytics. *arXiv* 2020.tbd; https://doi.org/tbd


## Requirements:
- Python >=3.6
- Download edge model's checkpoint and reference model from Tensorflow Model Zoo:
> In our experiments, we have used MobileNet V2 320x320 as the edge model and F-RCNN Inception Resnet152 1024x1024 as the reference model.

```wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz```

```wget http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz```


## Installation

<p align="center">
  <a href="https://drug2ways.readthedocs.io/en/latest/">
    <img src="http://readthedocs.org/projects/drug2ways/badge/?version=latest"
         alt="Documentation">
  </a>

  <img src='https://img.shields.io/pypi/pyversions/drug2ways.svg' alt='Stable Supported Python Versions'/>
  
  <a href="https://pypi.python.org/pypi/drug2ways">
    <img src="https://img.shields.io/pypi/pyversions/drug2ways.svg"
         alt="PyPi">
  </a>
</p>

The latest stable code can be installed from [EdgeAutoTuner](https://pypi.python.org/pypi/edge_autotuner) with:

```python
python -m pip install edge-autotuner
```

The most recent code can be installed from the source on [GitHub](https://github.com/danirivas/edge_autotuner) with:

```python
python -m pip install git+https://github.com/danirivas/edge_autotuner.git
```

For developers, the repository can be cloned from [GitHub](https://github.com/HiEST/edge_autotuner) and installed in
editable mode with:

```python
git clone https://github.com/HiEST/edge_autotuner.git
cd edge_autotuner
python -m pip install -e .
```

1. Install Python requirements:
```pip install -r requirements.txt```

2. Clone repository for VOC metrics:
```git clone https://github.com/rafaelpadilla/Object-Detection-Metrics```


## Configuration
Under `config/train_config.ini` you'll find an example of the config file with all the accepted options.

Before you can start EAT, you'll have to set some paths in the config file. 

### Setup paths
`root_dir`: Absolute path to the root directory of the project.

`pascalvoc_dir`: Path to the root of the cloned `Object-Detection-Metrics` repository.

`workload_dir`: relative path to `training/detection`. The remaining paths will be relative to this one.  
