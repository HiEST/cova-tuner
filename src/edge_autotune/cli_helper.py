# -*- coding: utf-8 -*-

import logging

from edge_autotune.api import server

logger = logging.getLogger(__name__)


def _server(
  model: str,
  model_id: str,
  port: int = 6000,
):
    """Start annotation server. 

    Args:
        model (str): [description]
        model_id (str, optional): Path to dir containing saved_model or checkpoint from TensorFlow. Defaults to ''.
        port (int, optional): Port to listen to clients. Defaults to 6000.
    """
    if not model_id:
        model_id = 'default'
    print(f'server.start_server({model}, {model_id}, {port})')
    server.start_server(model, model_id, port)


def _capture(
  stream: str,
  output: str,
  server: str,
  port: int = 6000,
  disable_motion: bool = False,
  min_score: float = 0.5,
  max_images: int = 1000,
  min_images: int = 100,
  timeout: int = 0,
):
  """Capture and annotate images from stream and generate dataset.

  Args:
      stream (str): Input stream from which to capture images.
      output (str): Path to the output dataset. 
      server (str): Server's url.
      port (int, optional): Port to connect to the server. Defaults to 6000.
      disable_motion (bool, optional): Disable motion detection for the region proposal. Defaults to False.
      min_score (float, optional): Minimum score to accept groundtruth model's predictions as valid. Defaults to 0.5.
      max_images (int, optional): Stop when maximum is reached. Defaults to 1000.
      min_images (int, optional): Prevents timeout to stop execution if the minimum of images has not been reached. 
      Used only if timeout > 0. Defaults to 0.
      timeout (int, optional): [description]. Defaults to 0.
  """
  print(f'capture: from {stream} to {output}. Connect to {server} (port={port}). Motion? {not disable_motion}')


def _tune(
  checkpoint: str,
  dataset: str,
  config: str,
  output: str,
):
  """Start fine-tuning of model using specified dataset and pipeline config. 

  Args:
      model (str): Path to dir containing saved_model or checkpoint from TensorFlow.
      port (int): Port to listen to.
  """
  print('app')


def _autotune(
  checkpoint: str,
  dataset: str,
  port: int,
):
  """Start capture, annotation, and fine-tuning. 

  Args:
      model (str): Path to dir containing saved_model or checkpoint from TensorFlow.
      port (int): Port to listen to.
  """
  print('app')


def _deploy(
  stream: str,
  dataset: str,
  port: int,
):
  """Start inference on stream. 

  Args:
      stream (str): Input video stream (file or url).
      model (str): Path to dir containing saved_model or checkpoint from TensorFlow.
      port (int): Port to listen to.
  """
  print('app')

