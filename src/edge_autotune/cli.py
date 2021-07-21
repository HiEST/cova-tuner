# -*- coding: utf-8 -*-

"""Command line interface for :mod:`edge_autotune`.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m edge_autotune`` python will execute``__main__.py`` as a script.
  That means there won't be any ``edge_autotune.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``edge_autotune.__main__`` in ``sys.modules``.

.. seealso:: https://click.palletsprojects.com/en/7.x/setuptools/#setuptools-integration
"""

import logging
from typing import Tuple

import click

if not '--help' in click.get_os_args():
  from edge_autotune.cli_helper import _server, _capture, _capture_aws, _tune, _deploy, _deploy_multi_cam


__all__ = ['main']

logger = logging.getLogger(__name__)

@click.group()
@click.version_option()
def main():
    """CLI for edge_autotune."""


input_model_option = click.option(
  '-m', '--model',
  help='Path to the model',
  type=click.Path(exists=True, dir_okay=True, file_okay=False),
)

input_model_id_option = click.option(
  '--model-id',
  help='ID for the model to load',
  type=str,
)

input_port_option = click.option(
  '-p', '--port',
  help='Port to listen to',
  default=6000,
  type=int
)

@input_model_option
@input_model_id_option
@input_port_option
@main.command()
def server(
  model: str,
  model_id: str,
  port: int,
):
  """Start annotation server. """
  _server(model, model_id, port)


input_stream_option = click.option(
  '-s', '--stream',
  help='Path or url to the input stream to capture images.',
  type=click.Path(exists=True, dir_okay=False, file_okay=True),
)

input_stream_list_option = click.option(
  '-s', '--stream-list',
  help='Path or url to the input stream to capture images.',
  multiple=True,
  default=[]
  # type=click.Path(exists=True, dir_okay=False, file_okay=True),
)

input_bucket_option = click.option(
  '--bucket',
  help='AWS S3 Bucket Name.',
  type=str,
)

input_key_option = click.option(
  '--key',
  help='Key prefix for files stored in S3 bucket.',
  type=str,
)

input_valid_classes_option = click.option(
  '--classes',
  help='Comma-separated list of classes to detect.',
  type=str,
)

input_server_url_option = click.option(
  '--server',
  help='Server\'s url',
  type=str,
)

input_disable_motion_option = click.option(
  '--disable-motion',
  help='Disable the use of motion detetion for the region proposal during annotation',
  default=False,
  is_flag=True,
)

input_debug_option = click.option(
  '--debug',
  help='Activate debug',
  default=False,
  is_flag=True,
)

output_dataset_option = click.option(
  '-o', '--output',
  help='Path to the output dataset file',
  required=True,
  type=click.Path(exists=False, dir_okay=False, file_okay=True),
)

input_min_score_option = click.option(
  '--min-score',
  help='Minimum score to accept groundtruth model\'s predictions as valid.',
  default=0.5,
  type=float,
)

input_min_area_option = click.option(
  '--min-area',
  help='Minimum area to consider a region as in containing motion.',
  default=1000,
  type=int,
)

input_max_images_option = click.option(
  '--max-images',
  help='Stop when the number of images captured and annotated reaches the maximum.',
  default=1000,
  type=int,
)

input_min_images_option = click.option(
  '--min-images',
  help='Prevents timeout to stop execution if the minimum of images has not been reached.'
      'Ignored if timeout is 0. Defaults to 0.',
  default=0,
  type=int,
)

input_timeout_option = click.option(
  '--timeout',
  help='Timeout to stop execution even if maximum images has not been reached.',
  default=0,
  type=int,
)


@input_stream_option
@input_bucket_option
@input_key_option
@input_valid_classes_option
@input_disable_motion_option
@input_min_score_option
@input_min_area_option
@input_max_images_option
@input_min_images_option
@input_timeout_option
@main.command()
def capture_aws(
  stream: str,
  bucket: str,
  key: str,
  classes: str = None,
  disable_motion: bool = False,
  min_score: float = 0.5,
  min_area: int = 1000,
  max_images: int = 1000,
  min_images: int = 100,
  timeout: int = 0,
):
  """Capture and annotate images from stream and generate dataset."""
  print(f'capture-aws: from {stream} to s3://{bucket}/{key}.')
  _capture_aws(
    stream=stream,
    bucket=bucket,
    key_prefix=key,
    valid_classes=classes,
    disable_motion=disable_motion,
    min_score=min_score,
    max_images=max_images,
    min_images=min_images,
    min_area=min_area,
  )


input_server_port_option = click.option(
  '-p', '--port',
  help='Port to connect to',
  default=6000,
  type=int
)


@input_stream_option
@input_server_url_option
@input_server_port_option
@input_valid_classes_option
@input_disable_motion_option
@output_dataset_option
@input_min_score_option
@input_min_area_option
@input_max_images_option
@input_min_images_option
@input_timeout_option
@main.command()
def capture(
  stream: str,
  output: str,
  server: str,
  port: int = 6000,
  classes: str = None,
  disable_motion: bool = False,
  min_score: float = 0.5,
  min_area: int = 1000,
  max_images: int = 1000,
  min_images: int = 100,
  timeout: int = 0,
):
  """Capture and annotate images from stream and generate dataset."""
  print(f'capture: from {stream} to {output}. Connect to {server} (port={port}). Motion? {not disable_motion}')
  _capture(
    stream=stream,
    output=output,
    server=server,
    port=port,
    valid_classes=classes,
    disable_motion=disable_motion,
    min_score=min_score,
    max_images=max_images,
    min_images=min_images,
    min_area=min_area,
  )


input_checkpoint_option = click.option(
  '-c', '--checkpoint',
  help='Path to the checkpoint directory',
  required=True,
  type=str,
)

input_dataset_option = click.option(
  '-d', '--dataset',
  help='Path to the training dataset (tfrecord)',
  required=True,
  type=click.Path(exists=True, dir_okay=False, file_okay=True),
)

input_config_option = click.option(
  '--config',
  help='Path to the pipeline.config file',
  required=True,
  type=click.Path(exists=True, dir_okay=False, file_okay=True),
)

output_dir_option = click.option(
  '-o', '--output',
  help='Path to the output dir',
  required=True,
  type=click.Path(exists=False, dir_okay=True, file_okay=False),
)

input_train_steps_option = click.option(
  '--train-steps',
  help='Number of train steps',
  type=int,
)

input_label_map_option = click.option(
  '-l', '--label-map',
  help='Path to the training dataset (tfrecord)',
  required=True,
  type=click.Path(exists=True, dir_okay=False, file_okay=True),
)


@input_checkpoint_option
@input_dataset_option
@input_config_option
@output_dir_option
@input_label_map_option
@input_train_steps_option
@main.command()
def tune(
  checkpoint: str,
  dataset: str,
  config: str,
  output: str,
  label_map: str,
  train_steps: int = 1000,
):
  """Start fine-tuning from base model's checkpoint.

  Args:
    checkpoint (str): Path to directory containing the checkpoint to use as base model.
    dataset (str): Path to the training dataset TFRecord file.
    config (str): Path to the pipeline.config file with the training config.
    output (str): Path to the output directory.
    label_map (str): Path to the pbtxt label_map file.
    train_steps (int, optional): Number of training steps. Defaults to 1000.
  """
  _tune(
    checkpoint=checkpoint,
    dataset=dataset,
    config=config,
    output=output,
    label_map=label_map,
    train_steps=train_steps,
  )


@input_checkpoint_option
@input_dataset_option
@output_dir_option
@main.command()
def autotune(
  checkpoint: str,
  dataset: str,
  port: int,
):
  """Start capture, annotation, and fine-tuning."""
  print('autotune')


input_window_size_option = click.option(
  '--window-size',
  help='Size of the output window to show the stream',
  type=(int,int),
  default=(1280,720),
)

input_roi_size_option = click.option(
  '--roi-size',
  help='Minimum size of the proposed regions of interest.',
  type=(int,int),
  default=(1,1),
)

input_label_map_deploy_option = click.option(
  '-l', '--label-map',
  help='Path to the training dataset (tfrecord)',
  type=str,
  default=None,
)

input_save_to_option = click.option(
  '--save-to',
  help='Path to save video with detection results',
  type=str,
  default=None,
)

input_frame_skip_option = click.option(
  '--frame-skip',
  help='Frame skipping value',
  default=1,
  type=int,
)


@input_stream_option
@input_model_option
@input_label_map_deploy_option
@input_valid_classes_option
@input_min_score_option
@input_disable_motion_option
@input_min_area_option
@input_window_size_option
@input_save_to_option
@input_debug_option
@input_frame_skip_option
@main.command()
def deploy(
  stream: str,
  model: str,
  label_map: str = None,
  classes: str = None,
  min_score: float = 0.5,
  disable_motion: bool = False,
  min_area: int = 1000,
  window_size: Tuple[int,int] = [1280,720],
  save_to: str = None,
  debug: bool = False,
  frame_skip: int = 1,
):
  """Start client for inference using the tuned model."""
  _deploy(
    stream=stream,
    model=model,
    label_map=label_map,
    valid_classes=classes,
    min_score=min_score,
    disable_motion=disable_motion,
    min_area=min_area,
    first_frame_background=True,
    window_size=window_size,
    save_to=save_to,
    debug=debug,
    frame_skip=frame_skip,
  )


@input_stream_list_option
@input_model_option
@input_label_map_deploy_option
@input_valid_classes_option
@input_min_score_option
@input_disable_motion_option
@input_min_area_option
@input_roi_size_option
@input_window_size_option
@input_save_to_option
@input_debug_option
@input_frame_skip_option
@main.command()
def deploy_multi_cam(
  stream_list: str,
  model: str,
  label_map: str = None,
  classes: str = None,
  min_score: float = 0.5,
  disable_motion: bool = False,
  min_area: int = 1000,
  roi_size: Tuple[int,int] = (1,1),
  window_size: Tuple[int,int] = [1280,720],
  save_to: str = None,
  debug: bool = False,
  frame_skip: int = 1,
):
  """Start client for inference using the tuned model."""
  _deploy_multi_cam(
    streams=stream_list,
    model=model,
    label_map=label_map,
    valid_classes=classes,
    min_score=min_score,
    disable_motion=disable_motion,
    min_area=min_area,
    roi_size=roi_size,
    first_frame_background=True,
    window_size=window_size,
    save_to=save_to,
    debug=debug,
    frame_skip=frame_skip,
  )

if __name__ == '__main__':
    main()



# @input_model_option
# @input_port_option
# @main.command()
# def app(
#   model: str,
#   port: int,
# ):
#   """Runs given app only. 

#   Args:
#       model (str): Path to dir containing saved_model or checkpoint from TensorFlow.
#       port (int): Port to listen to.
#   """
#   print('app')


# if __name__ == '__main__':
#     main()
