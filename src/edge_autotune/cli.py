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

import click

from edge_autotune.cli_helper import _server, _capture, _tune


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

input_server_port_option = click.option(
  '-p', '--port',
  help='Port to connect to',
  default=6000,
  type=int
)

input_disable_motion_option = click.option(
  '--disable-motion',
  help='Disable the use of motion detetion for the region proposal during annotation',
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

input_max_images_option = click.option(
  '--max-images',
  help='Stop when the number of images captured and annotated reaches the maximum.',
  default=1000,
  type=int,
)

input_min_images_option = click.option(
  '-p', '--port',
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
@input_server_url_option
@input_server_port_option
@input_valid_classes_option
@input_disable_motion_option
@output_dataset_option
@input_min_score_option
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
    min_area=1000,
  )


input_checkpoint_option = click.option(
  '-c', '--checkpoint',
  help='Path to the checkpoint directory',
  required=True,
  type=click.Path(exists=True, dir_okay=True, file_okay=False),
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


@input_checkpoint_option
@input_dataset_option
@input_config_option
@output_dir_option
@main.command()
def tune(
  checkpoint: str,
  dataset: str,
  config: str,
  output: str,
  train_steps: int = 1000,
):
  """Start fine-tuning from base model's checkpoint.

  Args:
    checkpoint (str): Path to directory containing the checkpoint to use as base model.
    dataset (str): Path to the training dataset TFRecord file.
    config (str): Path to the pipeline.config file with the training config.
    output (str): Path to the output directory.
    train_steps (int, optional): Number of training steps. Defaults to 1000.
  """
  _tune(
    checkpoint=checkpoint,
    dataset=dataset,
    config=config,
    output=output,
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


@input_checkpoint_option
@input_dataset_option
@output_dir_option
@main.command()
def deploy(
  stream: str,
  dataset: str,
  port: int,
):
  """Start client for inference using the tuned model."""
  print('app')

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
