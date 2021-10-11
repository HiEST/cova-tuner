"""This module implements a COVATrainer to fine-tune models using TensorFlow's Object Detection API."""

from edge_autotune.pipeline.pipeline import COVATrain
from edge_autotune.dnn import train


class ObjectDetectionAPI(COVATrain):
    def __init__(self):
        """Inits an ObjectDetectionAPI object. """
        pass

    def train(self, checkpoint: str, dataset: str, config: str,
             output_dir: str, label_map: str, train_steps: int = 1000):
        """Start fine-tuning from base model's checkpoint.
        Args:
            checkpoint (str): Path to directory containing the checkpoint to use as base model.
            dataset (str): Path to the training dataset TFRecord file.
            config (str): Path to the pipeline.config file with the training config.
            output (str): Path to the output directory.
            train_steps (int, optional): Number of training steps. Defaults to 1000.
        """

        train_datasets = dataset.split(',')
        train.train_loop_wrapper(
            pipeline_config=config,
            train_datasets=train_datasets,
            model_dir=output_dir,
            base_model=checkpoint,
            label_map=label_map,
            num_train_steps=train_steps
        )

        train.export_trained_model(
            pipeline_config_path=config,
            trained_checkpoint_dir=output_dir,
            output_dir=f'{output_dir}/saved_model'
        )

    def epilogue(self) -> None:
        pass