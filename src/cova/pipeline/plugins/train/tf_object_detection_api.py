"""This module implements a COVATrainer to fine-tune models using TensorFlow's Object Detection API."""

from cova.pipeline.pipeline import COVATrain
from cova.dnn import train


class TFObjectDetectionAPI(COVATrain):
    """Class implenting COVATrain using TensorFlow's Object Detection API"""

    def __init__(self, config: dict):
        """Constructs a TFObjectDetectionAPI object

        Args:
            config (dict):
                dictionary containing the training's configuration.
        """

        self.config = config


    def train(self):
        """Start fine-tuning from base model's checkpoint."""

        train_datasets = self.config['dataset'].split(',')
        train.train_loop_wrapper(
            pipeline_config=self.config['config'],
            train_datasets=train_datasets,
            model_dir=self.config['output_dir'],
            base_model=self.config['checkpoint'],
            label_map=self.config['label_map'],
            num_train_steps=self.config['train_steps']
        )

        train.export_trained_model(
            pipeline_config_path=self.config['config'],
            trained_checkpoint_dir=self.config['output_dir'],
            output_dir=f"{self.config['output_dir']}/saved_model"
        )