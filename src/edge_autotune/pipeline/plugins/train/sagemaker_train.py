"""This module implements a COVATrainer to fine-tune models using TensorFlow's Object Detection API."""

from edge_autotune.pipeline.pipeline import COVATrain
from sagemaker.tensorflow import TensorFlow
from sagemaker.debugger import TensorBoardOutputConfig


class SageMakerTrain(COVATrain):
    """Class implenting COVATrain using SageMaker from AWS"""

    def __init__(self, aws_config: dict, train_config: dict):
        """Constructs a SageMakerTrain object

        Args:
            aws_config (dict):
                dictionary containing all necessary information to connect to and use SageMaker from AWS.
            train_config (float):
                dictionary containing the training's configuration.
        """

        self.aws_config = aws_config
        self.aws_config['instance_type'] = aws_config.get('instance_type', "ml.g4dn.xlarge")
        self.aws_config['instance_count'] = aws_config.get('instance_count', 1)

        self.train_config = train_config
        self.train_config['num_train_steps'] = train_config.get('num_train_steps', 1000)
        self.train_config['sample_1_of_n_eval_examples'] = train_config.get('sample_1_of_n_eval_examples', 1)
        
        self.train_config['tensorboard_output_config'] = None
        if train_config.get('tensorboard_prefix', None) is not None:       
            self.train_config['tensorboard_output_config'] = TensorBoardOutputConfig(
                s3_output_path=self.train_config['tensorboard_prefix'],
            )

    def train(self, dataset_path: str):
        """Start fine-tuning from base model's checkpoint."""

        tensorboard_output_config = TensorBoardOutputConfig(
            s3_output_path=self.train_config['tensorboard_prefix'],
        )

        hyperparameters = {
            "model_dir": "/opt/ml/model",
            "pipeline_config_path": "pipeline.config",
            "checkpoint_dir": "checkpoint/",
            "num_train_steps": self.train_config['num_train_steps'],
            "sample_1_of_n_eval_examples": self.train_config['sample_1_of_n_eval_examples'],
        }

        estimator = TensorFlow(
            entry_point="train.py",
            role=self.aws_config['role'],
            instance_count=1,
            instance_type=self.aws_config['instance_type'],
            source_dir="source_dir",
            output_path=self.train_config['output_prefix'],
            image_uri=self.aws_config["erc_image"],
            hyperparameters=hyperparameters,
            tensorboard_output_config=tensorboard_output_config,
        )

        # TODO: We make sure to specify wait=False, so our notebook is not waiting for the training job to finish.
        inputs = {"train": dataset_path}
        estimator.fit(inputs)
