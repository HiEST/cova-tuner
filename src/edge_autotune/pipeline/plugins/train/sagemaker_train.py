"""This module implements a COVATrainer to fine-tune models using TensorFlow's Object Detection API."""
import logging

from edge_autotune.pipeline.pipeline import COVATrain
from sagemaker.tensorflow import TensorFlow
from sagemaker.debugger import TensorBoardOutputConfig


logger = logging.getLogger(__name__)

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
        self.aws_config["instance_type"] = aws_config.get(
            "instance_type", "ml.g4dn.xlarge"
        )
        self.aws_config["instance_count"] = aws_config.get("instance_count", 1)

        self.train_config = train_config
        self.train_config["num_train_steps"] = train_config.get("num_train_steps", 1000)
        self.train_config["sample_1_of_n_eval_examples"] = train_config.get(
            "sample_1_of_n_eval_examples", 1
        )

        self.train_config["tensorboard_output_config"] = None
        if train_config.get("tensorboard_prefix", None) is not None:
            self.train_config["tensorboard_output_config"] = TensorBoardOutputConfig(
                s3_output_path=self.train_config["tensorboard_prefix"],
                container_local_output_path="/opt/training/",
            )

    def train(self, dataset_path: str):
        """Start fine-tuning from base model's checkpoint."""

        hyperparameters = {
            "model_dir": "/opt/ml/model",
            "pipeline_config_path": "pipeline.config",
            "checkpoint_dir": "checkpoint/",
            "num_train_steps": self.train_config["num_train_steps"],
            "sample_1_of_n_eval_examples": self.train_config[
                "sample_1_of_n_eval_examples"
            ],
        }

        estimator = TensorFlow(
            # entry_point="train.py",
            entry_point="run_training.sh",
            role=self.aws_config["role"],
            instance_count=1,
            instance_type=self.aws_config["instance_type"],
            source_dir=self.train_config["source_dir"],
            output_path=self.train_config["output_prefix"],
            image_uri=self.aws_config["ecr_image"],
            hyperparameters=hyperparameters,
            tensorboard_output_config=self.train_config["tensorboard_output_config"],
            disable_profiler=True,
            base_job_name='tf2-object-detection',
        )

        # train_channel = os.path.join(dataset_path, 'train.record')
        # eval_channel = os.path.join(dataset_path, 'eval.record')
        # TODO: We make sure to specify wait=False, so our notebook is not waiting for the training job to finish.
        inputs = {"train": dataset_path}
        estimator.fit(inputs)
        
        job_artifacts_path = estimator.latest_job_tensorboard_artifacts_path()
        logger.info('Tensorboard artifacts path: %s', job_artifacts_path)
