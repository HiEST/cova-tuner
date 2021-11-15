"""This module implements generation of TFRecord dataset to be used by AWS Sagemaker."""

from datetime import datetime
import io
import logging
import json
import os
from pathlib import Path
import time

import boto3
import sagemaker
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput


from edge_autotune.pipeline.pipeline import COVADataset


logger = logging.getLogger(__name__)


class AWSDataset(COVADataset):
    """A class implementing COVADataset to generate a dataset to be used in a COVAPipeline."""

    def __init__(self, aws_config: dict, s3_config: dict, dataset_config: dict):
        """Constructs an AWSDataset object

        Args:
            aws_config (dict):
                dictionary containing all necessary information to connect to and use AWS services.
            s3_config (dict):
                dictionary containing all necessary information to connect to and use S3 storage.
            dataset_config (float):
                dictionary containing the configuration for the dataset creation.
        """

        self.aws_config = aws_config
        self.s3_config = s3_config
        self.s3_config["client"] = boto3.client("s3")
        s3_bucket = self.s3_config.get("bucket", None)
        if s3_bucket in [None, ""]:
            s3_bucket = sagemaker.Session().default_bucket()
        self.s3_config["bucket"] = s3_bucket

        self.dataset_config = dataset_config
        self.dataset_config["dataset_prefix"] = os.path.join(
            self.s3_config["prefix"],
            self.dataset_config["dataset_dir"],
        )
        self.dataset_config["save_to"] = os.path.join(
            self.s3_config["bucket"],
            self.dataset_config["dataset_prefix"],
            "{}.record".format(self.dataset_config["dataset_name"]),
        )

        logging.info("Dataset will be saved to %s", self.dataset_config["save_to"])

    def generate(self, images_path: str, annotations_path: str) -> str:
        """Generates dataset in TFRecord format and leaves it in an S3 bucket"""

        if "s3:" not in images_path:
            raise ValueError("Use full s3 URI path for images.")
        if "s3:" not in annotations_path:
            raise ValueError("Use full s3 URI path for annotations.")

        self.s3_config["s3_images"] = images_path
        images_path_parts = Path(images_path).parts
        bucket_index = images_path_parts.index(self.s3_config["bucket"])
        self.s3_config["images_prefix"] = "/".join(
            images_path_parts[bucket_index + 1 :]
        )

        annotations_path_parts = Path(annotations_path).parts
        bucket_index = annotations_path_parts.index(self.s3_config["bucket"])
        self.s3_config["annotations_prefix"] = "/".join(
            annotations_path_parts[bucket_index + 1 :]
        )

        self.generate_manifest()
        self.generate_tfrecord()
        return self.dataset_config["save_to"]

    # TODO: This should probably run in the Cloud.
    def generate_manifest(self) -> None:
        """Generates json manifest required to build TFRecord."""
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        dataset_name = self.dataset_config["dataset_name"]

        manifest_entries = []
        s3_objects = [
            data
            for data in self.s3_config["client"].list_objects_v2(
                Bucket=self.s3_config["bucket"],
                Prefix=self.s3_config["annotations_prefix"],
            )["Contents"]
            if data["Key"][-4:] == ".out"
        ]

        for obj in s3_objects:
            _, filename = os.path.split(obj["Key"])
            with io.BytesIO() as annotations_file:
                self.s3_config["client"].download_fileobj(
                    self.s3_config["bucket"], obj["Key"], annotations_file
                )
                annotations_file.seek(0)

                annotations = json.load(annotations_file)
                img_dict = {
                    "source-ref": f"s3://{self.s3_config['bucket']}/"
                    f"{self.s3_config['images_prefix']}/{Path(filename).stem}",
                    dataset_name: {},
                }
                img_dict[dataset_name]["annotations"] = []

                for ann in annotations:
                    ann_dict = {}

                    if float(ann["score"]) < self.dataset_config["min_score"]:
                        continue
                    try:
                        class_id = self.dataset_config["valid_classes"].index(ann["id"])
                    except ValueError:
                        continue

                    ann_dict["class_id"] = class_id
                    ann_dict["top"] = ann["top"]
                    ann_dict["left"] = ann["left"]
                    ann_dict["width"] = ann["right"] - ann["left"]
                    ann_dict["height"] = ann["top"] - ann["bottom"]

                    img_dict[dataset_name]["annotations"].append(ann_dict)

                img_dict[f"{dataset_name}-metadata"] = {
                    "class-map": {
                        str(class_id): label
                        for class_id, label in enumerate(
                            self.dataset_config["valid_classes"]
                        )
                    },
                    "human-annotated": "no",
                    "creation-date": timestamp,
                    "type": "groundtruth/object-detection",
                }

                manifest_entries.append(json.dumps(img_dict))

        with io.BytesIO() as manifest_file:
            manifest_file.write(str.encode("\n".join(manifest_entries)))
            manifest_file.seek(0)

            self.dataset_config["manifest_prefix"] = os.path.join(
                self.s3_config["images_prefix"], "manifest.json"
            )

            self.dataset_config["s3_manifest"] = os.path.join(
                's3://',
                self.s3_config["bucket"],
                self.dataset_config["manifest_prefix"]
            )
            logging.info("Saving manifest file to %s", self.dataset_config["s3_manifest"])

            self.s3_config["client"].upload_fileobj(
                manifest_file, self.s3_config["bucket"], self.dataset_config["manifest_prefix"]
            )

    def generate_tfrecord(self) -> None:
        """Generates TFRecord file containing the training dataset required by the training docker image."""
        ts0 = time.time()
        data_processor = Processor(
            role=self.aws_config["role"],
            image_uri=self.aws_config["ecr_image"],
            instance_count=1,
            instance_type=self.aws_config["instance_type"],
            volume_size_in_gb=30,
            max_runtime_in_seconds=1200,
            base_job_name="tf2-object-detection",
        )
        ts1 = time.time()
        logger.debug("Took %.2f seconds to create data Processor.", ts1 - ts0)

        input_folder = "/opt/ml/processing/input"
        ground_truth_manifest = "/opt/ml/processing/input/manifest.json"
        label_map = {
            str(i): c for i, c in enumerate(self.dataset_config["valid_classes"])
        }
        label_map = json.dumps(label_map)
        output_folder = "/opt/ml/processing/output"

        ts0 = time.time()
        data_processor.run(
            arguments=[
                "--input={}".format(input_folder),
                "--ground_truth_manifest={}".format(ground_truth_manifest),
                "--label_map={}".format(label_map),
                "--output={}".format(output_folder),
                "--dataset-name={}".format(self.dataset_config["dataset_name"]),
            ],
            inputs=[
                ProcessingInput(
                    input_name="input",
                    source=self.s3_config["s3_images"],
                    destination=input_folder,
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="tfrecords",
                    source=output_folder,
                    destination=self.dataset_config["save_to"],
                )
            ],
        )
        ts1 = time.time()
        logger.debug("Took %2f seconds to execute data Processor.", ts1 - ts0)

    def epilogue(self) -> None:
        pass
