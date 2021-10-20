"""This module implements generation of TFRecord dataset to be used by AWS Sagemaker."""

from datetime import datetime
import io
import logging
import json
import os
from pathlib import Path
import time

import boto3
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

        self.dataset_config = dataset_config
        self.dataset_destination = os.path.join(
            "s3://",
            self.s3_config["bucket"],
            "dataset",
            "{}.record".format(self.dataset_config["dataset_name"]),
        )

    def generate(self, images_path: str, annotations_path: str) -> str:
        """Generates dataset in TFRecord format and leaves it in an S3 bucket"""
        assert "s3://" in images_path and "s3://" in annotations_path
        self.s3_config["images_full"] = images_path
        self.s3_config["images_prefix"] = "/".join(images_path.split("/")[3:])

        self.s3_config["annotations_full"] = annotations_path
        self.s3_config["annotations_prefix"] = "/".join(annotations_path.split("/")[3:])

        self.generate_manifest()
        self.generate_tfrecord()
        return self.dataset_destination

    # TODO: This should probably run in the Cloud.
    def generate_manifest(self) -> None:
        """Generates json manifest required to build TFRecord."""
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        dataset_name = self.dataset_config["dataset_name"]

        manifest_entries = []
        s3_objects = self.s3_config["client"].list_objects_v2(
            Bucket=self.s3_config["bucket"], Prefix=self.s3_config["annotations_prefix"]
        )["Contents"]
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
                    f"{self.s3_config['imgs_prefix']}/{Path(filename).stem}",
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

        # with open('/tmp/manifest.json', 'w') as manifest_file:
        with io.BytesIO() as manifest_file:
            manifest_file.write("\n".join(manifest_entries))
            manifest_file.seek(0)

            self.s3_config["manifest"] = os.path.join(
                self.s3_config["annotations_prefix"], "manifest.json"
            )
            self.s3_config["client"].upload_fileobj(
                manifest_file, self.s3_config["bucket"], self.s3_config["manifest"]
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
        logger.info("Took %.2f seconds to create data Processor.", ts1 - ts0)

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
            ],
            inputs=[
                ProcessingInput(
                    input_name="input",
                    source=self.s3_config["manifest"],
                    destination=input_folder,
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="tfrecords",
                    source=output_folder,
                    destination=self.dataset_destination,
                )
            ],
        )
        ts1 = time.time()
        logger.info("Took %2f seconds to execute data Processor.", ts1 - ts0)

    def epilogue(self) -> None:
        pass
