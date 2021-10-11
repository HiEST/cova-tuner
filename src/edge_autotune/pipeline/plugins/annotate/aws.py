"""This module implements AWSAnnotation, a COVAAnnotate subclass, that annotates a dataset using AWS Sagemaker after uploading images to an S3 bucket."""

import io
import logging
import os
import time

import boto3
import cv2
from PIL import Image
import numpy as np


from edge_autotune.pipeline.pipeline import COVAAnnotate
from edge_autotune.api.sagemaker import ModelPackageArnProvider, deploy_model, batch_transform, get_default_bucket


logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


class AWSAnnotation(COVAAnnotate):
    """A class with all methods required to upload content to AWS S3.

    Provides methods to connect to S3 and SageMaker to
    store images, annotate them, and trigger training.

    Attributes:
        bucket: Bucket name in S3 where captured images are stored.
        key_prefix: Prefix of the key to store objects in S3 bucket.
    """
    def __init__(self, aws_config: dict, s3_config: dict):
        """Init AWSClient with bucket name to store captured images."""
        self.aws_config = aws_config

        if not s3_config.get('bucket', None):
            s3_config['bucket'] = get_default_bucket()

        default_images_prefix = os.path.join(s3_config['prefix'], 'images')
        s3_config['images_prefix'] = s3_config.get('images_prefix', default_images_prefix)
        s3_config['images_full'] = 's3://{}/{}'.format(s3_config['bucket'], s3_config['images_prefix'])

        # The location to store the results of the batch transform job
        default_annotations_prefix = os.path.join(s3_config['prefix'], 'annotations')
        s3_config['annotations_prefix'] = s3_config.get('annotations_prefix', default_annotations_prefix)
        s3_config['annotations_full'] = 's3://{}/{}'.format(s3_config['bucket'], s3_config['annotations_prefix'])
        self.s3_config = s3_config
        self.s3_config['client'] = boto3.client('s3')

        self.next_img_id = 0
        self.images_to_upload = []


    def annotate(self, img: np.array):
        """Appends image to list of images to upload"""
        self.images_to_upload.append(img)
        return True


    def upload_image(self, img: np.array, filename: str, encoding: str = 'PNG', to_rgb: bool = True):
        """Uplaods image to s3."""

        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(img)
        encoded_img = io.BytesIO()
        img_pil.save(encoded_img, encoding.upper())
        encoded_img.seek(0)

        key = os.path.join(self.s3_config['images_prefix'], filename)
        self.s3_config['client'].upload_fileobj(
            encoded_img,
            Bucket=self.s3_config['bucket'],
            Key=key,
        )

    def epilogue(self) -> str:
        """Finishes annotation step using AWS SageMaker.

        First, all images are uploaded. Then, annotation begins using
        a batch transform job in AWS Sagemaker.

        Returns:
            str: location of the annotations in s3 bucket.
        """
        for img in self.images_to_upload:
            self.upload_image(img, '{}.png'.format(self.next_img_id))
            self.next_img_id += 1

        annotations_path = self.annotate_sagemaker()
        return self.s3_config['images_full'], annotations_path


    def annotate_sagemaker(self) -> str:
        """Annotates dataset in s3 using AWS SageMaker.

        Returns:
            str: location of annotations in s3 bucket.
        """
        instance_type = self.aws_config.get('instance_type', "ml.m4.xlarge")
        instance_count = self.aws_config.get('instance_count', 1)
        max_concurrent_transforms = self.aws_config.get('max_concurrent_transforms', 4)
        content_type = self.aws_config.get('content_type', "image/png")
        region = self.aws_config.get('region', "eu-west-1")
        model_name = self.aws_config.get('model_name', 'yolov3')
        endpoint_name = self.aws_config.get('endpoint_name', "{}-gt-endpoint".format(model_name))

        model_arn = getattr(ModelPackageArnProvider, "get_{}_model_package_arn".format(model_name))(region)

        ts0 = time.time()
        _, batch = deploy_model(
            role=self.aws_config['role'],
            num_instances=instance_count,
            model_arn=model_arn,
            instance_type=instance_type,
            model_name=endpoint_name,
            output_path=self.s3_config['annotations_full'],
            max_concurrent_transforms=max_concurrent_transforms,
        )
        ts1 = time.time()

        logger.info('Model deployed successfuly after %.2f seconds.', ts1-ts0)

        ts0 = time.time()
        output_path = batch_transform(
            data=self.s3_config['images_full'], transformer=batch,
            batch_output=self.s3_config['annotations_full'], content_type=content_type)
        logger.info('Images successfuly annotated in %.2f seconds. Results stored in %s',
                    time.time()-ts0, output_path)
        return output_path
