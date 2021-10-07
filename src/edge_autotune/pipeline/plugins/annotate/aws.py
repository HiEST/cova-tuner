import io

import boto3
import cv2
from PIL import Image
import numpy as np

from edge_autotune.pipeline.pipeline import COVAAnnotate


class Annotator(COVAAnnotate):
    """A class with all methods required to upload content to AWS S3.

    Provides methods to connect to S3 and SageMaker to
    store images, annotate them, and trigger training.
    
    Attributes:
        bucket: Bucket name in S3 where captured images are stored.
        key_prefix: Prefix of the key to store objects in S3 bucket.
    """
    def __init__(self, bucket: str, key_prefix: str):
        """Init AWSClient with bucket name to store captured images."""
        self.bucket = bucket
        self.key_prefix = key_prefix
        if key_prefix[-1] != '/':
            self.key_prefix = key_prefix + '/'

        self.s3 = boto3.client('s3')
        self.next_img_id = 0
        self.images_to_upload = []

    def annotate(self, img: np.array):
        """Appends image to list of images to upload"""
        self.images_to_upload.append(img)
    
    def extend(self, img_list):
        self.images_to_upload.extend(img_list)

    def upload_image(self, img: np.array, filename: str, encoding: str = 'PNG', to_rgb: bool = True):
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(img)
        encoded_img = io.BytesIO()
        img_pil.save(encoded_img, encoding.upper())
        encoded_img.seek(0)

        key = self.key_prefix + filename
        self.s3.upload_fileobj(
            encoded_img,
            Bucket=self.bucket,
            Key=key,
        )
    
    def epilogue(self):
        for img in self.images_to_upload:
            self.upload_image(img, f'{self.next_img_id}.png')
            self.next_img_id += 1

        return True