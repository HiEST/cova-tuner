#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements EdgeClient class"""

import base64
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import io
from itertools import product
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import pandas as pd
import requests

from PIL import Image
import boto3

if (sys.version_info.major == 3 and sys.version_info.minor >= 7):
    Request = namedtuple(
        'Request', 
        ['img', 'id', 'results', 'ts_in', 'ts_out'],
        defaults=[-1, time.time(), []]
    )
else:
    Request = namedtuple(
        'Request', 
        ['img', 'id', 'results', 'ts_in', 'ts_out'],
    )
    Request.__new__.__defaults__ = (-1, time.time(), [])


class EdgeClient:
    """A class with all methods required by the client.

    EdgeClient provides methods to connect to the server, offload
    annotation of images to obtain grountruths, and query the server
    to get and post multiple parameters.  
    
    Attributes:
        pending: List of images pending to be annotated.
        processed: List containing pairs of already processed images and their annotations.
        num_reqs: Number of requests accepted. Used to give requests individual id's.
        url: Server's url.
        port: Port to connect to the server.
    """
    def __init__(self, url: str, port: int = 6000):
        """Init EdgeClient with url and port to connect to the server."""
        self._url = url
        self._port = port
        self.num_reqs = 0
        self.pending = []
        self.processed = []


    @staticmethod
    def _process_response(response):
        results = json.loads(response.text)
        results = results.get('data', response.text)
        return response.status_code, results


    @staticmethod
    def _encode_img(img, encoding):
        # FIXME: Move to PIL or check if cv2 is using BGR.  
        _, buf = cv2.imencode(encoding, img)
        return buf


    def post_infer(self, img: np.array, encoding: str = 'png', model: str = ''):
        buf = EdgeClient._encode_img(img, '.' + encoding)
        img64 = base64.b64encode(buf)

        req_url = f'{self._url}:{self._port}/infer'
        try:
            r = requests.post(req_url, data={
                'img': img64,
                'model': model,
            })
        except ConnectionResetError:
            return False, None

        return EdgeClient._process_response(r)


    def post_request(self, request: Request):
        """Post infer with request's image. 

        Args:
            request (Request): Request to post.

        Returns:
            Request: Request with the results of the annotation.
        """
        img = request['image']
        ret, results = self.post_infer(img)
        if ret:
            request['results'] = results
            request['ts_out'] = time.time()
        return request


    def append(self, img: np.array):
        """Append image to pending requests.

        Args:
            img (np.array): Image to append.

        Returns:
            int: id of the request. 
        """
        new_req = Request(img, self.num_reqs)
        self.pending.append(new_req)
        self.num_reqs += 1
        return self.num_reqs


    # def offload_async(self, max_workers=1):


    def offload_sync(self, max_workers=1):
        """Offload synchronous requests to the server for annotation.

        Args:
            max_workers (int, optional): Number of parallel requests to the server. Defaults to 1.

        Yields:
            list: List with id of the request, 3D np.array with the image, and annotation results.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(self.post_request, self.pending)

            print(f'Processed {len(results)} requests.')
            for _, req in enumerate(results):
                self.pending.remove(req)
                yield req['id'], req['image'], req['results']


class AWSClient:
    """A class with all methods required to upload content to AWS S3.

    EdgeClient provides methods to connect to S3 and SageMaker to
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
        self.images_to_upload = []


    def upload_image(self, img, filename, encoding='PNG', to_rgb=True):
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

    
    def append(self, img):
        self.images_to_upload.append(img)

    
    def extend(self, img_list):
        self.images_to_upload.extend(img_list)

    def upload_all(self):
        for img_id, img in enumerate(self.images_to_upload):
            self.upload_image(img, f'{img_id}.png')

        return True