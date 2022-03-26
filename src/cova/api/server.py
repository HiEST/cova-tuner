#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import json
import sys
import tempfile
import time

from flask import Flask, Response
from flask_restful import Resource, Api, reqparse
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from cova.dnn.infer import Model


app = Flask(__name__)
api = Api(app)

loaded_models = {}
model_in_use = ""


class Infer(Resource):
    def get(self):
        data = pd.DataFrame([], columns=["test"])
        data = data.to_dict()
        return {"data": data}, 200

    def post(self):
        print("received post request")
        parser = reqparse.RequestParser()

        parser.add_argument("img", required=True)
        parser.add_argument("model", required=True)
        # parser.add_argument('iou_threshold', required=False)

        args = parser.parse_args()

        if args.model != "" and args.model not in loaded_models.keys():
            return {"message": "Invalid model."}, 401

        img = base64.b64decode(args.img)
        nparr = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(nparr, flags=1)

        if args.model != "":
            detector = loaded_models[args.model]
        else:
            detector = loaded_models[model_in_use]

        results = detector.run([img])
        return Response(
            response=json.dumps(
                {
                    "data": results,
                }
            ),
            status=200,
            mimetype="application/json",
        )


api.add_resource(Infer, "/infer")


def start_server(
    model: str,
    model_id: str = "default",
    label_map: str = None,
    port: int = 6000,
):
    global loaded_models
    global model_in_use
    global app
    global api

    detector = Model(model, label_map)
    loaded_models[model_id] = detector
    model_in_use = model_id

    app.run(port=port)
