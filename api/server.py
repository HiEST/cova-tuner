#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import base64
import json
import sys
import time

from flask import Flask, request, Response
from flask_restful import Resource, Api, reqparse
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models import mobilenet_v2, resnet152
from torchvision.models.detection import faster_rcnn

from ..utils.detector import init_detector, run_detector, detect_and_draw, label_map

app = Flask(__name__)
api = Api(app)

models = {}
classes = None
with open('imagenet.txt', 'r') as c:
    classes = json.load(c)

class Infer(Resource):
    def get(self):
        data = pd.DataFrame([], columns=['test'])
        data = data.to_dict()
        return {'data': data}, 200


    def post(self):
        print('received post request')
        parser = reqparse.RequestParser()

        parser.add_argument('img', required=True)
        parser.add_argument('model', required=True)
        parser.add_argument('device', required=True)
        parser.add_argument('framework', required=True)

        args = parser.parse_args()

        if args.model not in ['edge', 'ref']:
            return {
                'message': 'Invalid model.'
            }, 401
        
        if args.device not in ['cpu', 'cuda']:
            return {
                'message': 'Invalid device.'
            }, 401

        if args.framework not in ['torch', 'tf']:
            return {
                'message': 'Invalid framework.'
            }, 401


        png_img = base64.b64decode(args.img)
        nparr = np.frombuffer(png_img, np.uint8)
        # import pdb; pdb.set_trace()
        img = cv2.imdecode(nparr, flags=1)

        m = models[args.model]
        
        if args.framework == 'torch':
            device = args.device if args.device == 'cpu' else devices[args.model]
            preds = self.infer_torch(m, img, device)
            return Response(
                response=json.dumps({
                    "data": self.get_top5_torch(preds)
                }),
                status=200,
                mimetype='application/json'
            )
        else:
            preds = self.infer_tf(m, img, args.device)
            return Response(
                response=json.dumps({
                    "data": self.get_top5_tf(preds)
                }),
                status=200,
                mimetype='application/json'
            )


    def infer_tf(self, model, img, device='cpu'):
        results = run_detector(model, img, model.input_size) 
        return results


    @torch.no_grad()
    def infer_torch(self, model, img, device='cpu'):
        dev = torch.device('cpu') if device == 'cpu' else device
        print(dev)
        print(type(model))
        model.to(dev)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        img = img.astype("single") / float(255)
        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        x = x.to(dev)

        ts0 = time.time()
        predictions = model(x)
        del x
        ts1 = time.time()
        print(f'inference took {ts1-ts0:.2f} seconds.')

        return predictions[0].detach().cpu().numpy()


    def get_top5_torch(self, preds):
        idxs = np.argpartition(-preds, 5)[:5]
        results = []
        for i, idx in enumerate(idxs):
            results.append([str(idx), f'{preds[idx]:.2f}'])
        return results


    def get_top5_tf(self, preds):
        boxes = preds['detection_boxes'][0]
        scores = preds['detection_scores'][0]
        class_ids = preds['detection_classes'][0]

        top5 = {
            'boxes': boxes[:5].tolist(),
            'scores': scores[:5].tolist(),
            'idxs': class_ids[:5].astype(int).tolist()
        }
        print(top5)
        return top5




api.add_resource(Infer, '/infer')


def main():
    global models
    global devices
    global app
    global api

    args = argparse.ArgumentParser()
    args.add_argument("-f", "--framework", 
                      default='torch',
                      choices=['torch', 'tf'],
                      help="Framework to use")

    config = args.parse_args()
    
    if config.framework == 'torch':
        models['edge'] = mobilenet_v2(pretrained=True)
        models['ref'] = resnet152(pretrained=True)
        # models['ref'] = faster_rcnn(pretrained=True)

        devices = {}
        if torch.cuda.is_available():
            devices['edge'] = torch.device('cuda:0')
            devices['ref'] = torch.device('cuda:1')
        else:
            devices['edge'] = 'cpu'
            devices['ref'] = 'cpu'

        models['edge'].eval()
        models['ref'].eval()
    elif config.framework == 'tf':
        ref_model = 'Faster R-CNN Inception ResNet V2 1024x1024'
        models['edge'] = init_detector()
        # models['ref'] = init_detector()
        models['ref'] = init_detector(ref_model)
        # print(models['ref'])
        models['edge'].input_size = (320, 320)
        # models['ref'].input_size = (320, 320)
        models['ref'].input_size = (1024, 1024)

    app.run(port=5001)
    

if __name__ == '__main__':
    main()
    # app.run()
  
