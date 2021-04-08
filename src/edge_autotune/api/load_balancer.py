#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import requests
import sys
from threading import Lock

from flask import Flask, Response
from flask_restful import Resource, Api, reqparse

MAX_REQS_PER_GPU = 4

url = 'http://localhost:{}/video'
servers = [5000, 5001]
servers_load = [0 for _ in servers]
lock = Lock()

videos = {}


class Video(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('video', required=True)
        parser.add_argument('model', required=True)
        args = parser.parse_args()

        status = 'video not loaded'
        data = []
        for server in servers:
            r = requests.get(url.format(server),
                             data={
                                'video': args.video,
                                'model': args.model})

            status, data = json.loads(r.text)['data']
            if status == 'ready':
                print(f'{filename} was already processed and got immediate results.')
                return True, data

        if args.video in videos.keys():
            status = 'model not requested'

        return {'data': [status, data]}, 200

    def put(self):
        parser = reqparse.RequestParser()
        parser.add_argument('video', required=True)
        parser.add_argument('overwrite', default=False, type=bool, required=False)
        parser.add_argument('file', type=FileStorage, location='files')
        args = parser.parse_args()

        if not args.overwrite and args.video in videos.keys():
            print(f'{args.video} already in server')
            return {}, 204

        tf = tempfile.NamedTemporaryFile(delete=False)
        args.file.save(tf)
        videos[args.video] = tf

        print(f'stored {args.video} in {tf.name}')
        return {}, 204

    def post(self):
        parser = reqparse.RequestParser()

        parser.add_argument('model', required=True)
        parser.add_argument('device', required=True)
        parser.add_argument('framework', required=True)
        parser.add_argument('video', required=True)
        # parser.add_argument('video', type=FileStorage, location='files')

        args = parser.parse_args()

        if args.video not in videos.keys():
            return {
                'message': 'Video does not exist.'
            }, 401


        with lock:
            # Pick one server
            min_load = min(servers_load)
            server_idx = servers_load.index(min_load)
            server_url = url.format(servers[server_idx]) 
            servers_load[server_idx] += 1

        try:
            print(f'sending request to {server_url} (server_idx={server_idx})')
            print(servers_load)
            with open(videos[args.video].name, 'rb') as video:
                r = requests.post(server_url,
                                  files={'file': video},
                                  data={
                                        'video': '',
                                        'model': model,
                                        'device': 'cuda',
                                        'framework': framework
                              })

        except ConnectionResetError:
            with lock:
                servers_load[server_idx] -= 1

            return {}, 500


        with lock:
            servers_load[server_idx] -= 1
    
        data = json.loads(r.text)['data']
        return Response(
            response=json.dumps({
                "data": data
            }),
            status=200,
            mimetype='application/json'
        )


def main():
    args = argparse.ArgumentParser()

    args.add_argument("-p", "--port",
                      default=5000,
                      type=int,
                      help="Port to listen to.")

    config = args.parse_args()
 

    app = Flask(__name__)
    api = Api(app)
    api.add_resource(Video, '/video')
    app.run(port=config.port)


if __name__ == '__main__':
    main()
