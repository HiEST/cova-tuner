#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os

cwd = os.path.dirname(os.path.realpath(__file__))

IMAGENET = None
with open(f'{cwd}/../aux/imagenet.txt', 'r') as c:
    IMAGENET = json.load(c)

MSCOCO = {}
with open(f'{cwd}/../aux/mscoco.json') as labels:
    mscoco = json.load(labels)
    
    for k in mscoco.keys():
        MSCOCO[int(k)] = mscoco[k]

    del mscoco
