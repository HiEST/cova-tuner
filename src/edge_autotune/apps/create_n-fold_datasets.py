#!/bin/env python3

import os
from pathlib import Path
import sys

import tensorflow as tf

output = sys.argv[1]
tf_files = sys.argv[2:]

for f in tf_files:
    print(f)
    all_but_one = tf_files.copy()
    all_but_one.remove(f)
    dataset = tf.data.TFRecordDataset(all_but_one)

    writer = tf.data.experimental.TFRecordWriter(os.path.join(output, Path(f).parts[-1]))
    writer.write(dataset)
