import matplotlib
import matplotlib.pyplot as plt

import shutil
import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

MODELS_CONFIG = {
    'ssd_mobilenet_v2_320x320': {
        'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
        'batch_size': 16
    },
    'ssd_mobilenet_v2_fpnlite_320x320': {
        'model_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
        'batch_size': 16
    }
}


# Utilities
def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.8)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)


def load_images(train_image_dir):
    train_images_np = []
    trian_images = os.listdir(train_image_dir)
    for img in trian_images:
        image_path = os.path.join(train_image_dir, img)
        train_images_np.append(load_image_into_numpy_array(image_path))


    for idx, train_image_np in enumerate(train_images_np):
        plt.subplot(2, 3, idx+1)
        plt.imshow(train_image_np)
    plt.show()

def download_weights(model):
    import tarfile
    import requests
    pretrained_checkpoint = MODELS_CONFIG[model]['pretrained_checkpoint']
    download_tar = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' + pretrained_checkpoint
    save_location = f'{checkpoint_dir}/{pretrained_checkpoint}'
    r = requests.get(download_tar)
    with open(save_location, 'wb') as f:
        f.write(r.content)

    tar = tarfile.open(save_location)
    tar.extractall()
    tar.close()

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())

    
def main():
    test_record = 'workspace/castelloli/tf_record/eval.record'
    train_record = 'workspace/castelloli/tf_record/train.record'
    label_map_path = 'workspace/castelloli/annotations/label_map.pbtxt'

    chosen_model = 'ssd_mobilenet_v2_320x320'
    num_steps = 2000 #The more steps, the longer the training. Increase if your loss function is still decreasing and validation metrics are increasing. 
    num_eval_steps = 500 #Perform evaluation after so many steps

    model_name = MODELS_CONFIG[chosen_model]['model_name']
    pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
    base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']
    batch_size = MODELS_CONFIG[chosen_model]['batch_size']

    fine_tune_checkpoint = f'workspace/castelloli/checkpoints/{model_name}/checkpoint/ckpt-0'
    default_pipeline = default_pipeline = 'models/research/object_detection/configs/tf2/' + base_pipeline_file
    pipeline_config = f'workspace/castelloli/checkpoints/{model_name}/' + base_pipeline_file

    print(f'default pipeline: {default_pipeline}')
    print(f'pipeline config: {pipeline_config}')

    
    shutil.copy(default_pipeline, pipeline_config)

    num_classes = get_num_classes(label_map_path)


    import re
    print('writing custom configuration file')

    with open(pipeline_config) as f:
        s = f.read()

    with open(pipeline_config, 'w') as f:
        
        # fine_tune_checkpoint
        s = re.sub('fine_tune_checkpoint: ".*?"',
                'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)
        
        # tfrecord files train and test.
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record), s)
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(test_record), s)

        # label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_path), s)

        # Set training batch_size.
        s = re.sub('batch_size: [0-9]+',
                'batch_size: {}'.format(batch_size), s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                'num_steps: {}'.format(num_steps), s)
        
        # Set number of classes num_classes.
        s = re.sub('num_classes: [0-9]+',
                'num_classes: {}'.format(num_classes), s)
        
        #fine-tune checkpoint type
        s = re.sub(
            'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
            
        f.write(s)


    model_dir = 'workspace/castelloli/training/'

    import subprocess
    process = subprocess.call(
                    [
                        '/usr/bin/env', 'python3',
                        'models/research/object_detection/model_main_tf2.py',
                        f'--pipeline_config_path={pipeline_config}',
                        f'--model_dir={model_dir}',
                        '--alsologtostderr',
                        f'--num_train_steps={num_steps}',
                        '--sample_1_of_n_eval_examples=1',
                        f'--num_eval_steps={num_eval_steps}'
                    ])    
                
if __name__ == "__main__":
    main()
