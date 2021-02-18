# -*- coding: utf-8 -*-

import os
import pathlib
import re
import sys

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
# from object_detection.utils import colab_utils
from object_detection.builders import model_builder
from object_detection.builders import optimizer_builder

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

# Load images and visualize
# train_images_np = []
# for i in range(1, 6):
#   image_path = os.path.join(train_image_dir, 'robertducky' + str(i) + '.jpg')
#   train_images_np.append(load_image_into_numpy_array(image_path))

# Read dataset from tfrecord
def load_dataset_from_tfrecord(tfrecord):
    num_classes = 2
    dataset = tf.data.TFRecordDataset(tfrecord)
    train_images_np = []
    train_image_tensors = []
    label_id_offset = 1
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []

    gt_boxes = []
    gt_labels = []
    for raw_example in dataset:
        parsed = tf.train.Example.FromString(raw_example.numpy())
        feature = parsed.features.feature

        raw_img = feature['image/encoded'].bytes_list.value[0]
        img = Image.open(BytesIO(raw_img))
        (im_width, im_height) = img.size
        img_np = np.array(img.getdata()).reshape(im_height, im_width, 3).astype(np.uint8)
        train_images_np.append(img_np)
        labels = [l for l in feature['image/object/class/label'].int64_list.value]

        boxes = [[] for _ in labels]
        coords = ['ymin', 'xmin', 'ymax', 'xmax']
        for coord in coords:
            for i, c in enumerate(feature[f'image/object/bbox/{coord}'].float_list.value):
                boxes[i].append(c)

        # boxes = [[0, 0, 0, 0] for _ in range(10)] #labels]
        # labels.extend([0 for _ in range(10-len(labels))])
        # # import pdb; pdb.set_trace()
        # coords = ['ymin', 'xmin', 'ymax', 'xmax']
        # for j, coord in enumerate(coords):
        #     for i, c in enumerate(feature[f'image/object/bbox/{coord}'].float_list.value):
        #         boxes[i][j] = c

        gt_boxes.append(np.array([box for box in boxes], dtype=np.float32))
        gt_labels.append(np.array([l for l in labels], dtype=np.int32))

    for (train_image_np, gt_box_np, gt_label_np) in zip(train_images_np, gt_boxes, gt_labels): 
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(
            img_np, dtype=tf.float32), axis=0))

        gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(
            gt_label_np - label_id_offset)
        gt_classes_one_hot_tensors.append(tf.one_hot(
            zero_indexed_groundtruth_classes, num_classes))

    return [train_images_np,
            train_image_tensors,
            gt_classes_one_hot_tensors,
            gt_box_tensors]

def main():
    if len(sys.argv) == 1:
        dataset_id = 'morning_25'
    else:
        dataset_id = sys.argv[1]

    os.makedirs(f'trained_models/{dataset_id}', exist_ok=True)
    print(f'Training on dataset {dataset_id}')

    pipeline_config = f'trained_models/{dataset_id}/pipeline.config'
    if not os.path.isfile(pipeline_config):
        template = f'pipeline.config'
        lines = [re.sub('TRAIN_DATASET', f'{dataset_id}', l) for l in open(template, 'r').readlines()]
        with open(pipeline_config, 'w') as f:
            for l in lines:
                f.write(l)

    tf.keras.backend.clear_session()

    print('Building model and restoring weights for fine-tuning...', flush=True)
    # pipeline_config = 'models/research/object_detection/configs/tf2/ssd_mobilenet_v2_320x320_coco17_tpu-8.config'
    # checkpoint_path = 'models/research/object_detection/test_data/checkpoint/ckpt-0'
    # checkpoint_path = 'base_model/checkpoint/ckpt-0'

    # Load pipeline config and build a detection model.
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    print(pipeline_config)
    model_config = configs['model']
    train_config = configs['train_config']
    checkpoint_path = train_config.fine_tune_checkpoint
    checkpoint_path = 'base_model/checkpoint/ckpt-0'

    num_classes = model_config.ssd.num_classes
    model_config.ssd.num_classes = 2
    print(f'Num classes: {num_classes}')
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(
          model_config=model_config, is_training=True)

    # Set up object-based checkpoint restore --- RetinaNet has two prediction
    # `heads` --- one for classification, the other for box regression.  We will
    # restore the box regression head but initialize the classification head
    # from scratch (we show the omission below by commenting out the line that
    # we would add if we wanted to restore both heads)
    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        # _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        # _prediction_heads=detection_model._box_predictor._prediction_heads,
        #    (i.e., the classification head that we *will not* restore)
        # _box_prediction_head=detection_model._box_predictor._box_prediction_head,
        _box_prediction_heads=detection_model._box_predictor._prediction_heads
        # _class_prediction_heads=detection_model._box_predictor._prediction_heads
        )
#     return fake_box_predictor
#     import pdb; pdb.set_trace()
    import pdb; pdb.set_trace()
    fake_model = tf.compat.v2.train.Checkpoint(
              _feature_extractor=detection_model._feature_extractor,
              _box_predictor=fake_box_predictor)

    # ckpt = tf.compat.v2.train.Checkpoint(model=detection_model, step=global_step, optimizer=optimizer)
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)#, step=global_step, optimizer=optimizer)
    ckpt.restore(checkpoint_path).expect_partial().assert_existing_objects_matched()# .assert_consumed()

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 300, 300, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    # status.assert_consumed()
    print('Weights restored!')

    # import pdb; pdb.set_trace()
    tfrecord_file = configs['train_input_config'].tf_record_input_reader.input_path
    dataset = load_dataset_from_tfrecord(tfrecord_file)
    train_images_np, train_image_tensors, gt_classes_one_hot_tensors, gt_box_tensors = dataset

    # By convention, our non-background classes start counting at 1.  Given
    # that we will be predicting just one class, we will therefore assign it a
    # `class id` of 1.

    category_index = {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'person'}}
    print('Done prepping data.')

    tf.keras.backend.set_learning_phase(True)

    # These parameters can be tuned; since our training set has 5 images
    # it doesn't make sense to have a much larger batch size, though we could
    # fit more examples in memory if we wanted to.
    batch_size = train_config.batch_size
    # learning_rate = 0.01
    num_steps = train_config.num_steps

    # Select variables in top layers to fine-tune.
    trainable_variables = detection_model.trainable_variables
    to_fine_tune = []
    prefixes_to_train = [
        'BoxPredictor/ConvolutionalBoxHead',
        'BoxPredictor/ConvolutionalClassHead'
    ]

    for var in trainable_variables:
      if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
        to_fine_tune.append(var)

    # Set up forward + backward pass for a single train step.
    def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
      """Get a tf.function for training step."""

      # Use tf.function for a bit of speed.
      # Comment out the tf.function decorator if you want the inside of the
      # function to run eagerly.
      # @tf.function
      def train_step_fn(image_tensors,
                        groundtruth_boxes_list,
                        groundtruth_classes_list,
                        global_step,
                        learning_rate):
        """A single training iteration.

        Args:
          image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
            Note that the height and width can vary across images, as they are
            reshaped within this function to be 640x640.
          groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
            tf.float32 representing groundtruth boxes for each image in the batch.
          groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
            with type tf.float32 representing groundtruth boxes for each image in
            the batch.

        Returns:
          A scalar tensor representing the total loss for the input batch.
        """
        shapes = tf.constant(batch_size * [[300, 300, 3]], dtype=tf.int32)
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list)
        with tf.GradientTape() as tape:
            preprocessed_images = tf.concat(
                [detection_model.preprocess(image_tensor)[0]
                for image_tensor in image_tensors], axis=0)
            prediction_dict = model.predict(preprocessed_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
      
            for loss_type in losses_dict:
                tf.compat.v2.summary.scalar(
                    loss_type, losses_dict[loss_type], step=global_step)

            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
      
            tf.compat.v2.summary.scalar('learning_rate', learning_rate, step=global_step)
            # tf.compat.v2.summary.image(
            #     name='train_input_images',
            #     step=global_step,
            #     data=features[fields.InputDataFields.image],
            #     max_outputs=3)
        return total_loss

      return train_step_fn

    print('Start fine-tuning!', flush=True)
    model_dir = f'trained_models/{dataset_id}'
    summary_writer = tf.compat.v2.summary.create_file_writer(f'{model_dir}/train')

    checkpoint_every_n = 200
    num_steps_per_iteration = 1
    with summary_writer.as_default():
        with tf.compat.v2.summary.record_if(
                lambda: global_step % num_steps_per_iteration == 0):

            global_step = tf.Variable(
                    0, trainable=False, dtype=tf.compat.v2.dtypes.int64, name='global_step',
                    aggregation=tf.compat.v2.VariableAggregation.ONLY_FIRST_REPLICA)
            # optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
            optimizer, (learning_rate,) = optimizer_builder.build(
                    train_config.optimizer, global_step=global_step)

            if callable(learning_rate):
                learning_rate_fn = learning_rate
            else:
                learning_rate_fn = lambda: learning_rate

            ckpt = tf.train.Checkpoint(model=detection_model, global_step=global_step, optimizer=optimizer)

            train_step_fn = get_model_train_step_function(
                detection_model, optimizer, to_fine_tune)

            manager = tf.compat.v2.train.CheckpointManager(ckpt, model_dir, max_to_keep=10)
            latest_checkpoint = tf.train.latest_checkpoint(model_dir)
            ckpt.restore(latest_checkpoint)
            if int(global_step.value()) == 0:
                manager.save()

            for idx in range(num_steps):
                # Grab keys for a random subset of examples
                all_keys = list(range(len(train_images_np)))
                random.shuffle(all_keys)
                example_keys = all_keys[:batch_size]

                # Note that we do not do data augmentation in this demo.  If you want a
                # a fun exercise, we recommend experimenting with random horizontal flipping
                # and random cropping :)
                gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
                gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
                image_tensors = [train_image_tensors[key] for key in example_keys]

                # Training step (forward pass + backwards pass)
                total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list, global_step, learning_rate_fn())

                if idx % 10 == 0:
                    print('batch ' + str(idx) + ' of ' + str(num_steps)
                        + ', loss=' +  str(total_loss.numpy()), flush=True)
                if idx % checkpoint_every_n == 0:
                    manager.save()
                global_step.assign_add(1)

    print('Done fine-tuning!')

    """# Load test images and run inference with new model!"""

    test_image_dir = 'models/research/object_detection/test_images/ducky/test/'
    test_images_np = []
    for i in range(1, 50):
      image_path = os.path.join(test_image_dir, 'out' + str(i) + '.jpg')
      test_images_np.append(np.expand_dims(
          load_image_into_numpy_array(image_path), axis=0))

    # Again, uncomment this decorator if you want to run inference eagerly
    @tf.function
    def detect(input_tensor):
      """Run detection on an input image.

      Args:
        input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
          Note that height and width can be anything since the image will be
          immediately resized according to the needs of the model within this
          function.

      Returns:
        A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
          and `detection_scores`).
      """
      preprocessed_image, shapes = detection_model.preprocess(input_tensor)
      prediction_dict = detection_model.predict(preprocessed_image, shapes)
      return detection_model.postprocess(prediction_dict, shapes)

    # Note that the first frame will trigger tracing of the tf.function, which will
    # take some time, after which inference should be fast.

    label_id_offset = 1
    for i in range(len(test_images_np)):
      input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
      detections = detect(input_tensor)

      plot_detections(
          test_images_np[i][0],
          detections['detection_boxes'][0].numpy(),
          detections['detection_classes'][0].numpy().astype(np.uint32)
          + label_id_offset,
          detections['detection_scores'][0].numpy(),
          category_index, figsize=(15, 20), image_name="gif_frame_" + ('%02d' % i) + ".jpg")

    imageio.plugins.freeimage.download()

    anim_file = 'duckies_test.gif'

    filenames = glob.glob('gif_frame_*.jpg')
    filenames = sorted(filenames)
    last = -1
    images = []
    for filename in filenames:
      image = imageio.imread(filename)
      images.append(image)

    imageio.mimsave(anim_file, images, 'GIF-FI', fps=5)

    display(IPyImage(open(anim_file, 'rb').read()))


if __name__ == '__main__':
    main()
