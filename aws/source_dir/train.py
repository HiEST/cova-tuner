#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import time
from typing import Tuple

from google.protobuf import text_format
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from object_detection import inputs
from object_detection import model_lib
from object_detection import model_lib_v2
from object_detection import exporter_lib_v2
from object_detection.utils import config_util
from object_detection.utils import ops
from object_detection.builders import model_builder, optimizer_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import pipeline_pb2

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def _compute_losses_and_predictions_dicts(
    model, features, labels, add_regularization_loss=True
):
    """[From TF's Object Detection API] Computes the losses dict and predictions dict for a model on inputs.

    Args:
      model: a DetectionModel (based on Keras).
      features: Dictionary of feature tensors from the input dataset.
        Should be in the format output by `inputs.train_input` and
        `inputs.eval_input`.
          features[fields.InputDataFields.image] is a [batch_size, H, W, C]
            float32 tensor with preprocessed images.
          features[HASH_KEY] is a [batch_size] int32 tensor representing unique
            identifiers for the images.
          features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
            int32 tensor representing the true image shapes, as preprocessed
            images could be padded.
          features[fields.InputDataFields.original_image] (optional) is a
            [batch_size, H, W, C] float32 tensor with original images.
      labels: A dictionary of groundtruth tensors post-unstacking. The original
        labels are of the form returned by `inputs.train_input` and
        `inputs.eval_input`. The shapes may have been modified by unstacking with
        `model_lib.unstack_batch`. However, the dictionary includes the following
        fields.
          labels[fields.InputDataFields.num_groundtruth_boxes] is a
            int32 tensor indicating the number of valid groundtruth boxes
            per image.
          labels[fields.InputDataFields.groundtruth_boxes] is a float32 tensor
            containing the corners of the groundtruth boxes.
          labels[fields.InputDataFields.groundtruth_classes] is a float32
            one-hot tensor of classes.
          labels[fields.InputDataFields.groundtruth_weights] is a float32 tensor
            containing groundtruth weights for the boxes.
          -- Optional --
          labels[fields.InputDataFields.groundtruth_instance_masks] is a
            float32 tensor containing only binary values, which represent
            instance masks for objects.
          labels[fields.InputDataFields.groundtruth_keypoints] is a
            float32 tensor containing keypoints for each box.
          labels[fields.InputDataFields.groundtruth_dp_num_points] is an int32
            tensor with the number of sampled DensePose points per object.
          labels[fields.InputDataFields.groundtruth_dp_part_ids] is an int32
            tensor with the DensePose part ids (0-indexed) per object.
          labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
            float32 tensor with the DensePose surface coordinates.
          labels[fields.InputDataFields.groundtruth_group_of] is a tf.bool tensor
            containing group_of annotations.
          labels[fields.InputDataFields.groundtruth_labeled_classes] is a float32
            k-hot tensor of classes.
          labels[fields.InputDataFields.groundtruth_track_ids] is a int32
            tensor of track IDs.
      add_regularization_loss: Whether or not to include the model's
        regularization loss in the losses dictionary.

    Returns:
      A tuple containing the losses dictionary (with the total loss under
      the key 'Loss/total_loss'), and the predictions dictionary produced by
      `model.predict`.

    """
    model_lib.provide_groundtruth(model, labels)
    preprocessed_images = features[fields.InputDataFields.image]

    prediction_dict = model.predict(
        preprocessed_images,
        features[fields.InputDataFields.true_image_shape],
        **model.get_side_inputs(features),
    )
    prediction_dict = ops.bfloat16_to_float32_nested(prediction_dict)

    losses_dict = model.loss(
        prediction_dict, features[fields.InputDataFields.true_image_shape]
    )
    losses = [loss_tensor for loss_tensor in losses_dict.values()]
    if add_regularization_loss:
        # TODO(kaftan): As we figure out mixed precision & bfloat 16, we may
        ## need to convert these regularization losses from bfloat16 to float32
        ## as well.
        regularization_losses = model.regularization_losses()
        if regularization_losses:
            regularization_losses = ops.bfloat16_to_float32_nested(
                regularization_losses
            )
            regularization_loss = tf.add_n(
                regularization_losses, name="regularization_loss"
            )
            losses.append(regularization_loss)
            losses_dict["Loss/regularization_loss"] = regularization_loss

    total_loss = tf.add_n(losses, name="total_loss")
    losses_dict["Loss/total_loss"] = total_loss

    return losses_dict, prediction_dict


# Modified version from model_lib_v2.py from the TF's OD API.
# + trainable_layers
def eager_train_step(
    detection_model,
    features,
    labels,
    unpad_groundtruth_tensors,
    optimizer,
    learning_rate,
    add_regularization_loss=True,
    clip_gradients_value=None,
    global_step=None,
    num_replicas=1.0,
    trainable_variables=None,
):
    """[From TF's Object Detection API] Process a single training batch.

    This method computes the loss for the model on a single training batch,
    while tracking the gradients with a gradient tape. It then updates the
    model variables with the optimizer, clipping the gradients if
    clip_gradients_value is present.

    This method can run eagerly or inside a tf.function.

    Args:
      detection_model: A DetectionModel (based on Keras) to train.
      features: Dictionary of feature tensors from the input dataset.
        Should be in the format output by `inputs.train_input.
          features[fields.InputDataFields.image] is a [batch_size, H, W, C]
            float32 tensor with preprocessed images.
          features[HASH_KEY] is a [batch_size] int32 tensor representing unique
            identifiers for the images.
          features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
            int32 tensor representing the true image shapes, as preprocessed
            images could be padded.
          features[fields.InputDataFields.original_image] (optional, not used
            during training) is a
            [batch_size, H, W, C] float32 tensor with original images.
      labels: A dictionary of groundtruth tensors. This method unstacks
        these labels using model_lib.unstack_batch. The stacked labels are of
        the form returned by `inputs.train_input` and `inputs.eval_input`.
          labels[fields.InputDataFields.num_groundtruth_boxes] is a [batch_size]
            int32 tensor indicating the number of valid groundtruth boxes
            per image.
          labels[fields.InputDataFields.groundtruth_boxes] is a
            [batch_size, num_boxes, 4] float32 tensor containing the corners of
            the groundtruth boxes.
          labels[fields.InputDataFields.groundtruth_classes] is a
            [batch_size, num_boxes, num_classes] float32 one-hot tensor of
            classes. num_classes includes the background class.
          labels[fields.InputDataFields.groundtruth_weights] is a
            [batch_size, num_boxes] float32 tensor containing groundtruth weights
            for the boxes.
          -- Optional --
          labels[fields.InputDataFields.groundtruth_instance_masks] is a
            [batch_size, num_boxes, H, W] float32 tensor containing only binary
            values, which represent instance masks for objects.
          labels[fields.InputDataFields.groundtruth_keypoints] is a
            [batch_size, num_boxes, num_keypoints, 2] float32 tensor containing
            keypoints for each box.
          labels[fields.InputDataFields.groundtruth_dp_num_points] is a
            [batch_size, num_boxes] int32 tensor with the number of DensePose
            sampled points per instance.
          labels[fields.InputDataFields.groundtruth_dp_part_ids] is a
            [batch_size, num_boxes, max_sampled_points] int32 tensor with the
            part ids (0-indexed) for each instance.
          labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
            [batch_size, num_boxes, max_sampled_points, 4] float32 tensor with the
            surface coordinates for each point. Each surface coordinate is of the
            form (y, x, v, u) where (y, x) are normalized image locations and
            (v, u) are part-relative normalized surface coordinates.
          labels[fields.InputDataFields.groundtruth_labeled_classes] is a float32
            k-hot tensor of classes.
          labels[fields.InputDataFields.groundtruth_track_ids] is a int32
            tensor of track IDs.
      unpad_groundtruth_tensors: A parameter passed to unstack_batch.
      optimizer: The training optimizer that will update the variables.
      learning_rate: The learning rate tensor for the current training step.
        This is used only for TensorBoard logging purposes, it does not affect
         model training.
      add_regularization_loss: Whether or not to include the model's
        regularization loss in the losses dictionary.
      clip_gradients_value: If this is present, clip the gradients global norm
        at this value using `tf.clip_by_global_norm`.
      global_step: The current training step. Used for TensorBoard logging
        purposes. This step is not updated by this function and must be
        incremented separately.
      num_replicas: The number of replicas in the current distribution strategy.
        This is used to scale the total loss so that training in a distribution
        strategy works correctly.

    Returns:
      The total loss observed at this training step
    """
    # """Execute a single training step in the TF v2 style loop."""
    is_training = True

    detection_model._is_training = is_training  # pylint: disable=protected-access
    tf.keras.backend.set_learning_phase(is_training)

    labels = model_lib.unstack_batch(
        labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors
    )

    with tf.GradientTape() as tape:
        losses_dict, _ = _compute_losses_and_predictions_dicts(
            detection_model, features, labels, add_regularization_loss
        )

        total_loss = losses_dict["Loss/total_loss"]

        # Normalize loss for num replicas
        total_loss = tf.math.divide(
            total_loss, tf.constant(num_replicas, dtype=tf.float32)
        )
        losses_dict["Loss/normalized_total_loss"] = total_loss

    for loss_type in losses_dict:
        tf.compat.v2.summary.scalar(loss_type, losses_dict[loss_type], step=global_step)

    if trainable_variables is None:
        trainable_variables = detection_model.trainable_variables
    gradients = tape.gradient(total_loss, trainable_variables)

    if clip_gradients_value:
        gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients_value)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    tf.compat.v2.summary.scalar("learning_rate", learning_rate, step=global_step)
    tf.compat.v2.summary.image(
        name="train_input_images",
        step=global_step,
        data=features[fields.InputDataFields.image],
        max_outputs=3,
    )
    return total_loss


# Modified version from model_lib_v2.py from the TF's OD API.
# + trainable_layers
def train_loop(
    pipeline_config_path,
    model_dir,
    fine_tune_checkpoint=None,
    label_map=None,
    config_override=None,
    train_datasets=None,
    train_steps=None,
    use_tpu=False,
    save_final_config=True,
    checkpoint_every_n=1000,
    checkpoint_max_to_keep=7,
    record_summaries=True,
    performance_summary_exporter=None,
    **kwargs,
):
    """[From TF's Object Detection API] Trains a model using eager + functions.

    This method:
      1. Processes the pipeline configs
      2. (Optionally) saves the as-run config
      3. Builds the model & optimizer
      4. Gets the training input data
      5. Loads a fine-tuning detection or classification checkpoint if requested
      6. Loops over the train data, executing distributed training steps inside
         tf.functions.
      7. Checkpoints the model every `checkpoint_every_n` training steps.
      8. Logs the training metrics as TensorBoard summaries.

    Args:
      pipeline_config_path: A path to a pipeline config file.
      model_dir:
        The directory to save checkpoints and summaries to.
      config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to
        override the config from `pipeline_config_path`.
      train_steps: Number of training steps. If None, the number of training steps
        is set from the `TrainConfig` proto.
      use_tpu: Boolean, whether training and evaluation should run on TPU.
      save_final_config: Whether to save final config (obtained after applying
        overrides) to `model_dir`.
      checkpoint_every_n:
        Checkpoint every n training steps.
      checkpoint_max_to_keep:
        int, the number of most recent checkpoints to keep in the model directory.
      record_summaries: Boolean, whether or not to record summaries.
      performance_summary_exporter: function for exporting performance metrics.
      **kwargs: Additional keyword arguments for configuration override.
    """
    ## Parse the configs
    steps_per_sec_list = []

    configs = config_util.get_configs_from_pipeline_file(
        pipeline_config_path, config_override=config_override
    )
    kwargs.update(
        {
            "train_steps": train_steps,
            "use_bfloat16": configs["train_config"].use_bfloat16 and use_tpu,
        }
    )
    configs = config_util.merge_external_params_with_configs(
        configs, None, kwargs_dict=kwargs
    )
    model_config = configs["model"]
    train_config = configs["train_config"]
    train_input_config = configs["train_input_config"]

    if train_datasets:
        datasets_in_config = len(train_input_config.tf_record_input_reader.input_path)
        for i in range(datasets_in_config):
            train_input_config.tf_record_input_reader.input_path[i] = train_datasets[i]
        datasets_left = len(train_datasets) - datasets_in_config
        for i in range(datasets_left):
            dataset_id = i + datasets_in_config
            train_input_config.tf_record_input_reader.input_path.append(
                train_datasets[dataset_id]
            )

        logger.info(f"train_input_config: {train_input_config}")

    if fine_tune_checkpoint:
        train_config.fine_tune_checkpoint = fine_tune_checkpoint
    else:
        fine_tune_checkpoint = train_config.fine_tune_checkpoint

    if label_map:
        train_input_config.label_map_path = label_map

    unpad_groundtruth_tensors = train_config.unpad_groundtruth_tensors
    add_regularization_loss = train_config.add_regularization_loss
    clip_gradients_value = None
    if train_config.gradient_clipping_by_norm > 0:
        clip_gradients_value = train_config.gradient_clipping_by_norm

    # update train_steps from config but only when non-zero value is provided
    if train_steps is None and train_config.num_steps != 0:
        train_steps = train_config.num_steps

    if kwargs["use_bfloat16"]:
        tf.compat.v2.keras.mixed_precision.experimental.set_policy("mixed_bfloat16")

    if train_config.load_all_detection_checkpoint_vars:
        raise ValueError(
            "train_pb2.load_all_detection_checkpoint_vars " "unsupported in TF2"
        )

    config_util.update_fine_tune_checkpoint_type(train_config)
    fine_tune_checkpoint_type = train_config.fine_tune_checkpoint_type
    fine_tune_checkpoint_version = train_config.fine_tune_checkpoint_version

    # Write the as-run pipeline config to disk.
    if save_final_config:
        pipeline_config_final = config_util.create_pipeline_proto_from_configs(configs)
        config_util.save_pipeline_config(pipeline_config_final, model_dir)

    # Build the model, optimizer, and training input
    strategy = tf.compat.v2.distribute.get_strategy()
    with strategy.scope():
        detection_model = model_builder.build(
            model_config=model_config, is_training=True
        )

        def train_dataset_fn(input_context):
            """Callable to create train input."""
            # Create the inputs.
            train_input = inputs.train_input(
                train_config=train_config,
                train_input_config=train_input_config,
                model_config=model_config,
                model=detection_model,
                input_context=input_context,
            )
            train_input = train_input.repeat()
            return train_input

        train_input = strategy.experimental_distribute_datasets_from_function(
            train_dataset_fn
        )

        global_step = tf.Variable(
            0,
            trainable=False,
            dtype=tf.compat.v2.dtypes.int64,
            name="global_step",
            aggregation=tf.compat.v2.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        optimizer, (learning_rate,) = optimizer_builder.build(
            train_config.optimizer, global_step=global_step
        )

        if callable(learning_rate):
            learning_rate_fn = learning_rate
        else:
            learning_rate_fn = lambda: learning_rate

    ## Train the model
    # Get the appropriate filepath (temporary or not) based on whether the worker
    # is the chief.
    summary_writer_filepath = model_lib_v2.get_filepath(
        strategy, os.path.join(model_dir, "train")
    )
    if record_summaries:
        summary_writer = tf.compat.v2.summary.create_file_writer(
            summary_writer_filepath
        )
    else:
        summary_writer = tf2.summary.create_noop_writer()

    if use_tpu:
        num_steps_per_iteration = 100
    else:
        # TODO(b/135933080) Explore setting to 100 when GPU performance issues
        # are fixed.
        num_steps_per_iteration = 1

    with summary_writer.as_default():
        with strategy.scope():
            with tf.compat.v2.summary.record_if(
                lambda: global_step % num_steps_per_iteration == 0
            ):
                # Load a fine-tuning checkpoint.
                if fine_tune_checkpoint:
                    model_lib_v2.load_fine_tune_checkpoint(
                        model=detection_model,
                        checkpoint_path=fine_tune_checkpoint,
                        checkpoint_type=fine_tune_checkpoint_type,
                        checkpoint_version=fine_tune_checkpoint_version,
                        input_dataset=train_input,
                        unpad_groundtruth_tensors=unpad_groundtruth_tensors,
                        run_model_on_dummy_input=True,
                    )

                ckpt = tf.compat.v2.train.Checkpoint(
                    step=global_step, model=detection_model, optimizer=optimizer
                )

                manager_dir = model_lib_v2.get_filepath(strategy, model_dir)
                if not strategy.extended.should_checkpoint:
                    checkpoint_max_to_keep = 1
                manager = tf.compat.v2.train.CheckpointManager(
                    ckpt, manager_dir, max_to_keep=checkpoint_max_to_keep
                )

                # We use the following instead of manager.latest_checkpoint because
                # manager_dir does not point to the model directory when we are running
                # in a worker.
                latest_checkpoint = tf.train.latest_checkpoint(model_dir)
                ckpt.restore(latest_checkpoint)

                trainable_variables = []
                include_variables = (
                    train_config.update_trainable_variables
                    if train_config.update_trainable_variables
                    else None
                )

                exclude_variables = (
                    train_config.freeze_variables
                    if train_config.freeze_variables
                    else None
                )

                if include_variables is not None:
                    trainable_variables = [
                        var
                        for var in detection_model.trainable_variables
                        if any(
                            [
                                var.name.startswith(prefix)
                                for prefix in include_variables
                            ]
                        )
                    ]
                elif exclude_variables is not None:
                    trainable_variables = [
                        var
                        for var in detection_model.trainable_variables
                        if not any(
                            [
                                var.name.startswith(prefix)
                                for prefix in exclude_variables
                            ]
                        )
                    ]
                else:
                    trainable_variables = detection_model.trainable_variables

                # with open('trainable_variables.txt', 'w') as f:
                #     for var in trainable_variables:
                #         f.write(f'{var.name}\n')

                def train_step_fn(features, labels):
                    """Single train step."""
                    loss = eager_train_step(
                        detection_model,
                        features,
                        labels,
                        unpad_groundtruth_tensors,
                        optimizer,
                        learning_rate=learning_rate_fn(),
                        add_regularization_loss=add_regularization_loss,
                        clip_gradients_value=clip_gradients_value,
                        global_step=global_step,
                        num_replicas=strategy.num_replicas_in_sync,
                        trainable_variables=trainable_variables,
                    )
                    global_step.assign_add(1)
                    return loss

                def _sample_and_train(strategy, train_step_fn, data_iterator):
                    features, labels = data_iterator.next()
                    if hasattr(tf.distribute.Strategy, "run"):
                        per_replica_losses = strategy.run(
                            train_step_fn, args=(features, labels)
                        )
                    else:
                        per_replica_losses = strategy.experimental_run_v2(
                            train_step_fn, args=(features, labels)
                        )
                    # TODO(anjalisridhar): explore if it is safe to remove the
                    ## num_replicas scaling of the loss and switch this to a ReduceOp.Mean
                    return strategy.reduce(
                        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
                    )

                @tf.function
                def _dist_train_step(data_iterator):
                    """A distributed train step."""

                    if num_steps_per_iteration > 1:
                        for _ in tf.range(num_steps_per_iteration - 1):
                            # Following suggestion on yaqs/5402607292645376
                            with tf.name_scope(""):
                                _sample_and_train(
                                    strategy, train_step_fn, data_iterator
                                )

                    return _sample_and_train(strategy, train_step_fn, data_iterator)

                train_input_iter = iter(train_input)

                if int(global_step.value()) == 0:
                    manager.save()

                checkpointed_step = int(global_step.value())
                logged_step = global_step.value()

                last_step_time = time.time()
                for _ in range(
                    global_step.value(), train_steps, num_steps_per_iteration
                ):

                    loss = _dist_train_step(train_input_iter)

                    time_taken = time.time() - last_step_time
                    last_step_time = time.time()
                    steps_per_sec = num_steps_per_iteration * 1.0 / time_taken

                    tf.compat.v2.summary.scalar(
                        "steps_per_sec", steps_per_sec, step=global_step
                    )

                    steps_per_sec_list.append(steps_per_sec)

                    if global_step.value() - logged_step >= 100:
                        tf.logging.info(
                            "Step {} per-step time {:.3f}s loss={:.3f}".format(
                                global_step.value(),
                                time_taken / num_steps_per_iteration,
                                loss,
                            )
                        )
                        logged_step = global_step.value()

                    if global_step.value() % 10 == 0:
                        logger.info(f"[step={global_step.value():.3f}] loss={loss}")

                    if (
                        int(global_step.value()) - checkpointed_step
                    ) >= checkpoint_every_n:
                        manager.save()
                        checkpointed_step = int(global_step.value())

    # Remove the checkpoint directories of the non-chief workers that
    # MultiWorkerMirroredStrategy forces us to save during sync distributed
    # training.
    model_lib_v2.clean_temporary_directories(strategy, manager_dir)
    model_lib_v2.clean_temporary_directories(strategy, summary_writer_filepath)
    # TODO(pkanwar): add accuracy metrics.
    if performance_summary_exporter is not None:
        metrics = {
            "steps_per_sec": np.mean(steps_per_sec_list),
            "steps_per_sec_p50": np.median(steps_per_sec_list),
            "steps_per_sec_max": max(steps_per_sec_list),
        }
        mixed_precision = "bf16" if kwargs["use_bfloat16"] else "fp32"
        performance_summary_exporter(metrics, mixed_precision)


def train_loop_wrapper(
    pipeline_config: str,
    model_dir: str,
    train_datasets: Tuple[str],
    num_train_steps: int,
    base_model: str = None,
    label_map: str = None,
    checkpoint_every_n: int = 1000,
    record_summaries: bool = True,
):

    tf2.config.set_soft_device_placement(True)

    strategy = tf2.compat.v2.distribute.MirroredStrategy()
    with strategy.scope():
        train_loop(
            pipeline_config_path=pipeline_config,
            model_dir=model_dir,
            train_datasets=train_datasets,
            train_steps=num_train_steps,
            use_tpu=False,
            fine_tune_checkpoint=base_model,
            label_map=label_map,
            checkpoint_every_n=min(checkpoint_every_n, num_train_steps),
            record_summaries=record_summaries,
            checkpoint_max_to_keep=max(1, int(num_train_steps / checkpoint_every_n)),
        )


def export_trained_model(
    pipeline_config_path: str,
    trained_checkpoint_dir: str,
    output_dir: str,
    input_type: str = "image_tensor",
):
    tf2.enable_v2_behavior()
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf2.io.gfile.GFile(pipeline_config_path, "r") as f:
        text_format.Merge(f.read(), pipeline_config)

    config_override = ""
    text_format.Merge(config_override, pipeline_config)

    exporter_lib_v2.export_inference_graph(
        input_type=input_type,
        pipeline_config=pipeline_config,
        trained_checkpoint_dir=trained_checkpoint_dir,
        output_directory=output_dir,
        use_side_inputs=False,
        side_input_shapes="",
        side_input_types="",
        side_input_names="",
    )


def train(args):
    train_loop_wrapper(
        pipeline_config=args.pipeline_config_path,
        model_dir=args.model_dir,
        train_datasets=[args.train_dataset],
        num_train_steps=args.num_train_steps,
        base_model=args.checkpoint_dir,
        label_map=args.label_map,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument(
        "--pipeline_config_path",
        type=str,
        default=os.environ.get("SM_HP_PIPELINE_CONFIG_PATH"),
    )

    parser.add_argument("--input", type=str, default=os.environ.get("SM_INPUT_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--eval", type=str, default=os.environ.get("SM_CHANNEL_EVAL"))

    parser.add_argument(
        "--num_train_steps", type=int, default=os.environ.get("SM_HP_NUM_TRAIN_STEPS")
    )
    parser.add_argument(
        "--sample_1_of_n_eval_examples",
        type=int,
        default=os.environ.get("SM_HP_SAMPLE_1_OF_N_EVAL_EXAMPLES"),
    )

    args = parser.parse_args()
    args.label_map = os.path.join(args.train, "label_map.pbtxt")
    args.train_dataset = os.path.join(args.train, "train.records")
    args.eval_dataset = os.path.join(args.train, "validation.records")
    # args.checkpoint_dir = os.path.join(args.input, args.checkpoint_dir)
    # args.pipeline_config_path = os.path.join(args.input, args.pipeline_config_path)

    assert os.path.exists(args.checkpoint_dir)
    assert os.path.isfile(args.pipeline_config_path)
    assert os.path.isfile(args.train_dataset)
    assert os.path.isfile(args.label_map)

    logger.info("Starting training...")
    logger.info(f"args: {args}")
    train(args)
    logger.info("Training finished.")

    # export_trained_model(
    #   pipeline_config_path=args.pipeline_config_path,
    #   trained_checkpoint_dir=args.
    # )


if __name__ == "__main__":
    main()
