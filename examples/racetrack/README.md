# Examples

## 0. Start server with groundtruth model

The following command will start the server using the groundtruth model. By default, the server listens to the port 6000.
```console
foo@bar:~$ edge_autotune server --model $GT_MODEL_DIR/saved_model
* Serving Flask app "cova.api.server" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
[04/13 10:55:59] werkzeug INFO:  * Running on http://127.0.0.1:6000/ (Press CTRL+C to quit)
```

## 1. Capture (create training dataset)
First step towards a specialized fine-tuned model is the creation of the training dataset using images from the edge camera feed where the model is to be deployed.

The following command will capture images from the input stream `racetrack_cam17.mp4` and annotate them querying the server we just started in `localhost` and port 6000.
Moreover, we want to stop the command after we have 10 annotated images and we want to consider only objects from the class `car` with a minimum score (confidence) of `0.5`.
```console
foo@bar:~$ edge_autotune capture \
                    --stream racetrack_cam17.mp4 \
                    --max-images 10 \
                    --classes car \
                    --min-score 0.5 \
                    --output datasets/racetrack.record \
                    --server http://localhost \
                    --port 6000
```

## 2. Start fine-tuning of model
Now, we can start the process of fine-tuning. Since we are working with just 10 images, we will freeze the feature extractor of the base model and train only the box regression layers for just 100 epochs to avoid overfitting.

```console
foo@bar:~$ edge_autotune tune \
                     --output models/racetrack/ \
                     --config config/pipeline.edge.frozen-augmented.config \
                     --train-steps 100 \
                     --checkpoint models/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0 \
                     --dataset datasets/racetrack.record \
                     --label-map datasets/racetrack.pbtxt
```

## 3. Deploy
Finally, we can deploy our newly specialized model to the edge.
The following command will basically start inference on the specified input stream and show detections with a score greater than 0.5 (by default will show all classes present in the .pbtxt).

```console
foo@bar:~$ edge_autotune deploy
                      --stream racetrack_cam17.mp4 \
                      --model models/castelloli/saved_model/saved_model \
                      --label-map datasets/racetrack.pbtxt \
                      --min-score 0.5
```
