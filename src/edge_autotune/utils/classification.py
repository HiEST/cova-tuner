# import the necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2

class Classifier:

    def __init__(self, model="resnet"):
        # define a dictionary that maps model names to their classes
        # inside Keras
        self.MODELS = {
            "vgg16": VGG16,
            "vgg19": VGG19,
            "inception": InceptionV3,
            "xception": Xception, # TensorFlow ONLY
            "resnet": ResNet50,
            "mobilenetv2": MobileNetV2,
        }

        # esnure a valid model name was supplied via command line argument
        if model not in self.MODELS.keys():
            raise AssertionError("The model should be a key in the `MODELS` dictionary")

        # initialize the input image shape (224x224 pixels) along with
        # the pre-processing function (this might need to be changed
        # based on which model we use to classify our image)
        self.inputShape = (224, 224)
        self.preprocess = imagenet_utils.preprocess_input
        # if we are using the InceptionV3 or Xception networks, then we
        # need to set the input shape to (299x299) [rather than (224x224)]
        # and use a different image pre-processing function
        if model in ("inception", "xception"):
            self.inputShape = (299, 299)
            self.preprocess = preprocess_input

        # load our the network weights from disk (NOTE: if this is the
        # first time you are running this script for a given network, the
        # weights will need to be downloaded first -- depending on which
        # network you are using, the weights can be 90-575MB, so be
        # patient; the weights will be cached and subsequent runs of this
        # script will be *much* faster)
        print("[INFO] loading {}...".format(model))
        Network = self.MODELS[model]
        self.model = Network(weights="imagenet")

    def classify(self, img):
        # load the input image using the Keras helper utility while ensuring
        # the image is resized to `inputShape`, the required input dimensions
        # for the ImageNet pre-trained network
        # print("[INFO] loading and pre-processing image...")
        # image = load_img(args["image"], target_size=self.inputShape)
        image = cv2.resize(img, self.inputShape)
        image = img_to_array(image)
        # our input image is now represented as a NumPy array of shape
        # (inputShape[0], inputShape[1], 3) however we need to expand the
        # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
        # so we can pass it through the network
        image = np.expand_dims(image, axis=0)
        # pre-process the image using the appropriate function based on the
        # model that has been loaded (i.e., mean subtraction, scaling, etc.)
        image = self.preprocess(image)

        # classify the image
        # print("[INFO] classifying image with '{}'...".format(model))
        preds = self.model.predict(image)
        P = imagenet_utils.decode_predictions(preds)
        # loop over the predictions and display the rank-5 predictions +
        # probabilities to our terminal
        # for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        #     print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        return P[0]


        # load the image via OpenCV, draw the top prediction on the image,
        # and display the image to our screen
        # orig = cv2.imread(args["image"])
        # (imagenetID, label, prob) = P[0][0]
        # cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
        # 	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # cv2.imshow("Classification", orig)
        # cv2.waitKey(0)