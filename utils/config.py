import os

ORIG_BASE_PATH = "datasets/castelloli"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations"])

# define the base path to the *new* dataset after running our dataset
# builder scripts and then use the base path to derive the paths to
# our output class label directories
BASE_PATH = "datasets/cars"
POSITIVE_PATH = os.path.sep.join([BASE_PATH, "car"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_car"])

NUM_CLASSES = 2

# define the maximum number of proposals when training and inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 100

# define the maximum number of positive and negative images to be
# generated from each image
MAX_POSITIVE = 30
MAX_NEGATIVE = 10

# initialize the input dimensions to the network
INPUT_DIMS = (224, 224)

# define the path to the output model and label binarizer
MODEL_PATH = "edge_detector.h5"
ENCODER_PATH = "label_encoder.pickle"

# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_SCORE = 0.99
