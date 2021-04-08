# This script uses a pre-trained model to annotate images 
# in a format that can be used by TensorFlow to fine-tune other models.
import argparse
from os import listdir
from os.path import isfile, join

import cv2

from detector import init_detector, run_detector
from object_detector import ObjectDetector

COCO_CLASSES = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    12: 'stop sign',
    13: 'parking meter',
    14: 'bench',
    15: 'bird',
    16: 'cat',
    17: 'dog',
    18: 'horse',
    19: 'sheep',
    20: 'cow',
    21: 'elephant',
    22: 'bear',
    23: 'zebra',
    24: 'giraffe',
    25: 'backpack',
    26: 'umbrella',
    27: 'handbag',
    28: 'tie',
    29: 'suitcase',
    30: 'frisbee',
    31: 'skis',
    32: 'snowboard',
    33: 'sports ball',
    34: 'kite',
    35: 'baseball bat',
    36: 'baseball glove',
    37: 'skateboard',
    38: 'surfboard',
    39: 'tennis racket',
    40: 'bottle',
    41: 'wine glass',
    42: 'cup',
    43: 'fork',
    44: 'knife',
    45: 'spoon',
    46: 'bowl',
    47: 'banana',
    48: 'apple',
    49: 'sandwich',
    50: 'orange',
    51: 'broccoli',
    52: 'carrot',
    53: 'hot dog',
    54: 'pizza',
    55: 'donut',
    56: 'cake',
    57: 'chair',
    58: 'couch',
    59: 'potted plant',
    60: 'bed',
    61: 'dining table',
    62: 'toilet',
    63: 'tv',
    64: 'laptop',
    65: 'mouse',
    66: 'remote',
    67: 'keyboard',
    68: 'cell phone',
    69: 'microwave',
    70: 'oven',
    71: 'toaster',
    72: 'sink',
    73: 'refrigerator',
    74: 'book',
    75: 'clock',
    76: 'vase',
    77: 'scissors',
    78: 'teddy bear',
    79: 'hair drier',
    80: 'toothbrush'
}

OIV4_CLASSES = [
    'Person',
    'Bicycle',
    'Bicycle wheel',
    'Wheel',
    'Animal',
    'Chair',
    'Clothing',
    'Human face',
    'Footwear',
    'Man',
    'Furniture',
    'Car',
    'Flower',
    'Truck',
    'Ambulance',
    'Laptop',
    'Knife',
    'Handbag',
    'Football',
    'Chicken',
    'Fireplace',
    'Book',
    'Window',
    'Vehicle',
    'Tire',
    # 'Human leg',
    # 'Candle',
    # 'Sandal',
    'Bus',
    'Suitcase',
    # 'Human mouth',
    'Woman',
    'Door',
    # 'Helicopter',
    # 'Coffee cup',
    # 'Beer',
    # 'Sunglasses',
    # 'Helmet',
    # 'Cart',
    # 'Skirt',
    'Bird',
    # 'Television',
    # 'Human ear',
    'Train',
    'Taxi',
    'Van',
    # 'Handgun',
    'Traffic light',
    'Backpack'
]

ACCEPTED_CLASSES = OIV4_CLASSES

def generate_labelmap(labelmap, class_names):
    with open(labelmap, 'w') as f:
        for i, name in enumerate(class_names):
            item_str = "item {\n"\
                    + f"  id: {i}\n"\
                    + f"  name: '{name}'\n"\
                    + "}\n\n"
            f.write(item_str)


def main():
    # construct the argument parser and parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", default="resnet", help="Model for image classification")
    args.add_argument("-i", "--input", required=True, help="path folder containing images")
    args.add_argument("-o", "--output", required=True, help="path to the output folder where annotations will be saved")
    args.add_argument("--min-score", type=float, default=0.3, help="minimum score for detections")
    

    config = args.parse_args()
    images = [f for f in listdir(config.input) if isfile(join(config.input, f))]
    # detector = init_detector("RCNN")
    detector = ObjectDetector(config.model)
    
    # 2. create trainval.txt: list of image names without file extensions.
    trainval = open(f'{config.output}/annotations/trainval.txt', 'w')

    detected_classes = []

    # 3. generate images
    images_path = f'{config.output}/images'
    for img_id, image in enumerate(images):
        if '.jpg' not in image:
            continue
        img_id = img_id + 1
        print(f'{img_id}/{len(images)}')

        img = cv2.imread(f'{config.input}/{image}')
        result = run_detector(detector, img)

        annotations = []
        for i, box in enumerate(result["detection_boxes"]):
            class_name = result["detection_class_entities"][i].decode("ascii")
            score = result["detection_scores"][i]
            if score < config.min_score:
                # print(f'score too low: {class_name} ({score}%)')
                continue
            if not class_name in ACCEPTED_CLASSES:
                print(f'class {class_name} not among accepted classes.')
                continue

            
            print(f'Found {class_name} ({score}%)')
            if not class_name in detected_classes:
                detected_classes.append(class_name)
            
            annotations.append([class_name, box])

        if len(annotations) > 0:
            trainval.write(f'{img_id}\n')
            print(f'writing annotations for {images_path}/{img_id}.jpg')
            cv2.imwrite(f'{images_path}/{img_id}.jpg', img)

            # Write annotation (VOC format)
            annotation_str = \
                "<annotation>\n" +\
                f"<filename>{img_id}.jpg</filename>\n" +\
                f"<path>{config.output}/images/{img_id}.jpg</path>\n" +\
                "<size>\n" +\
                f"\t<width>{img.shape[1]}</width>\n" +\
                f"\t<height>{img.shape[0]}</height>\n" +\
                "\t<depth>3</depth>\n" +\
                "</size>\n" +\
                "<segmented>0</segmented>\n"
            
            for an in annotations:
                name, box = an
                annotation_str = annotation_str +\
                    "<object>\n" +\
                    f"\t<name>{name}</name>\n" +\
                    "\t<difficult>0</difficult>\n" +\
                    "\t<bndbox>\n" +\
                    f"\t\t<xmin>{box[0]}</xmin>\n" +\
                    f"\t\t<ymin>{box[1]}</ymin>\n" +\
                    f"\t\t<xmax>{box[2]}</xmax>\n" +\
                    f"\t\t<ymax>{box[3]}</ymax>\n" +\
                    "\t<bndbox>\n" +\
                    "</object>\n"
            
            annotation_str = annotation_str + "</annotation>\n"
            with open(f'{config.output}/annotations/xmls/{img_id}.xml', 'w') as f:
                f.write(annotation_str)

        # 1. create label_map.pbtxt with the classes to identify.
        labelmap_path = f'{config.output}/annotations/label_map.pbtxt'
        generate_labelmap(labelmap_path, detected_classes)
    
    trainval.close()

if __name__ == "__main__":
    main()