import os
import sys
import numpy as np
import cv2
import csv
import re

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class PRRConfig(Config):

    NAME = "prrendering"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    IMAGE_MIN_DIM = 540
    IMAGE_MAX_DIM = 960

    NUM_CLASSES = 1 + 3  # Background + bench vise, phone, drill
    MAX_GT_INSTANCES = 50

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class PRRDataset(utils.Dataset):

    def load_images(self, dir):

        # Classes
        self.add_class("prrendering", 1, "wrench")
        self.add_class("prrendering", 2, "phone")
        self.add_class("prrendering", 3, "drill")

        # Load dataset
        with os.scandir(dir) as folder:
            for file in folder:
                self.add_image(
                    "prrendering",
                    image_id=file.name,
                    path=file.path)

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        if image_info["source"] != "prrendering":
            return super(self.__class__, self).load_mask(image_id)

        num = [int(s) for s in re.findall(r"\d+", image_info["path"])][0]
        label = os.path.join(os.path.split(os.path.split(image_info["path"])[0])[0], "annotations", "labels_{0:06d}.csv".format(num))
        seg = os.path.join(os.path.split(os.path.split(image_info["path"])[0])[0], "segs", "img_{0:06d}.png".format(num))

        objects = []

        with open(label, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=";")
            next(reader)
            for row in reader:
                if row[4] == "wrench":
                    currClass = 1
                elif row[4] == "phone":
                    currClass = 2
                elif row[4] == "drill":
                    currClass = 3
                else:
                    currClass = 0
                currID = int(row[6])
                objects.append((currClass, currID))

        segs = cv2.imread(seg, flags=cv2.IMREAD_GRAYSCALE)

        masks = []
        ids = []

        for currCls, currID in objects:
            currMask = segs == currID
            masks.append(currMask)
            ids.append(currCls)

        if len(masks) > 0:
            mask = np.stack(masks, axis=2)
            id = np.array(ids, dtype=np.int32)
        else:
            print(f"{num} no masks, fix first")
            exit()

        return mask, id

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "prrendering":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Training
############################################################

def traintest(model):

    # Training dataset.
    dataset_train = PRRDataset()
    dataset_train.load_images(os.path.join(args.dataset, "train", "rgb"))
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PRRDataset()
    dataset_val.load_images(os.path.join(args.dataset, "val", "rgb"))
    dataset_val.prepare()

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')

    # Training - Stage 2
    print("Fine tune stage 4+")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='4+')

    # Training - Stage 3
    print("Fine tune all stages")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=3,
                layers='all')

def train(model):

    # Training dataset.
    dataset_train = PRRDataset()
    dataset_train.load_images(os.path.join(args.dataset, "train", "rgb"))
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PRRDataset()
    dataset_val.load_images(os.path.join(args.dataset, "val", "rgb"))
    dataset_val.prepare()

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

    exit()

    # Training - Stage 2
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=45,
                layers='4+')

    exit()

    # Training - Stage 3
    print("Fine tune all stages")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100,
                epochs=60,
                layers='all')

############################################################
#  Evaluation
############################################################

def evaluate(model):

    basedir = args.dataset
    rgbs = os.path.join(basedir, "rgb")
    masks = os.path.join(basedir, "mask_visib")
    gts = os.path.join(basedir, "scene_gt_info.json")

    # Load dataset
    with os.scandir(rgbs) as folder:
        for file in folder:
            img = cv2.imread(file.path)
            r = model.detect([img])[0]

            print(r["rois"])
            print(r["class_ids"])
            print(r["scores"])

            if len(r["rois"]) > 0:
                split = np.split(r["masks"], len(r["class_ids"]), axis=2)

                count = 0
                for mask in split:
                    count += 1
                    cv2.imshow(f"{file.name}: {count}", mask * 255.0)

                cv2.waitKey(0)
                cv2.destroyAllWindows()


############################################################
#  Other
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect LineMOD objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "evaluate":
        assert args.dataset, "Argument --dataset is required for evaluation"

    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PRRConfig()
    elif args.command == "traintest":
        class TestConfig(PRRConfig):
            LEARNING_RATE = 0.00003
            STEPS_PER_EPOCH = 500
        config = TestConfig()
    else:
        class InferenceConfig(PRRConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train" or args.command == "traintest":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    # Find last trained weights
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
    else:
        weights_path = None

    # Load weights
    print(f"Loading weights {weights_path}")
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "traintest":
        traintest(model)
    elif args.command == "evaluate":
        evaluate(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
