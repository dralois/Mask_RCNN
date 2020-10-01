import os
import sys
import numpy as np
import cv2
import csv
import re
import json

from numpy.core.fromnumeric import mean, squeeze

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

    # Training - Stage 2
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=45,
                layers='4+')

    # Training - Stage 3
    print("Fine tune all stages")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100,
                epochs=60,
                layers='all')

    # Training - Stage 4
    print("Fine tune all stages again")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 1000,
                epochs=100,
                layers='all')

############################################################
#  Evaluation
############################################################

def evaluate(model):

    currClass = args.object
    baseDir = args.dataset
    rgbDir = os.path.join(baseDir, "rgb")
    maskDir = os.path.join(baseDir, "mask_visib")
    gtFile = json.load(open(os.path.join(baseDir, "scene_gt_info.json"), "r"))

    allF1s = []
    allIoUs = []

    # Load dataset
    with os.scandir(rgbDir) as folder:
        for file in folder:
            imgNum = [int(s) for s in re.findall(r"\d+", file.name)][0]
            # Real image
            img = cv2.imread(file.path)
            # Ground truth mask
            mask = cv2.imread(os.path.join(maskDir, "{0:06d}_000000.png".format(imgNum)),
                flags=cv2.IMREAD_GRAYSCALE).astype(np.bool)
            # Ground truth bbox
            bboxGt = gtFile[str(imgNum)][0]["bbox_visib"]
            bboxGt[2] += bboxGt[0]
            bboxGt[3] += bboxGt[1]

            # Run detection
            r = model.detect([img])[0]
            predClasses = r["class_ids"]
            if len(predClasses) > 1:
                splitMasks = np.squeeze(np.split(r["masks"], len(predClasses), axis=2))
            elif len(predClasses) == 1:
                splitMasks = [np.squeeze(r["masks"])]
            else:
                splitMasks = []
            bboxes = []
            classes = []
            scores = []
            masks = []

            # Filter out incorrect classes
            for i in range(0, len(predClasses)):
                if predClasses[i] == int(currClass):
                    bboxes.append(r["rois"][i])
                    classes.append(currClass)
                    scores.append(r["scores"][i])
                    masks.append(splitMasks[i])

            if len(classes) > 0:
                # Format filtered lists
                bboxes = np.array(bboxes, dtype=np.int32)
                classes = np.array(classes, dtype=np.int32)
                scores = np.array(scores, dtype=np.float32)
                masks = np.stack(masks, axis=2)

                # Format gt
                gt_boxes = np.array([bboxGt], dtype=np.int32)
                gt_class_ids = np.array([currClass], dtype=np.int32)
                gt_masks = np.stack([mask], axis=2)

                # Calculate precision, recall and IoU metric
                _, precisions, recalls, ious = utils.compute_ap(gt_boxes, gt_class_ids, gt_masks, bboxes, classes, scores, masks)

                # Calculate F1 metric
                f1 = 2.0 * (((mean(precisions) * mean(recalls))) / (mean(precisions) + mean(recalls)))
                filtered = ious[np.where(ious > 0.01)]
                iou = (0.0, mean(filtered))[len(filtered) > 0]

                # Store and log
                print(f"Metrics for {file.name}: F1: {f1}, IoUs: {iou}")
                allF1s.append(f1)
                allIoUs.append(iou)
            else:
                print(f"Metrics for {file.name}: F1: {0.0}, IoUs: {0.0}")
                allF1s.append(0.0)
                allIoUs.append(0.0)

    npF1s = np.array(allF1s)
    npIoUs = np.array(allIoUs)
    filteredF1 = npF1s[np.where(npF1s > 0.01)]
    filteredIoU = npIoUs[np.where(npIoUs > 0.01)]

    meanF1 = mean(allF1s)
    meanIoU = mean(allIoUs)
    meanF1Filtered = mean(filteredF1)
    meanIoUFiltered = mean(filteredIoU)

    print(f"Final scores for {baseDir}: F1: {meanF1}, IoU: {meanIoU}; Filtered: F1: {meanF1Filtered}, IoU: {meanIoUFiltered}")


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
    parser.add_argument('--object', required=False,
                        metavar="Object class 1,2 or 3",
                        help='Evaluate this object class')
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
        assert args.object, "Argument --object is required for evaluation"

    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PRRConfig()
    else:
        class InferenceConfig(PRRConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.6
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
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
        weights_path = args.weights

    # Load weights
    print(f"Loading weights {weights_path}")
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "evaluate":
        evaluate(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
