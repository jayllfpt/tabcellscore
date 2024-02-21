# Writer: TraiPPN2 # Date: 23/01/2024

import numpy as np
import cv2
from .utils import calculate_iou, visualize, gen_blank_image, load_label_from_file


class TabCellScore:
    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold

    def __call__(self, gt_path, pred_path,
                 img_path=None,
                 result_path=None):
        """Load the labels and calculate the score
        Args: 
            gt_path: ground-truth json file path
            pred_path: predict json file path
            img_path: image file path
            result_path: path to save visualized image
        Return: 
            the score between predicts bounding boxes 
                and ground-truth bounding box"""

        gt_boxes = load_label_from_file(gt_path)
        pred_boxes = load_label_from_file(pred_path)

        score = self.score(gt_boxes, pred_boxes)
        if result_path is not None:
            if img_path is None:
                image = gen_blank_image(pred_boxes)
            else:
                image = cv2.imread(img_path)
            visualize(image, pred_boxes, self.map_gt_pred, result_path)

        return score

    def score(self, gt_boxes, pred_boxes):
        """Calculate the score between predict boxes and ground-truth boxes
        Args:
            gt_boxes: ground-truth bounding boxes,
            pred_boxes: predicted bounding boxes
            **Note**: the input have format:
            [
                [x1, y1, x3, y3],
                ...
            ]
        Return: 
            score: Average Precision (AP)
        """
        true_positives = np.zeros(len(pred_boxes))
        false_positives = np.zeros(len(pred_boxes))
        total_gt_boxes = len(gt_boxes)
        self.map_gt_pred = {}

        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0.
            best_gt_box_idx = -1

            for j, gt_box in enumerate(gt_boxes):  # find match gt
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_box_idx = j

            if best_iou >= self.iou_threshold and best_gt_box_idx != -1:
                if not best_gt_box_idx in self.map_gt_pred:
                    true_positives[i] = 1
                    self.map_gt_pred[best_gt_box_idx] = i
                else:
                    false_positives[i] = 1
            else:
                false_positives[i] = 1

        precision = np.cumsum(true_positives) / \
            (np.cumsum(true_positives) + np.cumsum(false_positives))
        recall = np.cumsum(true_positives) / total_gt_boxes

        # Calculate Average Precision (AP) using the precision-recall curve
        return np.trapz(precision, recall)
