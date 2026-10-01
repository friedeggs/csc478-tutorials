"""Answer to E5 (Tutorial 4)."""
import numpy as np

import hog_detector


def nms(boxes, scores, iou_thresh):
    boxes, scores = np.asarray(boxes, float), np.asarray(scores, float)
    order = np.argsort(-scores, kind='stable')
    keep = []
    while order.size:
        best, rest = order[0], order[1:]
        keep.append(best)
        order = rest[hog_detector.iou(boxes[best], boxes[rest]) <= iou_thresh]
    return np.array(keep, dtype=int)
