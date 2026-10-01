"""Answer to E6 (Tutorial 4)."""
import numpy as np

import hog_detector
from hog_lib import BOX, warp


def detect(image, features, score_fn, threshold, cfg):
    all_boxes, all_scores = [np.zeros((0, 4))], [np.zeros(0)]
    for s in cfg.scales:
        for a in cfg.angles:
            warped, to_original = warp(image, s, a)
            centers, feats = features(warped, cfg.stride)
            scores = np.asarray(score_fn(feats))
            keep = scores > threshold
            c = to_original(centers[keep])
            half = BOX * s / 2
            all_boxes.append(np.column_stack([c - half, c + half]))
            all_scores.append(scores[keep])
    boxes, scores = np.concatenate(all_boxes), np.concatenate(all_scores)
    keep = hog_detector.nms(boxes, scores, cfg.nms_iou)
    return boxes[keep], scores[keep]
