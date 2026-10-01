"""Tutorial 4 - your code: a HoG sliding-window detector for the digit 3.

Fill in E1-E6. Each stub's last line, `return answer(...)(...)`, calls the
released answer once your instructor pushes it (git pull), and raises
NotImplementedError until then. Replace that line with your own code.

    python hog_detector.py            runs the self-tests (E1-E6)
    python 2026-10-01-hog_demo.py     the demo; it reloads this file when you save

Conventions (same as hog_lib.py): images are float arrays (H, W), 1 = paper,
0 = ink. Pixel (row i, column j) is at (x, y) = (j, i), so y points down.
Angles are in degrees. Boxes are (x0, y0, x1, y1).
"""
from dataclasses import dataclass

import numpy as np

from hog_lib import BOX, HOG_EPS, answer, warp


# ---------------------------------------------------------------- HoG (slide 48)

def image_gradients(img):
    """E1. Gradient magnitude and angle of every pixel.

    Central differences: gx = I[y, x+1] - I[y, x-1], gy = I[y+1, x] - I[y-1, x]
    (no division by 2). Pixels on the image border get gx = gy = 0.

    img: (H, W) float.
    Returns mag (H, W) = sqrt(gx^2 + gy^2) and ang (H, W) in degrees, in
    [0, 360), with 0 = brightness increasing to the right and 90 = increasing
    downwards (np.arctan2(gy, gx)).
    """
    # TODO E1
    return answer('image_gradients')(img)


def orientation_votes(mag, ang, n_bins, signed=True):
    """E2. Each pixel votes for the orientation bins near its angle, weighted by mag.

    The angle period is 360 degrees if signed, else 180 (an angle a and a + 180
    then count the same). Bin b covers [b w, (b + 1) w) with w = period / n_bins
    and has its centre at (b + 0.5) w. Split each vote linearly between the two
    nearest bin centres, wrapping around: with 8 signed bins, 30 degrees gives
    5/6 to bin 0 and 1/6 to bin 1, and 0 degrees gives 1/2 to bin 7 and 1/2 to bin 0.

    mag, ang: arrays of the same, any, shape S.
    Returns votes, shape S + (n_bins,), summing to mag over the last axis.
    """
    # TODO E2
    return answer('orientation_votes')(mag, ang, n_bins, signed)


def hog_descriptor(votes, n_cells, eps=HOG_EPS):
    """E3. The HoG descriptor of one window.

    votes: (S, S, n_bins) from orientation_votes for an S x S window.
    Split the window into n_cells x n_cells square cells (S is a multiple of
    n_cells), sum the votes in each cell into a histogram, and stack them into
    h of shape (n_cells * n_cells * n_bins,), ordered (cell row, cell column, bin).
    Returns h / sqrt(sum(h^2) + eps^2).
    """
    # TODO E3
    return answer('hog_descriptor')(votes, n_cells, eps)


# ---------------------------------------------------------------- detection (slides 44, 47)

def iou(box, boxes):
    """E4. Intersection over union of box (4,) with each of boxes (N, 4).

    Returns (N,). Boxes that do not overlap have IoU 0. Use numpy on the
    columns (no Python loop) so it also works for a single box (4,) -> scalar.
    """
    # TODO E4
    return answer('iou')(box, boxes)


def nms(boxes, scores, iou_thresh):
    """E5. Greedy non-maximum suppression.

    Repeat: keep the highest-scoring remaining box, then drop every remaining
    box whose IoU with it is above iou_thresh.

    boxes (N, 4), scores (N,). Returns the indices kept, highest score first.
    """
    # TODO E5
    return answer('nms')(boxes, scores, iou_thresh)


@dataclass
class DetectConfig:
    """Search settings. Edit these to experiment (the demo picks up changes)."""
    scales: tuple = (0.85, 1.0, 1.15)    # digit sizes to look for (image pyramid)
    angles: tuple = (-20, 0, 20)         # digit rotations to look for (rotation bank)
    stride: int = 4                      # window step in pixels
    nms_iou: float = 0.3
    template_threshold: float = 0.8      # cosine similarity to the reference HoG
    svm_threshold: float = 1.0           # SVM decision value
    svm_angles: tuple = (-5, 0, 5)       # rotations of the training 3s (augmentation)


def detect(image, features, score_fn, threshold, cfg):
    """E6. Sliding-window detection over a pyramid of scales and rotations.

    For every s in cfg.scales and a in cfg.angles:
      1. warped, to_original = warp(image, s, a). In `warped`, a digit of scale
         s and rotation a appears upright at scale 1.
      2. centers, feats = features(warped, cfg.stride): window centres (N, 2)
         as (x, y) in `warped`, and their HoG descriptors (N, D).
      3. scores = score_fn(feats) (N,); keep windows with score > threshold.
      4. Map the kept centres back with to_original. In the original image the
         object box is a square of side BOX * s around that centre.
    Then run nms over all levels together with cfg.nms_iou.

    Returns boxes (M, 4) and scores (M,), highest score first.
    """
    # TODO E6
    return answer('detect')(image, features, score_fn, threshold, cfg)


if __name__ == '__main__':
    import hog_lib
    import hog_detector   # this file, imported by name so released answers can call it
    marks = {'pass': 'ok    ', 'answer': 'answer', 'todo': 'TODO  ', 'needs': 'needs ', 'fail': 'FAIL  '}
    for tag, name, status, detail in hog_lib.self_test(hog_detector):
        print(f'{tag} {marks[status]} {name}()' + (f'  - {detail}' if status in ('fail', 'needs') else ''))
