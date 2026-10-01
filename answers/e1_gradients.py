"""Answer to E1 (Tutorial 4)."""
import numpy as np


def image_gradients(img):
    img = np.asarray(img, float)
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[1:-1, 1:-1] = img[1:-1, 2:] - img[1:-1, :-2]
    gy[1:-1, 1:-1] = img[2:, 1:-1] - img[:-2, 1:-1]
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 360
    return mag, ang
