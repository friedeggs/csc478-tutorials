"""Answer to E3 (Tutorial 4)."""
import numpy as np


def hog_descriptor(votes, n_cells, eps=1.0):
    S, _, B = votes.shape
    c = S // n_cells
    h = np.asarray(votes, float).reshape(n_cells, c, n_cells, c, B).sum(axis=(1, 3)).ravel()
    return h / np.sqrt(np.sum(h ** 2) + eps ** 2)
