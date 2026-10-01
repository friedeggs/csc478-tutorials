"""Answer to E2 (Tutorial 4)."""
import numpy as np


def orientation_votes(mag, ang, n_bins, signed=True):
    mag = np.asarray(mag, float)
    period = 360.0 if signed else 180.0
    pos = (np.asarray(ang, float) % period) / (period / n_bins) - 0.5   # bin b is centred at b
    lo = np.floor(pos)
    frac = (pos - lo)[..., None]
    lo = lo.astype(int)[..., None] % n_bins
    bins = np.arange(n_bins)
    weights = (1 - frac) * (bins == lo) + frac * (bins == (lo + 1) % n_bins)
    return mag[..., None] * weights
