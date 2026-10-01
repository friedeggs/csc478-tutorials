"""Given code for Tutorial 4 (HoG object detection). You do not need to edit it.

It holds everything around your TODOs in hog_detector.py:

    answer(name)         looks up a released answer in answers/ (see hog_detector.py)
    HogConfig, PRESETS   the HoG variants the demo switches between
    extract_patch, warp  image resampling (rotate / scale) for the probe and the pyramid
    patch_descriptor     HoG of one window, built from your E1-E3
    dense_windows        HoG of every window of an image at once (integral images)
    Template             reference HoG of the train 3s and the match score
    LinearSVM, sample_windows, evaluate
    self_test            the checks behind `python hog_detector.py` and the demo's E1-E6 chips

Conventions: images are float arrays (H, W), 1 = paper, 0 = ink. Pixel (row i,
column j) has its centre at (x, y) = (j, i), so y points down. Angles are in
degrees, counter-clockwise as seen on screen. Boxes are (x0, y0, x1, y1).
"""
import importlib
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

WINDOW = 48      # HoG window side in pixels (scale 1)
BOX = 36         # object box side at scale 1: the digit inside the window, 6 px margin
HOG_EPS = 1.0    # descriptor normalization: h / sqrt(|h|^2 + eps^2)


# ---------------------------------------------------------------- answers

ANSWERS = {
    'image_gradients':   ('E1', 'answers.e1_gradients'),
    'orientation_votes': ('E2', 'answers.e2_votes'),
    'hog_descriptor':    ('E3', 'answers.e3_descriptor'),
    'iou':               ('E4', 'answers.e4_iou'),
    'nms':               ('E5', 'answers.e5_nms'),
    'detect':            ('E6', 'answers.e6_detect'),
}
answers_used = set()   # names whose released answer was called (for the demo's chips)


def answer(name):
    """Return the released answer for the TODO `name`.

    Answers are released during the tutorial as files in answers/ (git pull).
    Until then this raises NotImplementedError. Setting the environment
    variable HOG_NO_ANSWERS=1 ignores released answers."""
    tag, module = ANSWERS[name]
    if os.environ.get('HOG_NO_ANSWERS'):
        raise NotImplementedError(f'{tag} {name}() is a TODO')
    if module not in sys.modules:
        importlib.invalidate_caches()   # notice files added since the last try
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name not in (module, 'answers'):
            raise
        raise NotImplementedError(f'{tag} {name}() is a TODO') from None
    answers_used.add(name)
    return getattr(mod, name)


# ---------------------------------------------------------------- HoG variants

@dataclass(frozen=True)
class HogConfig:
    name: str = 'Slide 48'
    layout: str = 'grid'    # 'grid': n_cells x n_cells square cells; 'rings': n_cells rings
    n_cells: int = 4
    n_bins: int = 8
    signed: bool = True     # angles over 0-360 (signed) or 0-180 (unsigned)
    align: bool = False     # circularly shift the bins so the strongest one is bin 0

    @property
    def n_regions(self):
        return self.n_cells ** 2 if self.layout == 'grid' else self.n_cells

    def summary(self):
        where = (f'{self.n_cells}x{self.n_cells} cells' if self.layout == 'grid'
                 else f'{self.n_cells} rings, angle vs radius')
        sign = 'signed' if self.signed else 'unsigned'
        return f'{where}, {self.n_bins} {sign} bins' + (', aligned' if self.align else '')


PRESETS = [
    HogConfig('Slide 48', 'grid', 4, 8, True),
    HogConfig('Unsigned', 'grid', 4, 8, False),
    HogConfig('Coarse', 'grid', 2, 8, True),
    HogConfig('Global, aligned', 'grid', 1, 16, True, align=True),
    HogConfig('Rings (rotation-invariant)', 'rings', 3, 8, True),
]
CELL_CHOICES = (1, 2, 3, 4, 6, 8)        # must divide WINDOW
BIN_CHOICES = (4, 8, 9, 12, 16)


def bin_centers(cfg):
    """Centre angle (degrees) of each orientation bin."""
    period = 360.0 if cfg.signed else 180.0
    return (np.arange(cfg.n_bins) + 0.5) * period / cfg.n_bins


# ---------------------------------------------------------------- resampling

def rot(angle):
    """2x2 matrix rotating (x, y) offsets counter-clockwise on screen (y down)."""
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return np.array([[c, s], [-s, c]])


def sample(img, x, y, fill=1.0):
    """Bilinear sample of img at float coordinates; outside the image -> fill."""
    H, W = img.shape
    pad = np.pad(img, 1, constant_values=fill)
    x0, y0 = np.floor(x), np.floor(y)
    wx, wy = x - x0, y - y0
    x0, y0 = x0.astype(int) + 1, y0.astype(int) + 1     # indices into pad
    xi, xj = np.clip(x0, 0, W + 1), np.clip(x0 + 1, 0, W + 1)
    yi, yj = np.clip(y0, 0, H + 1), np.clip(y0 + 1, 0, H + 1)
    return ((1 - wy) * ((1 - wx) * pad[yi, xi] + wx * pad[yi, xj])
            + wy * ((1 - wx) * pad[yj, xi] + wx * pad[yj, xj]))


def extract_patch(img, center, size, angle=0.0, zoom=1.0):
    """size x size patch around center. The patch is rotated by `angle` and
    covers size * zoom image pixels, so a digit drawn at that angle and scale
    comes out upright at scale 1."""
    o = np.arange(size) - (size - 1) / 2
    ox, oy = np.meshgrid(o, o)
    R = rot(angle) * zoom
    x = center[0] + R[0, 0] * ox + R[0, 1] * oy
    y = center[1] + R[1, 0] * ox + R[1, 1] * oy
    return sample(img, x, y)


def warp(img, scale, angle):
    """One level of the image pyramid.

    Returns (warped, to_original). In `warped`, a digit that has scale `scale`
    and rotation `angle` in `img` appears upright at scale 1, so a fixed
    WINDOW-sized, upright detector can find it. to_original maps (N, 2) points
    (x, y) of `warped` back to `img`."""
    H, W = img.shape
    c = np.array([(W - 1) / 2, (H - 1) / 2])
    R = rot(angle) * scale
    corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]]) - c
    q = corners @ np.linalg.inv(R).T            # corners in warped coordinates
    w_out = int(np.ceil(np.ptp(q[:, 0]))) + 1
    h_out = int(np.ceil(np.ptp(q[:, 1]))) + 1
    c_out = np.array([(w_out - 1) / 2, (h_out - 1) / 2])

    def to_original(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        return c + (pts - c_out) @ R.T

    gx, gy = np.meshgrid(np.arange(w_out, dtype=float), np.arange(h_out, dtype=float))
    p = to_original(np.column_stack([gx.ravel(), gy.ravel()]))
    warped = sample(img, p[:, 0], p[:, 1]).reshape(h_out, w_out)
    return warped, to_original


# ---------------------------------------------------------------- descriptors

def _post(desc, cfg):
    """Dominant-orientation alignment (optional) on (..., n_regions * n_bins)."""
    if not cfg.align:
        return desc
    d = desc.reshape(desc.shape[:-1] + (cfg.n_regions, cfg.n_bins))
    k = d.sum(axis=-2).argmax(axis=-1)                       # (...,)
    idx = (np.arange(cfg.n_bins) + k[..., None, None]) % cfg.n_bins
    d = np.take_along_axis(d, np.broadcast_to(idx, d.shape), axis=-1)
    return d.reshape(desc.shape)


def _ring_geometry(cfg):
    """Radial direction (degrees) and ring one-hot (WINDOW, WINDOW, n) of the window."""
    o = np.arange(WINDOW) - (WINDOW - 1) / 2
    ox, oy = np.meshgrid(o, o)
    phi = np.degrees(np.arctan2(oy, ox)) % 360
    r = np.hypot(ox, oy) / (WINDOW / 2) * cfg.n_cells
    ring = (np.floor(r)[..., None] == np.arange(cfg.n_cells)).astype(float)
    return phi, ring


def _ring_descriptors(mag, ang, cfg, hd):
    """(N, WINDOW, WINDOW) magnitudes and angles -> (N, n_rings * n_bins)."""
    phi, ring = _ring_geometry(cfg)
    votes = hd.orientation_votes(mag, (ang - phi) % 360, cfg.n_bins, cfg.signed)
    h = np.einsum('nijb,ijr->nrb', votes, ring).reshape(len(mag), -1)
    return h / np.sqrt(np.sum(h ** 2, axis=1, keepdims=True) + HOG_EPS ** 2)


def patch_descriptor(patch, cfg, hd):
    """HoG of one (WINDOW + 2)-pixel patch using your E1-E3 (hd = hog_detector).
    The extra pixel on each side gives the window's edge pixels real gradients."""
    mag, ang = hd.image_gradients(patch)
    mag, ang = np.asarray(mag)[1:-1, 1:-1], np.asarray(ang)[1:-1, 1:-1]
    if cfg.layout == 'rings':
        return _post(_ring_descriptors(mag[None], ang[None], cfg, hd)[0], cfg)
    votes = hd.orientation_votes(mag, ang, cfg.n_bins, cfg.signed)
    return _post(np.asarray(hd.hog_descriptor(votes, cfg.n_cells), float), cfg)


def dense_windows(img, cfg, hd, stride=4):
    """HoG of every WINDOW x WINDOW window of img on a stride grid.

    Returns (centers (N, 2) as (x, y), descriptors (N, D)). Same numbers as
    hog_descriptor() on each window, but cell sums come from integral images."""
    H, W = img.shape
    if cfg.layout == 'rings':
        stride *= 2           # rings are computed window by window: slower
    x0s = np.arange(0, W - WINDOW + 1, stride)
    y0s = np.arange(0, H - WINDOW + 1, stride)
    if len(x0s) == 0 or len(y0s) == 0:
        return np.zeros((0, 2)), np.zeros((0, cfg.n_regions * cfg.n_bins))
    mag, ang = hd.image_gradients(img)
    gx, gy = np.meshgrid(x0s, y0s)
    centers = np.column_stack([gx.ravel(), gy.ravel()]) + (WINDOW - 1) / 2

    if cfg.layout == 'rings':
        out = []
        for chunk in np.array_split(np.arange(len(centers)), max(1, len(centers) // 512)):
            ys = (gy.ravel()[chunk, None] + np.arange(WINDOW))[:, :, None]
            xs = (gx.ravel()[chunk, None] + np.arange(WINDOW))[:, None, :]
            out.append(_ring_descriptors(mag[ys, xs], ang[ys, xs], cfg, hd))
        return centers, _post(np.concatenate(out), cfg)

    votes = np.asarray(hd.orientation_votes(mag, ang, cfg.n_bins, cfg.signed), float)
    S = np.zeros((H + 1, W + 1, cfg.n_bins))
    S[1:, 1:] = votes.cumsum(0).cumsum(1)
    n, c = cfg.n_cells, WINDOW // cfg.n_cells
    Xs = x0s[:, None] + np.arange(n + 1) * c                     # (Nx, n + 1)
    rows = []
    for y0 in y0s:
        Ys = y0 + np.arange(n + 1) * c
        G = S[Ys[None, :, None], Xs[:, None, :]]                 # (Nx, n+1, n+1, B)
        cells = G[:, 1:, 1:] - G[:, :-1, 1:] - G[:, 1:, :-1] + G[:, :-1, :-1]
        rows.append(cells.reshape(len(x0s), -1))
    h = np.concatenate(rows)
    h = h / np.sqrt(np.sum(h ** 2, axis=1, keepdims=True) + HOG_EPS ** 2)
    return centers, _post(h, cfg)


def digit_patch(scene_img, d, upright=True, angle_offset=0.0, zoom_factor=1.0, shift=(0, 0)):
    """(WINDOW + 2) patch on digit d of a scene; upright=True undoes its rotation and scale."""
    angle = (d['angle'] if upright else 0.0) + angle_offset
    zoom = (d['scale'] if upright else 1.0) * zoom_factor
    return extract_patch(scene_img, (d['cx'] + shift[0], d['cy'] + shift[1]),
                         WINDOW + 2, angle, zoom)


class Template:
    """The reference HoG of the train 3s, and how well a window matches it.

    A HoG is all non-negative, so any patch with strokes has a fairly high plain
    cosine similarity to any other: every digit looks ~60% like a 3. The match
    therefore compares what is *specific* to a 3: subtract the average digit's
    HoG, mu, from both the window x and the reference r, then take the cosine,

        match(x) = (x - mu) . (r - mu) / (|x - mu| |r - mu|),   in [-1, 1].

    r = mean HoG of the 3s, mu = mean HoG of all digits in the train scene,
    upright=True de-rotates and rescales each digit first."""

    def __init__(self, scene, cfg, hd, upright=True):
        def mean(digits):
            return np.mean([patch_descriptor(digit_patch(scene.image, d, upright), cfg, hd)
                            for d in digits], axis=0)
        self.ref = mean(scene.threes)          # the reference histogram the demo draws
        self.mu = mean(scene.digits)
        r = self.ref - self.mu
        self.direction = r / (np.linalg.norm(r) + 1e-12)

    def score(self, X):
        """(N, D) descriptors -> (N,) match, or (D,) -> float."""
        Z = np.asarray(X, float) - self.mu
        return Z @ self.direction / (np.linalg.norm(Z, axis=-1) + 1e-12)


# ---------------------------------------------------------------- classifier

class LinearSVM:
    """Linear SVM, f(x) = w . x + b, minimizing

        1/2 |w|^2 + sum_i C_i max(0, 1 - y_i f(x_i)),   y_i in {-1, +1},

    by dual coordinate descent (Hsieh et al. 2008, the method behind liblinear).
    C_i weights the classes so 3s and background count equally (slide 104)."""

    def __init__(self, C=10.0, epochs=30):
        self.C, self.epochs = C, epochs
        self.w, self.b = None, 0.0

    def fit(self, X, y, seed=0):
        X = np.column_stack([np.asarray(X, float), np.ones(len(X))])   # last weight = b
        y = np.asarray(y, float)
        n_pos, n_neg = np.sum(y > 0), np.sum(y < 0)
        C = self.C * np.where(y > 0, len(y) / (2 * n_pos), len(y) / (2 * n_neg))
        alpha, w = np.zeros(len(y)), np.zeros(X.shape[1])
        q = np.einsum('ij,ij->i', X, X)
        rng = np.random.default_rng(seed)
        for _ in range(self.epochs):
            for i in rng.permutation(len(y)):
                new = min(max(alpha[i] - (y[i] * (X[i] @ w) - 1) / q[i], 0.0), C[i])
                if new != alpha[i]:
                    w += (new - alpha[i]) * y[i] * X[i]
                    alpha[i] = new
        self.w, self.b = w[:-1], w[-1]
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.w + self.b


def sample_windows(scene, cfg, hd, augment_angles=(0,), n_random=2500, seed=0):
    """Training windows from the train scene.

    Positives: every 3, at each angle offset in augment_angles, 3 zooms and 5
    small shifts. Negatives: random windows whose box has IoU < 0.3 with every 3
    (uses your iou), plus windows on every other digit (the hard cases).
    Returns X (N, D), y (N,) in {-1, +1}, windows [(cx, cy, size, angle, label)]."""
    rng = np.random.default_rng(seed)
    X, y, wins = [], [], []

    def add(patch, label, cx, cy, size, angle):
        X.append(patch_descriptor(patch, cfg, hd))
        y.append(label)
        wins.append((cx, cy, size, angle, label))

    for d in scene.threes:
        for da in augment_angles:
            for dz in (0.9, 1.0, 1.1):
                for sh in ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)):
                    p = digit_patch(scene.image, d, True, da, dz, sh)
                    add(p, 1, d['cx'] + sh[0], d['cy'] + sh[1], WINDOW * d['scale'] * dz,
                        d['angle'] + da)
    three_boxes = scene.three_boxes()
    H, W = scene.image.shape
    while sum(1 for v in y if v < 0) < n_random:
        z = rng.uniform(0.85, 1.15)
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        box = np.array([cx, cy, cx, cy]) + np.array([-1, -1, 1, 1]) * BOX * z / 2
        if len(three_boxes) and np.max(hd.iou(box, three_boxes)) >= 0.3:
            continue
        a = rng.uniform(-30, 30)
        add(extract_patch(scene.image, (cx, cy), WINDOW + 2, a, z), -1, cx, cy, WINDOW * z, a)
    for d in scene.digits:
        if d['label'] == 3:
            continue
        for _ in range(3):
            da, dz = rng.uniform(-30, 30), rng.uniform(0.9, 1.1)
            add(digit_patch(scene.image, d, True, da, dz), -1, d['cx'], d['cy'],
                WINDOW * d['scale'] * dz, d['angle'] + da)
    return np.array(X), np.array(y), wins


# ---------------------------------------------------------------- evaluation

def evaluate(boxes, scores, gt_boxes, iou_fn, thresh=0.5):
    """Slide 44-45 metrics. A detection is a true positive if it overlaps a
    not-yet-matched ground-truth box with IoU > thresh (highest scores first).
    Returns dict(tp, fp, fn, precision, recall, ap, is_tp (per detection))."""
    boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    order = np.argsort(-scores)
    matched = np.zeros(len(gt_boxes), bool)
    is_tp = np.zeros(len(boxes), bool)
    for i in order:
        if not len(gt_boxes):
            break
        ious = np.where(matched, -1.0, iou_fn(boxes[i], gt_boxes))
        j = int(np.argmax(ious))
        if ious[j] > thresh:
            matched[j] = is_tp[i] = True
    tp_cum = np.cumsum(is_tp[order])
    precision = tp_cum / np.arange(1, len(order) + 1)
    recall = tp_cum / max(len(gt_boxes), 1)
    ap = np.mean([precision[recall >= r].max() if np.any(recall >= r) else 0.0
                  for r in np.linspace(0, 1, 11)])
    tp = int(is_tp.sum())
    return dict(tp=tp, fp=len(boxes) - tp, fn=len(gt_boxes) - tp,
                precision=tp / max(len(boxes), 1), recall=tp / max(len(gt_boxes), 1),
                ap=float(ap) if len(boxes) else 0.0, is_tp=is_tp)


# ---------------------------------------------------------------- self-tests

def _check(cond, why):
    if not bool(np.all(cond)):
        raise AssertionError(why)


def _test_e1(hd):
    x = np.tile(np.arange(5.0), (5, 1))                      # brightness grows to the right
    mag, ang = hd.image_gradients(x)
    mag, ang = np.asarray(mag), np.asarray(ang)
    _check(mag.shape == (5, 5) and ang.shape == (5, 5), 'mag and ang must have the image shape')
    _check(np.allclose(mag[1:-1, 1:-1], 2) and np.allclose(ang[1:-1, 1:-1] % 360, 0),
           'ramp to the right: expected mag 2 (= I[x+1] - I[x-1]) and angle 0')
    _check(np.allclose(mag[0], 0) and np.allclose(mag[:, -1], 0), 'border pixels must be 0')
    _, ang = hd.image_gradients(x.T)                          # grows downwards
    _check(np.allclose(np.asarray(ang)[1:-1, 1:-1], 90), 'ramp downwards: expected 90 degrees')
    _, ang = hd.image_gradients(-x)
    _check(np.allclose(np.asarray(ang)[1:-1, 1:-1], 180), 'ramp to the left: expected 180 degrees')
    mag, ang = hd.image_gradients(x + x.T)
    _check(np.allclose(np.asarray(mag)[2, 2], 2 * math.sqrt(2)) and np.isclose(np.asarray(ang)[2, 2], 45),
           'diagonal ramp: expected mag 2 sqrt(2), angle 45')
    _, ang = hd.image_gradients(-x - x.T)
    _check(np.isclose(np.asarray(ang)[2, 2], 225), 'angles must be in [0, 360)')


def _test_e2(hd):
    v = np.asarray(hd.orientation_votes(np.ones(4), np.array([22.5, 45.0, 0.0, 30.0]), 8, True))
    _check(v.shape == (4, 8), 'expected shape mag.shape + (n_bins,)')
    exp = np.zeros((4, 8))
    exp[0, 0] = 1                                   # exactly on the bin-0 centre
    exp[1, 0] = exp[1, 1] = 0.5                     # half way between bins 0 and 1
    exp[2, 7] = exp[2, 0] = 0.5                     # wraps around: 337.5 and 22.5
    exp[3, 0], exp[3, 1] = 1 - 7.5 / 45, 7.5 / 45
    _check(np.allclose(v, exp), f'signed 8 bins: expected\n{exp.round(3)}\ngot\n{v.round(3)}')
    v = np.asarray(hd.orientation_votes(np.array([[2.0]]), np.array([[200.0]]), 8, False))
    exp = np.zeros((1, 1, 8))
    exp[0, 0, 0], exp[0, 0, 1] = 2 * (1 - 8.75 / 22.5), 2 * 8.75 / 22.5
    _check(np.allclose(v, exp), 'unsigned: 200 deg is 20 deg; votes are weighted by mag')


def _test_e3(hd):
    votes = np.zeros((40, 40, 8))
    votes[3, 5, 0] = 4                              # cell (0, 0), bin 0
    votes[15, 27, 3] = 3                            # cell (row 1, col 2), bin 3
    d = np.asarray(hd.hog_descriptor(votes, 4), float)
    _check(d.shape == (128,), 'expected a flat vector of n_cells * n_cells * n_bins')
    exp = np.zeros((4, 4, 8))
    exp[0, 0, 0], exp[1, 2, 3] = 4, 3
    exp = exp.ravel() / math.sqrt(25 + HOG_EPS ** 2)
    _check(np.allclose(d, exp), 'order is (cell row, cell column, bin); normalize by sqrt(|h|^2 + eps^2)')
    # The demo's fast path must give the same numbers as your descriptor
    # (checked once E1 and E2 work).
    rng = np.random.default_rng(1)
    img = rng.uniform(size=(WINDOW + 4, WINDOW + 7))
    try:
        _, fast = dense_windows(img, PRESETS[0], hd, stride=4)
        mag, ang = hd.image_gradients(img)
        v = hd.orientation_votes(mag, ang, 8, True)
    except NotImplementedError:
        return
    slow = np.asarray(hd.hog_descriptor(np.asarray(v)[4:4 + WINDOW, 4:4 + WINDOW], 4))
    _check(np.allclose(fast[1 * 2 + 1], slow), 'differs from the integral-image fast path')


def _test_e4(hd):
    r = np.asarray(hd.iou(np.array([0, 0, 2, 2.]), np.array([1, 1, 3, 3.])), float)
    _check(r.size == 1 and np.isclose(r.ravel()[0], 1 / 7),
           f'IoU of [0,0,2,2] and [1,1,3,3] is 1/7 (one number); got {r}')
    r = np.asarray(hd.iou(np.array([0, 0, 2, 2.]),
                          np.array([[0, 0, 2, 2.], [5, 5, 6, 6], [1, 0, 3, 2]])))
    _check(r.shape == (3,) and np.allclose(r, [1, 0, 1 / 3]),
           'one box against (N, 4) boxes must return (N,) IoUs: expected [1, 0, 1/3]')


def _test_e5(hd):
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30.]])
    k = np.asarray(hd.nms(boxes, np.array([0.5, 0.9, 0.7]), 0.3))
    _check(list(k) == [1, 2], f'expected [1, 2] (best first, box 0 suppressed by box 1, '
                              f'IoU 0.68 by your iou); got {list(k)}')
    k = np.asarray(hd.nms(boxes, np.array([0.5, 0.9, 0.7]), 0.9))
    _check(list(k) == [1, 2, 0], 'with iou_thresh 0.9 nothing overlaps enough: expected [1, 2, 0]')


def _test_e6(hd):
    img = np.ones((60, 60))
    feats = lambda im, stride: (np.array([[29.5, 29.5]]), np.ones((1, 3)))
    boxes, scores = hd.detect(img, feats, lambda f: np.ones(len(f)), 0.5, hd.DetectConfig(
        scales=(1.0,), angles=(0,)))
    boxes = np.asarray(boxes).reshape(-1, 4)
    exp = [29.5 - BOX / 2, 29.5 - BOX / 2, 29.5 + BOX / 2, 29.5 + BOX / 2]
    _check(len(boxes) == 1 and np.allclose(boxes[0], exp, atol=1),
           f'one window at the centre should give box ~{exp}; got {boxes.round(1)}')


TESTS = [('E1', 'image_gradients', _test_e1), ('E2', 'orientation_votes', _test_e2),
         ('E3', 'hog_descriptor', _test_e3), ('E4', 'iou', _test_e4),
         ('E5', 'nms', _test_e5), ('E6', 'detect', _test_e6)]


def self_test(hd):
    """[(tag, name, status, detail)] with status 'pass', 'answer', 'todo', 'needs'
    (waiting on another TODO, named in detail) or 'fail'."""
    results = []
    for tag, name, test in TESTS:
        answers_used.clear()
        try:
            test(hd)
            status, detail = ('answer' if name in answers_used else 'pass'), ''
        except NotImplementedError as e:
            other = str(e).split(' ')[0]
            status, detail = ('todo', str(e)) if other == tag else ('needs', other)
        except Exception as e:                     # AssertionError or a bug in the code
            status, detail = 'fail', f'{type(e).__name__}: {e}'
        results.append((tag, name, status, detail))
    answers_used.clear()
    return results
