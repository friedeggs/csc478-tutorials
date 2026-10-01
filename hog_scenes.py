"""Scenes for Tutorial 4: "I spy" images of ~100 cluttered MNIST digits.

    load_train()          the train scene (assets/hog_train.png + labels)
    load_blind_test()     the blind test scene (assets/hog_test.bin, never shown)
    make_scene(seed)      a new test scene from the MNIST test split (downloads
                          MNIST into data/mnist/ the first time, ~11 MB)

Each digit is rotated by up to +-30 degrees, scaled by 0.85-1.15 and dropped at
a random position. Digits may overlap, but a 3 is only lightly overlapped so
every 3 stays findable. Ground truth per digit: label, centre (cx, cy), angle,
scale, a square box of side ~BOX * scale (used for IoU) and the tight ink box.
"""
import gzip
import io
import json
import math
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame  # noqa: E402  (PNG reading and writing only)

from hog_lib import BOX, rot, sample  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'assets'
MNIST_DIR = HERE / 'data' / 'mnist'
MNIST_MIRRORS = ['https://ossci-datasets.s3.amazonaws.com/mnist/',
                 'https://storage.googleapis.com/cvdf-datasets/mnist/']
MNIST_FILES = {'train': ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
               'test': ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')}

SIZE = (720, 480)          # scene width, height
BASE_SCALE = BOX / 20      # MNIST ink spans ~20 px, so a scale-1 digit is ~BOX px
MAX_ANGLE = 30
SCALE_RANGE = (0.85, 1.15)
OVERLAP_ANY = 0.35         # max intersection / smaller ink box, any two digits
OVERLAP_THREE = 0.10       # ... when one of them is a 3
TEST_KEY = 0x478           # obfuscation key of assets/hog_test.bin (not a secret)


@dataclass
class Scene:
    name: str
    image: np.ndarray            # (H, W) float32, 1 = paper, 0 = ink
    digits: list                 # dicts: label, cx, cy, angle, scale, box, ink_box
    revealable: bool = True
    seed: int = None
    meta: dict = field(default_factory=dict)

    @property
    def threes(self):
        return [d for d in self.digits if d['label'] == 3]

    def three_boxes(self):
        return np.array([d['box'] for d in self.threes], float).reshape(-1, 4)


# ---------------------------------------------------------------- MNIST

def load_mnist(split):
    """(images (N, 28, 28) uint8, white ink on black; labels (N,))."""
    MNIST_DIR.mkdir(parents=True, exist_ok=True)
    arrays = []
    for name in MNIST_FILES[split]:
        path = MNIST_DIR / name
        if not path.exists():
            for mirror in MNIST_MIRRORS:
                try:
                    print(f'Downloading {mirror}{name}')
                    urllib.request.urlretrieve(mirror + name, path)
                    break
                except OSError as e:
                    print(f'  failed: {e}')
            else:
                raise RuntimeError(f'Could not download MNIST file {name}')
        with gzip.open(path, 'rb') as f:
            data = f.read()
        magic, n = int.from_bytes(data[:4], 'big'), int.from_bytes(data[4:8], 'big')
        if magic == 2051:
            arrays.append(np.frombuffer(data, np.uint8, offset=16).reshape(n, 28, 28))
        else:
            arrays.append(np.frombuffer(data, np.uint8, offset=8))
    return arrays[0], arrays[1]


# ---------------------------------------------------------------- generator

def _overlap(a, b):
    """Intersection of boxes a and b over the area of the smaller one."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return w * h / min(area(a), area(b))


def make_scene(seed, split='test', name=None, size=SIZE, mnist=None):
    """A random scene: 8-12 threes and about as many of each other digit."""
    rng = np.random.default_rng(seed)
    images, labels = mnist if mnist is not None else load_mnist(split)
    W, H = size
    n_three = int(rng.integers(8, 13))
    todo = [3] * n_three
    for lab in (0, 1, 2, 4, 5, 6, 7, 8, 9):
        todo += [lab] * max(4, n_three + int(rng.integers(-2, 3)))
    rest = todo[n_three:]
    rng.shuffle(rest)
    todo = todo[:n_three] + rest                      # place the 3s first

    ink = np.zeros((H, W))
    digits, skipped = [], 0
    for lab in todo:
        idx = int(rng.choice(np.flatnonzero(labels == lab)))
        src = images[idx].astype(float) / 255
        rows, cols = np.nonzero(src > 0)
        r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
        u0 = np.array([(c0 + c1) / 2, (r0 + r1) / 2])        # ink centre in the 28 px frame
        half = np.array([(c1 - c0 + 1) / 2, (r1 - r0 + 1) / 2])
        side = max(c1 - c0 + 1, r1 - r0 + 1)
        for _ in range(400):
            angle = float(rng.uniform(-MAX_ANGLE, MAX_ANGLE))
            scale = float(rng.uniform(*SCALE_RANGE))
            k = BASE_SCALE * scale
            R = rot(angle) * k
            ext = np.abs(R) @ half                              # half-size of the rotated ink box
            if np.any(2 * ext + 4 > (W, H)):
                continue
            cx = float(rng.uniform(ext[0] + 2, W - ext[0] - 2))
            cy = float(rng.uniform(ext[1] + 2, H - ext[1] - 2))
            est = (cx - ext[0], cy - ext[1], cx + ext[0], cy + ext[1])
            if all(_overlap(est, d['ink_box'])
                   <= (OVERLAP_THREE if 3 in (lab, d['label']) else OVERLAP_ANY)
                   for d in digits):
                break
        else:
            skipped += 1
            continue
        # Render: canvas point p shows source point u0 + R^-1 (p - c).
        x0, y0 = int(max(0, cx - ext[0] - 2)), int(max(0, cy - ext[1] - 2))
        x1, y1 = int(min(W, cx + ext[0] + 3)), int(min(H, cy + ext[1] + 3))
        gx, gy = np.meshgrid(np.arange(x0, x1, dtype=float), np.arange(y0, y1, dtype=float))
        Ri = np.linalg.inv(R)
        dx, dy = gx - cx, gy - cy
        val = sample(src, u0[0] + Ri[0, 0] * dx + Ri[0, 1] * dy,
                     u0[1] + Ri[1, 0] * dx + Ri[1, 1] * dy, fill=0.0)
        ink[y0:y1, x0:x1] = np.maximum(ink[y0:y1, x0:x1], val)
        ys, xs = np.nonzero(val > 0.2)
        ink_box = [float(x0 + xs.min()), float(y0 + ys.min()),
                   float(x0 + xs.max() + 1), float(y0 + ys.max() + 1)]
        L = side * k / 2
        digits.append(dict(label=lab, cx=round(cx, 2), cy=round(cy, 2), angle=round(angle, 2),
                           scale=round(scale, 4), box=[round(v, 2) for v in (cx - L, cy - L, cx + L, cy + L)],
                           ink_box=ink_box, mnist_index=idx))
    if skipped:
        print(f'make_scene({seed}): could not place {skipped} digits')
    img = (1 - np.clip(ink, 0, 1)).astype(np.float32)
    img = np.round(img * 255) / 255                     # what a PNG round trip gives
    return Scene(name or f'Test #{seed}', img.astype(np.float32), digits, True, seed,
                 dict(split=split))


# ---------------------------------------------------------------- files

def save_png(path, img):
    """Save (H, W) float in [0, 1] or (H, W, 3) uint8 as PNG."""
    if img.ndim == 2:
        img = np.repeat((np.clip(img, 0, 1) * 255).round().astype(np.uint8)[..., None], 3, axis=2)
    pygame.image.save(pygame.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1))),
                      str(path))


def load_png(path):
    """(H, W) float32 in [0, 1]."""
    surf = pygame.image.load(str(path))
    return (pygame.surfarray.array3d(surf)[..., 0].T / 255).astype(np.float32)


def save_train(scene, png=ASSETS / 'hog_train.png', labels=ASSETS / 'hog_train_labels.json'):
    save_png(png, scene.image)
    with open(labels, 'w') as f:
        json.dump(dict(seed=scene.seed, split=scene.meta.get('split'), digits=scene.digits), f, indent=1)


def load_train():
    with open(ASSETS / 'hog_train_labels.json') as f:
        lab = json.load(f)
    return Scene('Train', load_png(ASSETS / 'hog_train.png'), lab['digits'], True, lab['seed'],
                 dict(split=lab['split']))


def _keystream(n, which):
    rng = np.random.default_rng([TEST_KEY, which])
    return rng.permutation(n), rng.integers(0, 256, n, dtype=np.uint8)


def save_blind_test(scene, path=ASSETS / 'hog_test.bin'):
    """Pixels shuffled and XORed, boxes of the 3s XORed: not viewable as an image."""
    pix = (scene.image * 255).round().astype(np.uint8).ravel()
    perm, key = _keystream(pix.size, 0)
    raw = np.array([[d['cx'], d['cy'], *d['box']] for d in scene.threes], np.float64).tobytes()
    _, kb = _keystream(len(raw), 1)
    buf = io.BytesIO()
    np.savez_compressed(buf, p=pix[perm] ^ key, b=np.frombuffer(raw, np.uint8) ^ kb,
                        shape=np.array(scene.image.shape))
    Path(path).write_bytes(buf.getvalue())


def load_blind_test(path=ASSETS / 'hog_test.bin'):
    """The blind test scene. Only its 3s are labelled, and it is not revealable."""
    z = np.load(io.BytesIO(Path(path).read_bytes()))
    shape = tuple(z['shape'])
    perm, key = _keystream(int(np.prod(shape)), 0)
    pix = np.empty(perm.size, np.uint8)
    pix[perm] = z['p'] ^ key
    _, kb = _keystream(z['b'].size, 1)
    rows = np.frombuffer((z['b'] ^ kb).tobytes(), np.float64).reshape(-1, 6)
    digits = [dict(label=3, cx=r[0], cy=r[1], angle=None, scale=None, box=list(r[2:]))
              for r in rows]
    return Scene('Blind test', (pix.reshape(shape) / 255).astype(np.float32), digits, False)


def draw_boxes(img, boxes, color=(22, 163, 74), width=2):
    """(H, W) float image -> (H, W, 3) uint8 with box outlines."""
    out = np.repeat((np.clip(img, 0, 1) * 255).round().astype(np.uint8)[..., None], 3, axis=2)
    H, W = img.shape
    for x0, y0, x1, y1 in np.asarray(boxes).reshape(-1, 4):
        x0, y0 = int(max(0, math.floor(x0))), int(max(0, math.floor(y0)))
        x1, y1 = int(min(W - 1, math.ceil(x1))), int(min(H - 1, math.ceil(y1)))
        out[y0:y0 + width, x0:x1 + 1] = color
        out[y1 - width + 1:y1 + 1, x0:x1 + 1] = color
        out[y0:y1 + 1, x0:x0 + width] = color
        out[y0:y1 + 1, x1 - width + 1:x1 + 1] = color
    return out
