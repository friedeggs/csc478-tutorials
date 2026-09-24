import numpy as np

def bilinear_sample(image, x, y):
    """Sample image at floating-point coordinates using bilinear interpolation."""
    height, width = image.shape[:2]

    # Integer neighbors
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp coordinates to the image boundaries.
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    # Interpolation weights
    wx = np.clip(x - x0, 0, 1)
    wy = np.clip(y - y0, 0, 1)

    # Four neighboring pixels
    Ia = image[y0, x0].astype(np.float32)
    Ib = image[y0, x1].astype(np.float32)
    Ic = image[y1, x0].astype(np.float32)
    Id = image[y1, x1].astype(np.float32)

    # Bilinear interpolation
    result = (
        (1 - wx)[..., None] * (1 - wy)[..., None] * Ia
        + wx[..., None] * (1 - wy)[..., None] * Ib
        + (1 - wx)[..., None] * wy[..., None] * Ic
        + wx[..., None] * wy[..., None] * Id
    )

    return result