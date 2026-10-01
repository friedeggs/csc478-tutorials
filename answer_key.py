import numpy as np
def compute_homography(src, tgt):
    A = []
    for (x, y), (xp, yp) in zip(src, tgt):
        A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp, -xp])
        A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp, -yp])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]