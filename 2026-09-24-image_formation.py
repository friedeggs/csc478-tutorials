import cv2
import numpy as np
from tqdm import tqdm

from utils import bilinear_sample

np.random.seed(0)

IMG_FILENAME = 'assets/CS478_Poster_Warped.png'
DATA_FILENAME = 'assets/CS478_Poster_Warped_corners.txt'
TARGET_HEIGHT = 500
TARGET_WIDTH = 360


def compute_homography(src, tgt):
    """src, tgt: (4, 2) arrays of (x, y). Returns H with tgt ~ H @ src."""
    print("Source coordinates:")
    print(src)
    print("Target coordinates:")
    print(tgt)
    # TODO implement this
    return np.eye(3)


def inverse_warp(source_image, target_image, H):
    print("Warping the image...")
    target_height, target_width, _ = target_image.shape
    H_inv = np.linalg.inv(H)

    for i in tqdm(range(target_height * target_width), desc="Warping"):
        v = i // target_width
        u = i % target_width

        # Target pixel
        target = np.array([u, v, 1])

        # Target -> source
        source = H_inv @ target

        # Homogeneous divide
        x = source[0] / source[2]
        y = source[1] / source[2]

        # Sample source image
        target_image[v, u] = bilinear_sample(source_image, x, y)


def warp_into_quad(source_image, target_image, H, quad):
    """Like inverse_warp, but only fills target pixels inside the (4, 2)
    quad, all at once instead of one pixel at a time."""
    mask = np.zeros(target_image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad, 1)
    v, u = np.nonzero(mask)

    # Target -> source, with the homogeneous divide
    source = np.linalg.inv(H) @ np.stack([u, v, np.ones_like(u)])
    x = source[0] / source[2]
    y = source[1] / source[2]

    target_image[v, u] = bilinear_sample(source_image, x, y)


def homography_demo():
    img = cv2.imread(IMG_FILENAME, cv2.IMREAD_COLOR)
    source_coords = np.loadtxt(DATA_FILENAME)
    target_coords = np.array([
        [0, 0],
        [TARGET_WIDTH, 0],
        [TARGET_WIDTH, TARGET_HEIGHT],
        [0, TARGET_HEIGHT],
    ])

    H = compute_homography(source_coords, target_coords)

    # uint8, like the source: cv2.imshow expects floats in [0, 1], so a
    # float image with values in [0, 255] displays as almost pure white.
    target_img = np.full((TARGET_HEIGHT, TARGET_WIDTH, 3), 255, dtype=np.uint8)
    inverse_warp(img, target_img, H)

    cv2.imshow('Source Image', img)
    cv2.waitKey(0)

    cv2.imshow('Target Image', target_img)
    cv2.waitKey(0)


SCENE_RADIUS = 10
FOCAL_LENGTH = 400  # pixels; wide angle so the hemisphere visibly curves

# Unit cube centred at the origin: front face 0-3 (z = -0.5, normal -z),
# back face 4-7.
CUBOID_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
]) - 0.5


def camera_scene():
    img_height = 1080
    img_width = 1920
    img = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    # Poster photo and its four corners (TL, TR, BR, BL), the homography source.
    poster = cv2.imread(IMG_FILENAME, cv2.IMREAD_COLOR)
    poster_corners = np.loadtxt(DATA_FILENAME)

    # Camera at the origin looking down +z (x right, y down).
    K = np.array([
        [FOCAL_LENGTH, 0, img_width / 2],
        [0, FOCAL_LENGTH, img_height / 2],
        [0, 0, 1],
    ])

    # Initialize scene with cuboids on a grid of directions covering the
    # hemisphere in front of the camera (z > 0), front face towards the camera.
    cuboids = []
    for azimuth in np.radians(np.arange(-60, 61, 30)):
        for elevation in np.radians(np.arange(-30, 31, 30)):
            direction = np.array([
                np.cos(elevation) * np.sin(azimuth),
                -np.sin(elevation),  # y points down, so up is -y
                np.cos(elevation) * np.cos(azimuth),
            ])
            # Poster-shaped front face (width:height = 360:500), 4-5 units tall.
            height = np.random.uniform(4, 5)
            size = [height * TARGET_WIDTH / TARGET_HEIGHT, height, 1]
            # Local z points away from the camera, so the front face normal (-z)
            # points back at it. Local x stays horizontal.
            x_axis = np.cross([0, 1, 0], direction)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(direction, x_axis)
            R = np.column_stack([x_axis, y_axis, direction])
            cuboids.append((CUBOID_CORNERS * size) @ R.T + SCENE_RADIUS * direction)

    # Project them into the image
    for X in cuboids:
        if np.any(X[:, 2] <= 0.1):  # skip cuboids reaching behind the camera
            continue
        x = X @ K.T
        pixels = x[:, :2] / x[:, 2:]

        # Front face corners 0-3 run TL, TR, BR, BL on screen, like the poster's.
        face = pixels[:4]
        H = compute_homography(poster_corners, face)
        quad = np.round(face).astype(np.int32)  # fillConvexPoly needs int32
        warp_into_quad(poster, img, H, quad)

    cv2.imshow('Scene', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    homography_demo()
    # camera_scene() # TODO uncomment this