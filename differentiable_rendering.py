import cv2
import numpy as np

class Object3D:
    def __init__(self, vertices, faces):
        self.vertices = vertices  # (N, 3) array of vertex coordinates
        self.faces = faces        # (M, 3) array of indices into vertices

class Camera:
    def __init__(self, intrinsic_matrix, extrinsic_matrix):
        self.intrinsic_matrix = intrinsic_matrix  # (3, 3) camera intrinsic matrix
        self.extrinsic_matrix = extrinsic_matrix  # (4, 4) camera extrinsic matrix

class Image:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = np.ones((height, width, 3), dtype=np.uint8) * 255  # RGB image

def traditional_renderer(obj: Object3D, camera: Camera, image: Image):
    """A non-differentiable silhouette renderer that projects 3D vertices to 2D image space and rasterizes the faces."""

    # Transform the object to image space
    pts_3d = obj.vertices  # (N, 3)
    pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))  # Convert to homogeneous coordinates
    pts_cam = (camera.extrinsic_matrix @ pts_hom.T).T  # Transform to camera coordinates
    pts_img = (camera.intrinsic_matrix @ pts_cam[:, :3].T).T  # Project to image plane
    pts_img /= pts_img[:, 2:3]  # Normalize by depth

    # Sort the triangles by depth (z-index), far to near
    tri_depths = np.mean(pts_cam[obj.faces], axis=1)  # Average depth of each triangle
    sorted_indices = np.argsort(-tri_depths[:, 2])  # Sort by z-value in reverse order (far to near)
    tris = obj.faces[sorted_indices]

    # Rasterize the triangles onto the image
    for tri in tris:
        pts = pts_img[tri][:, :2].astype(np.int32)  # Get 2D points of the triangle
        cv2.fillPoly(image.data, [pts], (0, 0, 0))  # Fill the triangle with black color

    return image

def differentiable_renderer(obj: Object3D, camera: Camera, image: Image):
    """[DIB-R] Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer (https://arxiv.org/abs/1908.01210).
    """
    pass

