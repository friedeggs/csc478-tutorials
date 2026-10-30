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
    """A differentiable renderer that computes a soft silhouette of the 3D object
    in the image space. Instead of hard rasterization, it uses a soft contribution
    based on barycentric coordinates to allow gradients to flow through the
    rendering process.
    """

    # Project the triangles to image space, as before
    pts_3d = obj.vertices  # (N, 3)
    pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))  # Convert to homogeneous coordinates
    pts_cam = (camera.extrinsic_matrix @ pts_hom.T).T  # Transform to camera coordinates
    pts_img = (camera.intrinsic_matrix @ pts_cam[:, :3].T).T  # Project to image plane
    pts_img /= pts_img[:, 2:3]

    tri_img = pts_img[obj.faces][:, :, :2]  # (M, 3, 2) array of triangle vertices in image space

    image.data = np.ones((image.height, image.width, 3), dtype=np.float32) * 255  # Reset to white background
    
    # Loop over the image pixels
    for i in range(image.height):
        for j in range(image.width):
            # Compute each triangle's contribution to the pixel (i, j) using barycentric coordinates
            for tri in tri_img:
                # Compute barycentric coordinates
                A = np.array([[tri[0, 0], tri[1, 0], tri[2, 0]],
                              [tri[0, 1], tri[1, 1], tri[2, 1]],
                              [1, 1, 1]])
                b = np.array([j, i, 1])
                try:
                    bary_coords = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue  # Skip degenerate triangles

                if np.all(bary_coords >= 0):  # If the pixel is inside the triangle
                    # For simplicity, we just set the pixel to black if it's inside any triangle
                    image.data[i, j] = [0, 0, 0]
                    break
                else:
                    # If the pixel is outside the triangle, we compute a soft contribution based on distance.
                    # The background is white and the silhouette black, so the contribution darkens the pixel.
                    soft = np.clip(1 - np.linalg.norm(bary_coords), 0, 1)
                    image.data[i, j] = np.maximum(image.data[i, j] - soft * 255, 0)  # Soft contribution

    return image

def fast_differentiable_renderer(obj: Object3D, camera: Camera, image: Image):
    """The same image as differentiable_renderer, fast enough to run in real time.

    Instead of looping over pixels and solving for each pixel's barycentric
    coordinates, loop over triangles only: a triangle's barycentric coordinates
    are an affine function of the pixel, inv(A) @ (j, i, 1), so all its pixels
    are found at once. A pixel inside any triangle is black; any other pixel
    ends up at 255 minus 255 times the sum of its soft contributions (at least
    0), whatever the order of the triangles. The soft contribution is zero
    unless every barycentric coordinate is above -1/3, which only happens inside
    the triangle doubled in size about its centroid, so only that box is visited.
    """

    # Project the triangles to image space, as before
    pts_3d = obj.vertices  # (N, 3)
    pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))  # Convert to homogeneous coordinates
    pts_cam = (camera.extrinsic_matrix @ pts_hom.T).T  # Transform to camera coordinates
    pts_img = (camera.intrinsic_matrix @ pts_cam[:, :3].T).T  # Project to image plane
    pts_img /= pts_img[:, 2:3]

    tri_img = pts_img[obj.faces][:, :, :2]  # (M, 3, 2) array of triangle vertices in image space

    inside = np.zeros((image.height, image.width), dtype=bool)  # Pixels inside any triangle
    soft_sum = np.zeros((image.height, image.width), dtype=np.float32)  # Sum of the soft contributions

    for tri in tri_img:
        A = np.array([[tri[0, 0], tri[1, 0], tri[2, 0]],
                      [tri[0, 1], tri[1, 1], tri[2, 1]],
                      [1, 1, 1]])
        try:
            A_inv = np.linalg.inv(A)  # Barycentric coordinates of pixel (i, j) are A_inv @ (j, i, 1)
        except np.linalg.LinAlgError:
            continue  # Skip degenerate triangles

        # Pixels where the soft contribution can be nonzero: the triangle doubled about its centroid
        centroid = tri.mean(axis=0)
        big = centroid + 2 * (tri - centroid)
        j0, i0 = np.maximum(np.ceil(big.min(axis=0)), 0).astype(int)
        j1, i1 = np.minimum(np.floor(big.max(axis=0)), (image.width - 1, image.height - 1)).astype(int)
        if j1 < j0 or i1 < i0:
            continue  # Off the image
        jj, ii = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1))

        bary_coords = A_inv[:, 0, None, None] * jj + A_inv[:, 1, None, None] * ii + A_inv[:, 2, None, None]  # (3, h, w)
        tri_inside = np.all(bary_coords >= 0, axis=0)
        soft = np.clip(1 - np.sqrt(np.sum(bary_coords ** 2, axis=0)), 0, 1)

        inside[i0:i1 + 1, j0:j1 + 1] |= tri_inside
        soft_sum[i0:i1 + 1, j0:j1 + 1] += np.where(tri_inside, 0, soft)

    gray = np.where(inside, 0, np.maximum(255 - 255 * soft_sum, 0))
    image.data = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)  # White background, black silhouette

    return image
