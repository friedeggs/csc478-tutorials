from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ASSETS = Path(__file__).parent / 'assets'

class Object3D(torch.nn.Module):
    """A triangle mesh whose vertex coordinates are PyTorch parameters, so an optimizer can move them."""

    def __init__(self, vertices, faces):
        super().__init__()
        self.vertices = torch.nn.Parameter(torch.as_tensor(vertices, dtype=torch.float32))  # (N, 3) vertex coordinates, optimizable
        self.register_buffer('faces', torch.as_tensor(faces, dtype=torch.long))  # (M, 3) indices into vertices, fixed

class Camera:
    def __init__(self, intrinsic_matrix, extrinsic_matrix):
        self.intrinsic_matrix = intrinsic_matrix  # (3, 3) camera intrinsic matrix
        self.extrinsic_matrix = extrinsic_matrix  # (4, 4) camera extrinsic matrix

class Image:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = np.ones((height, width, 3), dtype=np.uint8) * 255  # RGB image

def load_obj(path):
    """Load a Wavefront .obj file as an Object3D, splitting polygons into triangles."""
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue  # Skip blank lines
        if parts[0] == 'v':  # Vertex: v x y z
            vertices.append([float(p) for p in parts[1:4]])
        elif parts[0] == 'f':  # Face: f a b c ..., 1-based indices, possibly a/t/n
            idx = [int(p.split('/')[0]) - 1 for p in parts[1:]]
            faces += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]  # Fan triangulation
    return Object3D(np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64))

def load_images(folder):
    """Load every .png in a folder, in filename order, as a list of Image objects (RGB, uint8)."""
    images = []
    for path in sorted(Path(folder).glob('*.png')):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)  # OpenCV loads colour images as BGR
        if bgr is None:
            raise IOError(f'could not read {path}')
        image = Image(bgr.shape[1], bgr.shape[0])
        image.data = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(image)
    return images

def load_cameras(path):
    """Load the cameras saved with a set of target silhouettes (cameras.npz: K and E per view) as a list of Camera objects."""
    data = np.load(path)
    return [Camera(K, E) for K, E in zip(data['K'], data['E'])]

def traditional_renderer(obj: Object3D, camera: Camera, image: Image):
    """A non-differentiable silhouette renderer that projects 3D vertices to 2D image space and rasterizes the faces."""

    # Transform the object to image space
    pts_3d = obj.vertices.detach().numpy()  # (N, 3), a NumPy copy: no gradients
    faces = obj.faces.numpy()  # (M, 3)
    pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))  # Convert to homogeneous coordinates
    pts_cam = (camera.extrinsic_matrix @ pts_hom.T).T  # Transform to camera coordinates
    pts_img = (camera.intrinsic_matrix @ pts_cam[:, :3].T).T  # Project to image plane
    pts_img /= pts_img[:, 2:3]  # Normalize by depth

    # Sort the triangles by depth (z-index), far to near
    tri_depths = np.mean(pts_cam[faces], axis=1)  # Average depth of each triangle
    sorted_indices = np.argsort(-tri_depths[:, 2])  # Sort by z-value in reverse order (far to near)
    tris = faces[sorted_indices]

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
    pts_3d = obj.vertices.detach().numpy()  # (N, 3), a NumPy copy: no gradients flow back to the mesh yet
    pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))  # Convert to homogeneous coordinates
    pts_cam = (camera.extrinsic_matrix @ pts_hom.T).T  # Transform to camera coordinates
    pts_img = (camera.intrinsic_matrix @ pts_cam[:, :3].T).T  # Project to image plane
    pts_img /= pts_img[:, 2:3]

    tri_img = pts_img[obj.faces.numpy()][:, :, :2]  # (M, 3, 2) array of triangle vertices in image space

    image.data = np.ones((image.height, image.width, 3), dtype=np.float32) * 255  # Reset to white background
    
    # Loop over the image pixels
    for i in tqdm(range(image.height), desc="Rendering"):
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
    pts_3d = obj.vertices.detach().numpy()  # (N, 3), a NumPy copy: no gradients
    pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))  # Convert to homogeneous coordinates
    pts_cam = (camera.extrinsic_matrix @ pts_hom.T).T  # Transform to camera coordinates
    pts_img = (camera.intrinsic_matrix @ pts_cam[:, :3].T).T  # Project to image plane
    pts_img /= pts_img[:, 2:3]

    tri_img = pts_img[obj.faces.numpy()][:, :, :2]  # (M, 3, 2) array of triangle vertices in image space

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

def fast_differentiable_renderer_pytorch(obj: Object3D, camera: Camera, image: Image):
    """fast_differentiable_renderer in PyTorch: the same image, but differentiable.

    Every step is a PyTorch operation on obj.vertices, so image.data, an (H, W, 3)
    float32 tensor, can be put in a loss and backpropagated to the vertices.
    Gradients come from the soft contributions: pixels inside the silhouette are
    a flat 0, and so are pixels darkened to 0, so neither passes gradients back.
    """

    # Project the triangles to image space, as before, in float64 like the NumPy renderers
    pts_3d = obj.vertices.double()  # (N, 3)
    K = torch.as_tensor(camera.intrinsic_matrix, dtype=torch.float64)
    E = torch.as_tensor(camera.extrinsic_matrix, dtype=torch.float64)
    pts_hom = torch.cat((pts_3d, torch.ones(pts_3d.shape[0], 1, dtype=torch.float64)), dim=1)  # Convert to homogeneous coordinates
    pts_cam = (E @ pts_hom.T).T  # Transform to camera coordinates
    pts_img = (K @ pts_cam[:, :3].T).T  # Project to image plane
    pts_img = pts_img / pts_img[:, 2:3]

    tri_img = pts_img[obj.faces][:, :, :2]  # (M, 3, 2) tensor of triangle vertices in image space

    # Every triangle's A at once: (M, 3, 3), its columns the triangle's corners (x, y, 1)
    A = torch.cat((tri_img.transpose(1, 2), torch.ones(len(tri_img), 1, 3, dtype=torch.float64)), dim=1)
    A_inv, info = torch.linalg.inv_ex(A)  # Barycentric coordinates of pixel (i, j) are A_inv @ (j, i, 1)
    tri_np = tri_img.detach().numpy()  # Only for finding each triangle's box of pixels

    ii, jj = torch.meshgrid(torch.arange(image.height, dtype=torch.float64),
                            torch.arange(image.width, dtype=torch.float64), indexing='ij')  # Pixel rows and columns
    inside = torch.zeros(image.height, image.width, dtype=torch.bool)  # Pixels inside any triangle
    soft_sum = torch.zeros(image.height, image.width, dtype=torch.float64)  # Sum of the soft contributions

    for t, tri in enumerate(tri_np):
        if info[t] != 0:
            continue  # Skip degenerate triangles

        # Pixels where the soft contribution can be nonzero: the triangle doubled about its centroid
        centroid = tri.mean(axis=0)
        big = centroid + 2 * (tri - centroid)
        j0, i0 = np.maximum(np.ceil(big.min(axis=0)), 0).astype(int)
        j1, i1 = np.minimum(np.floor(big.max(axis=0)), (image.width - 1, image.height - 1)).astype(int)
        if j1 < j0 or i1 < i0:
            continue  # Off the image
        box = (slice(i0, i1 + 1), slice(j0, j1 + 1))

        M = A_inv[t]
        bary_coords = M[:, 0, None, None] * jj[box] + M[:, 1, None, None] * ii[box] + M[:, 2, None, None]  # (3, h, w)
        tri_inside = torch.all(bary_coords >= 0, dim=0)
        soft = torch.clamp(1 - torch.sqrt(torch.sum(bary_coords ** 2, dim=0)), 0, 1)

        inside[box] |= tri_inside
        soft_sum[box] += torch.where(tri_inside, 0, soft)

    gray = torch.where(inside, 0, torch.clamp(255 - 255 * soft_sum, min=0))
    image.data = gray[:, :, None].repeat(1, 1, 3).float()  # White background, black silhouette

    return image

def batched_differentiable_renderer_pytorch(obj: Object3D, camera: Camera, image: Image, budget=1_000_000):
    """fast_differentiable_renderer_pytorch, with the triangles done in batches: the same image, faster.

    Looping over the triangles one at a time in PyTorch costs about the same for every triangle,
    however few pixels it covers, which is slow for a mesh of thousands of small triangles. Here the
    triangles are sorted by the size of their box of pixels and done many at a time: each batch pads
    its boxes to the largest one (at most budget numbers per batch), finds all their barycentric
    coordinates at once, and adds them into the image with one index_add.
    """

    # Project the triangles to image space, as before
    pts_3d = obj.vertices.double()  # (N, 3)
    K = torch.as_tensor(camera.intrinsic_matrix, dtype=torch.float64)
    E = torch.as_tensor(camera.extrinsic_matrix, dtype=torch.float64)
    pts_hom = torch.cat((pts_3d, torch.ones(pts_3d.shape[0], 1, dtype=torch.float64)), dim=1)  # Convert to homogeneous coordinates
    pts_cam = (E @ pts_hom.T).T  # Transform to camera coordinates
    pts_img = (K @ pts_cam[:, :3].T).T  # Project to image plane
    pts_img = pts_img / pts_img[:, 2:3]

    tri_img = pts_img[obj.faces][:, :, :2]  # (M, 3, 2) tensor of triangle vertices in image space
    A = torch.cat((tri_img.transpose(1, 2), torch.ones(len(tri_img), 1, 3, dtype=torch.float64)), dim=1)
    A_inv, info = torch.linalg.inv_ex(A)  # Barycentric coordinates of pixel (i, j) are A_inv @ (j, i, 1)

    # Every triangle's box of pixels: the triangle doubled about its centroid, as before
    H, W = image.height, image.width
    tri_np = tri_img.detach().numpy()
    centroid = tri_np.mean(axis=1, keepdims=True)
    big = centroid + 2 * (tri_np - centroid)
    j0, i0 = np.maximum(np.ceil(big.min(axis=1)), 0).astype(int).T
    j1, i1 = np.minimum(np.floor(big.max(axis=1)), (W - 1, H - 1)).astype(int).T
    h, w = i1 - i0 + 1, j1 - j0 + 1
    keep = np.nonzero((info.numpy() == 0) & (h > 0) & (w > 0))[0]  # Skip degenerate triangles and those off the image
    keep = keep[np.argsort(h[keep] * w[keep], kind='stable')]  # Small boxes together, so padding wastes little

    inside = torch.zeros(H * W, dtype=torch.float64)  # How many triangles each pixel is inside
    soft_sum = torch.zeros(H * W, dtype=torch.float64)  # Sum of the soft contributions
    start = 0
    while start < len(keep):
        end = start + 1  # Grow the batch while its padded boxes fit the budget
        while end < len(keep) and (end + 1 - start) * h[keep[end]] * w[keep[end]] <= budget:
            end += 1
        t = keep[start:end]
        ii = i0[t, None, None] + np.arange(h[t].max())[None, :, None]  # (B, bh, 1) pixel rows
        jj = j0[t, None, None] + np.arange(w[t].max())[None, None, :]  # (B, 1, bw) pixel columns
        in_box = torch.as_tensor((ii <= i1[t, None, None]) & (jj <= j1[t, None, None]))  # (B, bh, bw), not padding
        pixels = torch.as_tensor(np.minimum(ii, H - 1) * W + np.minimum(jj, W - 1))[in_box]  # Flat pixel indices

        M = A_inv[t]  # (B, 3, 3)
        jt, it = torch.as_tensor(jj, dtype=torch.float64), torch.as_tensor(ii, dtype=torch.float64)
        bary_coords = M[:, :, 0, None, None] * jt[:, None] + M[:, :, 1, None, None] * it[:, None] + M[:, :, 2, None, None]  # (B, 3, bh, bw)
        tri_inside = torch.all(bary_coords >= 0, dim=1)
        soft = torch.clamp(1 - torch.sqrt(torch.sum(bary_coords ** 2, dim=1)), 0, 1)

        inside = inside.index_add(0, pixels, tri_inside[in_box].double())
        soft_sum = soft_sum.index_add(0, pixels, torch.where(tri_inside, 0, soft)[in_box])
        start = end

    gray = torch.where(inside > 0, 0, torch.clamp(255 - 255 * soft_sum, min=0)).reshape(H, W)
    image.data = gray[:, :, None].repeat(1, 1, 3).float()  # White background, black silhouette

    return image

def neighbour_average(faces, n):
    """(n, n) matrix whose row v averages the vertices joined to vertex v by an edge."""
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    W = torch.zeros(n, n)
    W[edges[:, 0], edges[:, 1]] = 1
    W[edges[:, 1], edges[:, 0]] = 1
    return W / W.sum(dim=1, keepdim=True)

def rotate_quarter_turn(points):
    """Rotate (N, 3) points 90 degrees about the z axis: the pylon's square base and 8-sided cone look the same."""
    return torch.stack((-points[:, 1], points[:, 0], points[:, 2]), dim=1)

def mirror_side_to_side(points):
    """Reflect (N, 3) points in the plane y = 0: the car's left and right (x forward, y left, z up)."""
    return points * torch.tensor([1.0, -1.0, 1.0])

def symmetry_partners(vertices, symmetry):
    """For each vertex, the index of the vertex nearest to its image under symmetry (a function of points)."""
    with torch.no_grad():
        return torch.cdist(symmetry(vertices), vertices).argmin(dim=1)

# What to fit: the starting mesh, the target silhouettes and their cameras, the symmetry the result
# should have, and the settings that work for it (tuned on these meshes; the car is about 6 times the
# pylon's size, so it moves further per step and its terms in square metres need other weights). The
# pylon keeps its sharp edges (no Laplacian term); the high-poly car is rounded, so the car's
# Laplacian term pulls its surface smooth.
PROBLEMS = {
    'pylon': dict(init='pylon_init.obj', targets='pylon_target_silhouettes', symmetry=rotate_quarter_turn,
                  lr=3e-3, smoothness=1e5, laplacian=0, symmetric=3e5),
    'car': dict(init='car_init.obj', targets='car_target_silhouettes', symmetry=mirror_side_to_side,
                lr=1e-2, smoothness=0, laplacian=1e6, symmetric=3e6),
}

def optimize(problem='pylon', steps=200, lr=None, smoothness=None, laplacian=None, symmetric=None, callback=None):
    """Fit a starting mesh's vertices to a target's silhouettes, and return the fitted Object3D.

    problem is a key of PROBLEMS: 'pylon' (a plain cone to the complex pylon) or 'car' (the low-poly
    car, split to as many vertices as the high-poly car, to the high-poly car). lr, smoothness,
    laplacian and symmetric override the problem's settings.

    The silhouettes only say where the outline is, so more terms keep the surface between them in
    shape: smoothness (each vertex should move about as much as its neighbours do on average, which
    keeps the starting mesh's edges), the Laplacian (each vertex should sit at its neighbours'
    average, which smooths edges away) and symmetry (each vertex, turned or reflected by the
    problem's symmetry, should land on its partner, the vertex that was there at the start).

    If callback is given, callback(step, obj, loss) is called after every step (the viewer uses it to
    show the mesh as it changes).
    """
    settings = PROBLEMS[problem]
    lr = settings['lr'] if lr is None else lr
    smoothness = settings['smoothness'] if smoothness is None else smoothness
    laplacian = settings['laplacian'] if laplacian is None else laplacian
    symmetric = settings['symmetric'] if symmetric is None else symmetric

    # Load initial mesh, and the target silhouettes it should match, with their cameras
    init = load_obj(ASSETS / settings['init'])  # Object3D with optimizable vertices
    imgs = load_images(ASSETS / settings['targets'])  # List of Image objects
    cams = load_cameras(ASSETS / settings['targets'] / 'cameras.npz')  # List of Camera objects

    start = init.vertices.detach().clone()  # Where the vertices start, to measure how far they move
    neighbours = neighbour_average(init.faces, len(init.vertices))
    symmetry = settings['symmetry']
    partners = symmetry_partners(start, symmetry)

    optimizer = torch.optim.Adam(init.parameters(), lr=lr)

    for i in tqdm(range(steps), desc="Optimizing"):
        optimizer.zero_grad()  # Clear the gradients from the last step

        # For each target silhouette, render the current mesh and compute a loss
        total_loss = 0.0
        for img, cam in zip(imgs, cams):
            rendered = batched_differentiable_renderer_pytorch(init, cam, Image(img.width, img.height))  # A fresh image, not the target
            loss = torch.mean((rendered.data.float() - torch.as_tensor(img.data, dtype=torch.float32)) ** 2)
            total_loss += loss

        # Smoothness: how different each vertex's move is from its neighbours' average move
        moved = init.vertices - start
        total_loss += smoothness * torch.mean(torch.sum((moved - neighbours @ moved) ** 2, dim=1))

        # Laplacian: how far each vertex is from its neighbours' average
        total_loss += laplacian * torch.mean(torch.sum((init.vertices - neighbours @ init.vertices) ** 2, dim=1))

        # Symmetry: how far each vertex's image under the symmetry is from its partner
        total_loss += symmetric * torch.mean(torch.sum((symmetry(init.vertices) - init.vertices[partners]) ** 2, dim=1))

        # Backpropagate and update the mesh vertices
        total_loss.backward()
        optimizer.step()

        if callback is not None:
            callback(i, init, total_loss.item())

    return init
