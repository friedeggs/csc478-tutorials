"""Silhouette viewer for Tutorial 9 (2026-11-12, differentiable rendering).

Renders silhouettes of meshes in assets/ (by default the pylons,
pylon_simple.obj and pylon_complex.obj; the cars with --obj) with
traditional_renderer from differentiable_rendering.py: project the vertices
with K [R | t], then fill every triangle. The big view is the camera you steer,
lit (a key light from above and a headlight on the camera, flat shaded, back
faces culled, a z-buffer) or as the silhouette; the sidebar shows the
silhouettes from 16 fixed views (8 azimuths at two elevations) and a map of
where the cameras stand, seen from above. The camera looks at the centre of
the meshes' bounding box, from distances scaled to its size.

D shows differentiable_renderer's image instead, made at a lower resolution
(K scaled to match) and blown up. It runs in a separate process, which loads
differentiable_rendering.py afresh each time: the window stays responsive
however slow the renderer is, a render that no longer matches the camera is
stopped, a new one starts once the camera has stood still for a moment (or
the file is saved), and an exception is shown in the view. D again shows
fast_differentiable_renderer's image, live, every frame, reloaded when the
file is saved; D a third time goes back. Either renderer may return uint8 or
float images, in [0, 1] or [0, 255], gray or RGB, NumPy or PyTorch.

O runs optimize() from differentiable_rendering.py in a separate process
(loading the file afresh) and adds the mesh it is fitting, "optimized", to the
meshes: after every step its vertices are sent back, so every view shows the
mesh as it changes. Its thumbnails are the silhouettes the loss compares, from
the target cameras: black where the mesh and the target both cover a pixel,
red where only the mesh does, blue where only the target does. A small plot
in the big view shows the loss so far. O again stops it; the mesh stays.

Mouse
    left-drag           orbit the camera (azimuth, elevation)
    wheel               move closer / further
    click a view        jump to that view
    click a mesh        show that mesh (the buttons in the title bar)

Keys
    M / 1-9         next mesh / mesh by number (the camera stays where it is)
    L               big view lit / as the silhouette from traditional_renderer
    D               big view from differentiable_renderer / fast_differentiable_renderer / back
    O               run optimize() / stop it
    , / .           lower / higher resolution for the renderer on show
    [ / ]           narrower / wider field of view
    W               wireframe of the projected triangles on top
    Space           turntable: spin the camera round the car
    R               reset the camera
    Esc             quit

Run:  python 2026-11-12-differentiable_rendering.py [--obj assets/car.obj assets/car_highpoly.obj ...]
"""
import argparse
import importlib
import math
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
if __name__ != '__mp_main__':        # not in differentiable_renderer's process, which imports it itself (diff_worker)
    import pygame  # noqa: E402
    import torch  # noqa: E402

    from differentiable_rendering import (Camera, Image, Object3D, load_cameras, load_images,  # noqa: E402
                                          traditional_renderer)

PAD = 12
TITLE_H = 32
VX, VY, VW, VH = PAD, TITLE_H, 640, 480             # main view
IX, IY, IW = PAD, VY + VH + 8, VW                   # camera info under it
SBX = VX + VW + 2 * PAD                              # sidebar
TW, TH, TCOLS, TGAP, TLABEL = 112, 84, 4, 8, 18      # thumbnails
SBW = TCOLS * TW + (TCOLS - 1) * TGAP
WIN_W = SBX + SBW + PAD
WIN_H = 720
MAP_Y = TITLE_H + 20 + 4 * (TH + TLABEL + TGAP) + 20
MAP_H = WIN_H - PAD - MAP_Y

BG = (248, 250, 252)          # #f8fafc
FG = (15, 23, 42)             # #0f172a
MUTED = (100, 116, 139)
FAINT = (148, 163, 184)
LINE = (203, 213, 225)
ACCENT = (37, 99, 235)        # #2563eb
BAD = (220, 38, 38)
PANE_BG = (255, 255, 255)
VIEW_BG = (241, 245, 249)
PAINT = np.array([190, 190, 194])                    # the car's colour when lit, a light gray
KEY_LIGHT = np.array([0.4, 0.5, 0.75]) / np.linalg.norm([0.4, 0.5, 0.75])   # towards the light, world
AMBIENT, KEY, HEADLIGHT = 0.22, 0.65, 0.3
MONO = 'menlo,consolas,dejavusansmono,couriernew,monospace'

# Distances below are for a scene 4 m across (the cars); the viewer scales them to the meshes it loads
# and points the camera at the centre of their bounding box.
SCENE_SIZE = 4.0
HOME = dict(az=35.0, el=15.0, dist=8.0, fov=40.0)
VIEWS = [(az, el) for el in (10.0, 40.0) for az in range(0, 360, 45)]
THUMB_DIST, THUMB_FOV = 8.0, 40.0
EL_RANGE, DIST_RANGE, FOV_RANGE = (-80.0, 85.0), (3.5, 30.0), (10.0, 100.0)
SPIN_DEG_PER_S = 40.0
DIFF_SIZES = [(20, 15), (40, 30), (80, 60), (160, 120), (320, 240), (640, 480)]
DIFF_SIZE = 2                                        # differentiable_renderer starts at 80 x 60,
FAST_SIZE = len(DIFF_SIZES) - 1                      # fast_differentiable_renderer at the view's 640 x 480
DIFF_SETTLE_S = 0.3                                  # render once the camera has stood still this long
RENDERER_FILE = HERE / 'differentiable_rendering.py'
TARGET_DIR = HERE / 'assets' / 'pylon_target_silhouettes'   # what optimize() fits: images and cameras.npz
EXTRA, MISSING = (239, 68, 68), (59, 130, 246)       # thumbnails while optimizing: mesh only, target only


class Mesh:
    """The viewer's own copy of a mesh, as NumPy arrays; renderers get it as an Object3D (PyTorch parameters)."""

    def __init__(self, vertices, faces):
        self.vertices, self.faces = vertices, faces


def load_obj(path):
    verts, faces = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'v':
            verts.append([float(p) for p in parts[1:4]])
        elif parts[0] == 'f':
            idx = [int(p.split('/')[0]) - 1 for p in parts[1:]]
            faces += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    return Mesh(np.array(verts), np.array(faces))


def camera_centre(az, el, dist, target):
    a, e = math.radians(az), math.radians(el)
    return target + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def look_at(centre, target, up=(0.0, 0.0, 1.0)):
    """World-to-camera [R | t] (4 x 4), OpenCV axes: x right, y down, z forward."""
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    E = np.eye(4)
    E[:3, :3] = np.stack([x, y, z])
    E[:3, 3] = -E[:3, :3] @ centre
    return E


def intrinsics(w, h, fov):
    f = (w / 2) / math.tan(math.radians(fov) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def make_camera(az, el, dist, fov, w, h, target):
    return Camera(intrinsics(w, h, fov), look_at(camera_centre(az, el, dist, target), target))


def silhouette(obj, cam, w, h):
    return traditional_renderer(Object3D(obj.vertices, obj.faces), cam, Image(w, h))


def shaded(obj, cam, w, h):
    """Flat-shaded render: cull back faces, light each face, keep the nearest at each pixel (a z-buffer).

    Painting far to near is not enough here: the cars' parts cut into each other."""
    V, F = obj.vertices, obj.faces
    T = V[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    norm = np.linalg.norm(n, axis=1)
    n = n / np.where(norm > 0, norm, 1)[:, None]
    R, t = cam.extrinsic_matrix[:3, :3], cam.extrinsic_matrix[:3, 3]
    centres = T.mean(axis=1)
    to_eye = -R.T @ t - centres
    to_eye /= np.linalg.norm(to_eye, axis=1)[:, None]
    facing = (norm > 0) & (np.einsum('ij,ij->i', n, to_eye) > 0)
    light = AMBIENT + KEY * np.clip(n @ KEY_LIGHT, 0, None) + HEADLIGHT * np.clip(np.einsum('ij,ij->i', n, to_eye), 0, None)
    colours = np.clip(PAINT * light[:, None], 0, 255).astype(np.uint8)
    z = V @ R[2] + t[2]                                   # camera depth of each vertex
    p = project(obj, cam) - 0.5                           # pixel (i, j) covers [j - 0.5, j + 0.5) etc.
    image = Image(w, h)
    image.data = np.full((h, w, 3), VIEW_BG, np.uint8)
    zbuf = np.full((h, w), np.inf)
    for k in np.flatnonzero(facing):
        (a, b, c), (za, zb, zc) = p[F[k]], z[F[k]]
        x0, y0 = np.maximum(np.ceil(np.minimum(np.minimum(a, b), c)), 0).astype(int)
        x1, y1 = np.minimum(np.floor(np.maximum(np.maximum(a, b), c)), (w - 1, h - 1)).astype(int)
        if x1 < x0 or y1 < y0:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if area == 0:
            continue
        wa = ((b[0] - X) * (c[1] - Y) - (b[1] - Y) * (c[0] - X)) / area
        wb = ((c[0] - X) * (a[1] - Y) - (c[1] - Y) * (a[0] - X)) / area
        wc = 1 - wa - wb
        d = wa * za + wb * zb + wc * zc
        win = zbuf[y0:y1 + 1, x0:x1 + 1]
        hit = (wa >= 0) & (wb >= 0) & (wc >= 0) & (d < win)
        win[hit] = d[hit]
        image.data[y0:y1 + 1, x0:x1 + 1][hit] = colours[k]
    return image


def as_rgb(data):
    """An image as (H, W, 3) uint8: from uint8 or float in [0, 1] or [0, 255], gray or RGB, NumPy or PyTorch."""
    if hasattr(data, 'detach'):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    a = data.astype(float)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if data.dtype.kind == 'f' and np.nanmax(a, initial=0) <= 1.0:
        a = a * 255
    return np.ascontiguousarray(np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8))


def to_surface(image):
    rgb = as_rgb(image.data)
    return pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), 'RGB')


def diff_worker(vertices, faces, K, E, w, h, out):
    """Run differentiable_renderer in its own process, so the window never waits for it and it can be stopped.

    It gets plain arrays and imports differentiable_rendering.py itself, so the file is read afresh and an
    error in it is caught here and shown in the window."""
    try:
        from differentiable_rendering import Camera, Image, Object3D, differentiable_renderer
        obj, cam, image = Object3D(vertices, faces), Camera(K, E), Image(w, h)
        start = time.perf_counter()
        result = differentiable_renderer(obj, cam, image)
        took = time.perf_counter() - start
        data = getattr(result if result is not None else image, 'data', result)
        rgb = as_rgb(data)
        if rgb.shape[:2] != (h, w):
            raise ValueError(f'differentiable_renderer returned a {rgb.shape[1]} x {rgb.shape[0]} image, '
                             f'expected {w} x {h}')
        out.put(('ok', rgb, took))
    except BaseException:
        out.put(('error', traceback.format_exc(limit=-3), 0.0))


def optimize_worker(out):
    """Run optimize() in its own process, sending the mesh and the loss back after every step."""
    try:
        from differentiable_rendering import optimize

        def send(step, obj, loss):
            out.put(('step', step, obj.vertices.detach().numpy().copy(), obj.faces.numpy().copy(), float(loss)))

        optimize(callback=send)
        out.put(('done', None))
    except BaseException:
        out.put(('error', traceback.format_exc(limit=-3)))


def project(obj, cam):
    P = np.hstack([obj.vertices, np.ones((len(obj.vertices), 1))])
    p = (cam.intrinsic_matrix @ (cam.extrinsic_matrix @ P.T)[:3]).T
    return p[:, :2] / p[:, 2:3]


class App:
    def __init__(self, args):
        pygame.init()
        pygame.display.set_caption('Tutorial 9: silhouettes')
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.font = pygame.font.SysFont(MONO, 14)
        self.bold = pygame.font.SysFont(MONO, 14, bold=True)
        self.small = pygame.font.SysFont(MONO, 12)
        self.wire = False
        self.lit = True
        self.spin = False
        self.drag = None
        self.diff_mode = None                # None, 'slow' (differentiable_renderer) or 'fast'
        self.diff_size, self.fast_size = DIFF_SIZE, FAST_SIZE
        self.diff_job = self.diff_result = self.diff_error = None
        self.diff_seen, self.diff_since = None, 0.0
        self.fast_module = sys.modules['differentiable_rendering']
        self.fast_mtime, self.fast_error, self.fast_ms = None, None, None
        self.opt_job, self.opt_k, self.opt_losses = None, None, []   # optimize() in its own process
        self.opt_status, self.opt_error = None, None   # None, 'starting', 'running', 'done', 'stopped', 'error'
        self.mesh_version = 0                          # goes up whenever the optimizer moves the mesh
        self.thumb_rects = [pygame.Rect(SBX + (k % TCOLS) * (TW + TGAP),
                                        TITLE_H + 20 + (k // TCOLS) * (TH + TLABEL + TGAP), TW, TH)
                            for k in range(len(VIEWS))]
        self.meshes = []                     # (name, Mesh, thumbnails), thumbnails made when first shown
        for path in args.obj:
            self.meshes.append([Path(path).name, load_obj(path), None])
        self.layout_meshes()
        # frame all the meshes: look at the centre of their bounding box, scale distances to its size
        allv = np.vstack([obj.vertices for _, obj, _ in self.meshes])
        lo, hi = allv.min(axis=0), allv.max(axis=0)
        self.target = (lo + hi) / 2
        self.scale = float(((hi - lo) * (1, 1, VW / VH)).max()) / SCENE_SIZE   # the view is wider than tall
        self.cam_state = self.home()
        self.targets = self.load_targets()
        self.select_mesh(0)

    def layout_meshes(self):
        self.mesh_rects, x = [], SBX
        for k, (name, _, _) in enumerate(self.meshes):
            w = self.small.size(f'{k + 1} {name}')[0] + 16
            self.mesh_rects.append(pygame.Rect(x, 5, w, 22))
            x += w + 6

    def load_targets(self):
        """optimize()'s target silhouettes and cameras, if they are the thumbnails' views (as with the pylons)."""
        try:
            images, cams = load_images(TARGET_DIR), load_cameras(TARGET_DIR / 'cameras.npz')
        except Exception:
            return None
        if len(images) != len(VIEWS) or len(cams) != len(VIEWS):
            return None
        for (az, el), cam in zip(VIEWS, cams):
            thumb = make_camera(az, el, THUMB_DIST * self.scale, THUMB_FOV, TW, TH, self.target)
            if not np.allclose(thumb.extrinsic_matrix, cam.extrinsic_matrix, atol=1e-6):
                return None                  # other meshes, framed differently: plain thumbnails
        return [(image.data[..., 0] < 128, cam, image.width, image.height) for image, cam in zip(images, cams)]

    def compare_thumbs(self, obj):
        """Thumbnails of obj against the targets, from the target cameras: what the loss compares."""
        thumbs = []
        for target, cam, w, h in self.targets:
            mine = silhouette(obj, cam, w, h).data[..., 0] < 128
            rgb = np.full((h, w, 3), 255, np.uint8)
            rgb[mine & target] = 0
            rgb[mine & ~target] = EXTRA
            rgb[~mine & target] = MISSING
            surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), 'RGB')
            thumbs.append(pygame.transform.smoothscale(surf, (TW, TH)))
        return thumbs

    def home(self):
        return dict(HOME, dist=HOME['dist'] * self.scale)

    def select_mesh(self, k):
        self.mesh_k = k % len(self.meshes)
        mesh = self.meshes[self.mesh_k]
        self.obj_name, self.obj = mesh[0], mesh[1]
        if mesh[2] is None and self.mesh_k == self.opt_k and self.targets:
            mesh[2] = self.compare_thumbs(self.obj)
        if mesh[2] is None:
            mesh[2] = [to_surface(silhouette(self.obj, make_camera(az, el, THUMB_DIST * self.scale, THUMB_FOV,
                                                                   TW, TH, self.target), TW, TH))
                       for az, el in VIEWS]
        self.thumbs = mesh[2]

    # ---- camera ---------------------------------------------------------------
    def camera(self):
        s = self.cam_state
        return make_camera(s['az'], s['el'], s['dist'], s['fov'], VW, VH, self.target)

    def orbit(self, dx, dy):
        s = self.cam_state
        s['az'] = (s['az'] - 0.4 * dx) % 360
        s['el'] = float(np.clip(s['el'] + 0.4 * dy, *EL_RANGE))

    def zoom(self, closer):
        s = self.cam_state
        s['dist'] = float(np.clip(s['dist'] * (0.9 if closer else 1 / 0.9), *(np.array(DIST_RANGE) * self.scale)))

    def widen(self, wider):
        s = self.cam_state
        s['fov'] = float(np.clip(s['fov'] + (5 if wider else -5), *FOV_RANGE))

    # ---- differentiable_renderer, in its own process ---------------------------
    def diff_key(self):
        """What the differentiable render depends on: mesh, camera, resolution and the renderer's file."""
        s = self.cam_state
        try:
            mtime = RENDERER_FILE.stat().st_mtime
        except OSError:
            mtime = None
        return (self.mesh_k, self.mesh_version if self.mesh_k == self.opt_k else 0, round(s['az'], 3), round(s['el'], 3), round(s['dist'], 3), s['fov'],
                self.diff_size, mtime)

    def start_diff(self, key):
        w, h = DIFF_SIZES[self.diff_size]
        s = self.cam_state
        cam = make_camera(s['az'], s['el'], s['dist'], s['fov'], w, h, self.target)
        ctx = mp.get_context('spawn')
        out = ctx.Queue()
        proc = ctx.Process(target=diff_worker, daemon=True,
                           args=(self.obj.vertices, self.obj.faces, cam.intrinsic_matrix, cam.extrinsic_matrix, w, h, out))
        proc.start()
        self.diff_job = dict(proc=proc, out=out, key=key, size=(w, h), started=time.monotonic())

    def stop_diff(self):
        if self.diff_job:
            self.diff_job['proc'].terminate()
            self.diff_job['proc'].join(timeout=1)
            self.diff_job = None

    def finish_diff(self, status, payload, took):
        job, self.diff_job = self.diff_job, None
        job['proc'].join(timeout=1)
        if status == 'ok':
            surf = pygame.image.frombuffer(payload.tobytes(), job['size'], 'RGB')
            cover = 1 - payload.mean() / 255
            self.diff_result = dict(surf=surf, key=job['key'], size=job['size'], took=took, cover=cover)
            self.diff_error = None
        else:
            self.diff_error = dict(text=payload, key=job['key'])

    def update_diff(self):
        """Collect a finished render, stop one that no longer matches, start one once the camera stands still."""
        key, now = self.diff_key(), time.monotonic()
        if key != self.diff_seen:
            self.diff_seen, self.diff_since = key, now
        job = self.diff_job
        if job:
            try:
                self.finish_diff(*job['out'].get_nowait())
            except queue.Empty:
                if not job['proc'].is_alive():
                    try:
                        self.finish_diff(*job['out'].get(timeout=0.5))
                    except queue.Empty:
                        self.finish_diff('error', f'the render process stopped (exit code {job["proc"].exitcode}); '
                                                  'see the terminal', 0.0)
                elif job['key'] != key:
                    self.stop_diff()
        done = {(self.diff_result or {}).get('key'), (self.diff_error or {}).get('key')}
        if self.diff_job is None and key not in done and now - self.diff_since >= DIFF_SETTLE_S:
            self.start_diff(key)
        return key

    # ---- optimize(), in its own process -----------------------------------------
    def start_opt(self):
        ctx = mp.get_context('spawn')
        out = ctx.Queue()
        proc = ctx.Process(target=optimize_worker, args=(out,), daemon=True)
        proc.start()
        self.opt_job, self.opt_losses, self.opt_status, self.opt_error = dict(proc=proc, out=out), [], 'starting', None

    def stop_opt(self, status='stopped'):
        if self.opt_job:
            self.opt_job['proc'].terminate()
            self.opt_job['proc'].join(timeout=1)
            self.opt_job, self.opt_status = None, status

    def update_opt(self):
        """Take the optimizer's latest mesh (skipping steps the window had no time to show) and its losses."""
        job, latest = self.opt_job, None
        if job is None:
            return
        while True:
            try:
                msg = job['out'].get_nowait()
            except queue.Empty:
                break
            if msg[0] == 'step':
                self.opt_losses.append(msg[4])
                latest, self.opt_status = msg, 'running'
            elif msg[0] == 'done':
                self.opt_status = 'done'
            else:
                self.opt_status, self.opt_error = 'error', msg[1]
        if latest is not None:
            mesh = Mesh(latest[2], latest[3])
            if self.opt_k is None:
                self.meshes.append(['optimized', mesh, None])
                self.opt_k = len(self.meshes) - 1
                self.layout_meshes()
            self.meshes[self.opt_k][1:] = [mesh, None]
            self.mesh_version += 1
            if len(self.opt_losses) == 1 or self.mesh_k == self.opt_k:
                self.select_mesh(self.opt_k)      # show the mesh being fitted from its first step on
        if self.opt_status in ('done', 'error'):
            job['proc'].join(timeout=1)
            self.opt_job = None
        elif not job['proc'].is_alive() and job['out'].empty():
            self.opt_job, self.opt_status = None, 'error'
            self.opt_error = f'the optimizer process stopped (exit code {job["proc"].exitcode}); see the terminal'

    def draw_opt(self):
        """The optimizer's progress, bottom right in the big view: status and the loss so far."""
        if self.opt_status is None:
            return
        box = pygame.Rect(VX + VW - 244, VY + VH - 26 - 112, 240, 112)
        pygame.draw.rect(self.screen, PANE_BG, box)
        pygame.draw.rect(self.screen, LINE, box, 1)
        losses = self.opt_losses
        status = {'starting': 'starting...', 'running': f'step {len(losses)}', 'done': f'done, {len(losses)} steps',
                  'stopped': f'stopped at step {len(losses)}', 'error': 'failed'}[self.opt_status]
        self.text(f'optimize(): {status}', (box.x + 6, box.y + 4), self.small, BAD if self.opt_status == 'error' else FG)
        self.text(f'loss {losses[-1]:.0f}' if losses else '', (box.x + 6, box.y + 20), self.small, MUTED)
        self.text('O stop' if self.opt_job else 'O run again', (box.right - 80, box.y + 20), self.small, MUTED)
        plot = pygame.Rect(box.x + 6, box.y + 40, box.w - 12, box.h - 46)
        pygame.draw.line(self.screen, LINE, plot.bottomleft, plot.bottomright)
        if len(losses) > 1:
            top = max(losses)
            pts = [(plot.x + plot.w * i / (len(losses) - 1), plot.bottom - plot.h * l / top) for i, l in enumerate(losses)]
            pygame.draw.lines(self.screen, ACCENT, False, pts, 2)
        if self.opt_status == 'error':
            self.draw_error(self.opt_error)

    # ---- events ---------------------------------------------------------------
    def handle(self, ev):
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
            return False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_w:
                self.wire = not self.wire
            elif ev.key == pygame.K_l:
                self.lit = not self.lit
            elif ev.key == pygame.K_d:
                self.diff_mode = {None: 'slow', 'slow': 'fast', 'fast': None}[self.diff_mode]
                if self.diff_mode != 'slow':
                    self.stop_diff()
            elif ev.key in (pygame.K_COMMA, pygame.K_PERIOD):
                step = 1 if ev.key == pygame.K_PERIOD else -1
                if self.diff_mode == 'fast':
                    self.fast_size = int(np.clip(self.fast_size + step, 0, len(DIFF_SIZES) - 1))
                else:
                    self.diff_size = int(np.clip(self.diff_size + step, 0, len(DIFF_SIZES) - 1))
            elif ev.key == pygame.K_o:
                if self.opt_job:
                    self.stop_opt()
                else:
                    self.start_opt()
            elif ev.key == pygame.K_SPACE:
                self.spin = not self.spin
            elif ev.key == pygame.K_r:
                self.cam_state, self.spin = self.home(), False
            elif ev.key == pygame.K_LEFTBRACKET:
                self.widen(False)
            elif ev.key == pygame.K_RIGHTBRACKET:
                self.widen(True)
            elif ev.key == pygame.K_m:
                self.select_mesh(self.mesh_k + 1)
            elif pygame.K_1 <= ev.key <= pygame.K_9 and ev.key - pygame.K_1 < len(self.meshes):
                self.select_mesh(ev.key - pygame.K_1)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for k, r in enumerate(self.mesh_rects):
                if r.collidepoint(ev.pos):
                    self.select_mesh(k)
                    return True
            for k, r in enumerate(self.thumb_rects):
                if r.collidepoint(ev.pos):
                    self.cam_state['az'], self.cam_state['el'] = map(float, VIEWS[k])
                    self.spin = False
                    return True
            if pygame.Rect(VX, VY, VW, VH).collidepoint(ev.pos):
                self.drag = ev.pos
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.drag = None
        elif ev.type == pygame.MOUSEMOTION and self.drag:
            self.orbit(ev.pos[0] - self.drag[0], ev.pos[1] - self.drag[1])
            self.drag = ev.pos
        elif ev.type == pygame.MOUSEWHEEL and pygame.Rect(VX, VY, VW, VH).collidepoint(pygame.mouse.get_pos()):
            self.zoom(ev.y > 0)
        return True

    # ---- drawing --------------------------------------------------------------
    def text(self, s, pos, font=None, col=FG):
        self.screen.blit((font or self.font).render(s, True, col), pos)

    def draw_main(self, cam):
        if self.diff_mode == 'slow':
            cover = self.draw_diff()
        elif self.diff_mode == 'fast':
            cover = self.draw_fast()
        else:
            if self.lit:
                image = shaded(self.obj, cam, VW, VH)
                cover = (image.data != VIEW_BG).any(axis=2).mean()
            else:
                image = silhouette(self.obj, cam, VW, VH)
                cover = (image.data[..., 0] < 128).mean()
            self.screen.blit(to_surface(image), (VX, VY))
        if self.wire:
            p = project(self.obj, cam) + (VX, VY)
            for tri in self.obj.faces:
                pygame.draw.lines(self.screen, ACCENT, True, p[tri].tolist(), 1)
        pygame.draw.rect(self.screen, LINE, (VX, VY, VW, VH), 1)
        if not self.diff_mode:
            self.text('lit (L: silhouette, D: differentiable)' if self.lit else
                      'silhouette from traditional_renderer (L: lit, D: differentiable)', (VX + 8, VY + 6), self.small, MUTED)
        info = f'{self.obj_name}: {len(self.obj.vertices)} vertices, {len(self.obj.faces)} triangles'
        if cover is not None:
            info += f'   silhouette covers {100 * cover:.1f}% of the image'
        self.text(info, (VX + 8, VY + VH - 22), self.small, MUTED)
        self.draw_opt()

    def draw_diff(self):
        """The big view from differentiable_renderer: the last render, blown up, with what is going on."""
        key = self.update_diff()
        w, h = DIFF_SIZES[self.diff_size]
        self.screen.fill(VIEW_BG, (VX, VY, VW, VH))
        res, err, job = self.diff_result, self.diff_error, self.diff_job
        shown = res if res and res['key'][0] == self.mesh_k else None
        if shown:
            self.screen.blit(pygame.transform.scale(shown['surf'], (VW, VH)), (VX, VY))
        if job:
            status = f'rendering at {w} x {h}, {time.monotonic() - job["started"]:.0f} s so far'
        elif err and err['key'] == key:
            status = 'failed, fix differentiable_rendering.py and save to try again'
        elif shown and shown['key'] == key:
            status = f'{shown["size"][0]} x {shown["size"][1]} in {shown["took"]:.1f} s'
        else:
            status = 'waits for the camera to stand still'
        if shown and shown['key'] != key:
            status += '; showing the last render'
        self.draw_status(f'differentiable_renderer: {status}', ', . resolution   D fast')
        if err and err['key'] == key:
            self.draw_error(err['text'])
        return shown['cover'] if shown else None

    def fast_renderer(self):
        """fast_differentiable_renderer, reloaded from differentiable_rendering.py whenever the file is saved."""
        try:
            mtime = RENDERER_FILE.stat().st_mtime
        except OSError:
            mtime = None
        if mtime != self.fast_mtime:
            self.fast_mtime = mtime
            try:
                self.fast_module = importlib.reload(self.fast_module)
                self.fast_error = None
            except Exception:
                self.fast_error = traceback.format_exc(limit=-3)
        return getattr(self.fast_module, 'fast_differentiable_renderer', None)

    def draw_fast(self):
        """The big view from fast_differentiable_renderer, made afresh every frame."""
        w, h = DIFF_SIZES[self.fast_size]
        self.screen.fill(VIEW_BG, (VX, VY, VW, VH))
        fn, cover = self.fast_renderer(), None
        if self.fast_error is None:
            if fn is None:
                self.fast_error = 'differentiable_rendering.py has no fast_differentiable_renderer yet'
            else:
                s = self.cam_state
                cam = make_camera(s['az'], s['el'], s['dist'], s['fov'], w, h, self.target)
                try:
                    obj, image = self.fast_module.Object3D(self.obj.vertices, self.obj.faces), self.fast_module.Image(w, h)
                    start = time.perf_counter()
                    with torch.no_grad():                    # just looking: no gradients needed
                        result = fn(obj, cam, image)
                    ms = 1000 * (time.perf_counter() - start)
                    rgb = as_rgb(getattr(result if result is not None else image, 'data', result))
                    if rgb.shape[:2] != (h, w):
                        raise ValueError(f'fast_differentiable_renderer returned a {rgb.shape[1]} x {rgb.shape[0]} '
                                         f'image, expected {w} x {h}')
                    self.fast_ms = ms if self.fast_ms is None else 0.8 * self.fast_ms + 0.2 * ms
                    surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), 'RGB')
                    self.screen.blit(surf if (w, h) == (VW, VH) else pygame.transform.scale(surf, (VW, VH)), (VX, VY))
                    cover = 1 - rgb.mean() / 255
                except Exception:
                    self.fast_error = traceback.format_exc(limit=-3)
        if self.fast_error:
            self.draw_status('fast_differentiable_renderer: failed, fix differentiable_rendering.py and save',
                             ', . resolution   D back')
            self.draw_error(self.fast_error)
        else:
            self.draw_status(f'fast_differentiable_renderer: {w} x {h} in {self.fast_ms:.0f} ms, live',
                             ', . resolution   D back')
        return cover

    def draw_status(self, *lines):
        box = pygame.Rect(VX + 4, VY + 4, max(self.small.size(s)[0] for s in lines) + 10, 18 * len(lines) + 6)
        pygame.draw.rect(self.screen, PANE_BG, box)
        for i, s in enumerate(lines):
            self.text(s, (box.x + 5, box.y + 4 + 18 * i), self.small, FG if i == 0 else MUTED)

    def draw_error(self, text):
        rows = [r for line in text.strip().splitlines() for r in self.wrap(line, VW - 40)][-14:]
        panel = pygame.Rect(VX + 10, VY + 60, VW - 20, 18 * len(rows) + 12)
        pygame.draw.rect(self.screen, PANE_BG, panel)
        pygame.draw.rect(self.screen, BAD, panel, 1)
        for i, r in enumerate(rows):
            self.text(r, (panel.x + 8, panel.y + 6 + 18 * i), self.small, BAD)

    def wrap(self, line, width):
        """Split a line of text into pieces no wider than width pixels."""
        out = []
        while self.small.size(line)[0] > width:
            n = max(1, int(len(line) * width / self.small.size(line)[0]))
            out.append(line[:n])
            line = '    ' + line[n:]
        return out + [line]

    def draw_info(self, cam):
        s = self.cam_state
        self.text(f"azimuth {s['az']:5.1f} deg   elevation {s['el']:5.1f} deg   "
                  f"distance {s['dist']:5.2f} m   fov {s['fov']:3.0f} deg", (IX, IY), self.bold)
        K, E = cam.intrinsic_matrix, cam.extrinsic_matrix
        self.text('K (intrinsics)', (IX, IY + 26), self.small, MUTED)
        self.text('[R | t] (extrinsics, world -> camera)', (IX + 230, IY + 26), self.small, MUTED)
        for i in range(3):
            self.text('  '.join(f'{v:7.1f}' for v in K[i]), (IX, IY + 44 + 18 * i))
            self.text('  '.join(f'{v:6.3f}' for v in E[i]), (IX + 230, IY + 44 + 18 * i))
        self.text('drag orbit  wheel distance  [ ] fov  Space spin  R reset  M mesh', (IX, WIN_H - PAD - 34), self.small, FAINT)
        self.text('W wire  L lit  D differentiable  O optimize', (IX, WIN_H - PAD - 16), self.small, FAINT)

    def draw_meshes(self):
        for k, ((name, _, _), r) in enumerate(zip(self.meshes, self.mesh_rects)):
            on = k == self.mesh_k
            pygame.draw.rect(self.screen, ACCENT if on else PANE_BG, r, border_radius=4)
            pygame.draw.rect(self.screen, ACCENT if on else LINE, r, 1, border_radius=4)
            self.text(f'{k + 1} {name}', (r.x + 8, r.y + 4), self.small, PANE_BG if on else FG)

    def nearest_view(self):
        s = self.cam_state
        d = [abs((s['az'] - az + 180) % 360 - 180) + abs(s['el'] - el) for az, el in VIEWS]
        k = int(np.argmin(d))
        return k if d[k] < 1e-6 else None

    def draw_thumbs(self):
        self.text('Views (click to open)', (SBX, TITLE_H), self.bold)
        if self.mesh_k == self.opt_k and self.targets:
            x = SBX + SBW
            for label, col in (('target only', MISSING), ('mesh only', EXTRA), ('both', FG)):
                x -= self.small.size(label)[0] + 22
                pygame.draw.rect(self.screen, col, (x, TITLE_H + 3, 10, 10))
                self.text(label, (x + 14, TITLE_H + 1), self.small, MUTED)
        current = self.nearest_view()
        for k, ((az, el), surf, r) in enumerate(zip(VIEWS, self.thumbs, self.thumb_rects)):
            self.screen.blit(surf, r.topleft)
            pygame.draw.rect(self.screen, ACCENT if k == current else LINE, r, 2 if k == current else 1)
            self.text(f'az {az:3.0f}  el {el:2.0f}', (r.x, r.bottom + 2), self.small,
                      ACCENT if k == current else MUTED)

    def draw_map(self):
        """Top view: x (forward) up the screen, y (left) to the left."""
        self.text('Cameras, seen from above', (SBX, MAP_Y - 20), self.bold)
        rect = pygame.Rect(SBX, MAP_Y, SBW, MAP_H)
        pygame.draw.rect(self.screen, PANE_BG, rect)
        pygame.draw.rect(self.screen, LINE, rect, 1)
        s = self.cam_state
        reach = max(THUMB_DIST * self.scale, s['dist']) + 0.8 * self.scale
        scale = (MAP_H / 2 - 10) / reach
        cx, cy = rect.center

        def to_map(p):
            return (cx - (p[1] - self.target[1]) * scale, cy - (p[0] - self.target[0]) * scale)

        for tri in self.obj.faces:
            pygame.draw.polygon(self.screen, FAINT, [to_map(v) for v in self.obj.vertices[tri]])
        pygame.draw.circle(self.screen, LINE, (cx, cy), int(THUMB_DIST * self.scale * math.cos(math.radians(10)) * scale), 1)
        for az, el in VIEWS:
            pygame.draw.circle(self.screen, MUTED, to_map(camera_centre(az, el, THUMB_DIST * self.scale, self.target)), 3)
        c = camera_centre(s['az'], s['el'], s['dist'], self.target)
        fwd = (self.target - c)[:2] / max(np.linalg.norm((self.target - c)[:2]), 1e-9)
        half = math.radians(s['fov']) / 2
        for sgn in (-1, 1):
            rot = np.array([[math.cos(sgn * half), -math.sin(sgn * half)], [math.sin(sgn * half), math.cos(sgn * half)]])
            tip = c[:2] + rot @ fwd * 1.6 * self.scale
            pygame.draw.line(self.screen, ACCENT, to_map(c), to_map(tip), 2)
        pygame.draw.circle(self.screen, ACCENT, to_map(c), 6)
        self.text('front', (cx - 18, rect.y + 4), self.small, FAINT)

    def draw(self):
        self.update_opt()
        self.screen.fill(BG)
        self.text('Tutorial 9: silhouettes from traditional_renderer', (PAD, 8), self.bold)
        self.draw_meshes()
        cam = self.camera()
        self.draw_main(cam)
        self.draw_info(cam)
        self.draw_thumbs()
        self.draw_map()
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            dt = clock.tick(60) / 1000
            for ev in pygame.event.get():
                running = running and self.handle(ev)
            if self.spin:
                self.cam_state['az'] = (self.cam_state['az'] + SPIN_DEG_PER_S * dt) % 360
            self.draw()
        self.stop_diff()
        self.stop_opt()
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--obj', nargs='+', default=[str(HERE / 'assets' / n) for n in ('pylon_simple.obj', 'pylon_complex.obj')],
                    help='meshes to switch between')
    App(ap.parse_args()).run()


if __name__ == '__main__':
    main()
