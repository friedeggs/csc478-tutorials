"""Silhouette viewer for Tutorial 9 (2026-11-12, differentiable rendering).

Renders silhouettes of assets/car.obj with traditional_renderer from
differentiable_rendering.py: project the vertices with K [R | t], then fill
every triangle. The big view is the camera you steer; the sidebar shows the
same car from 16 fixed views (8 azimuths at two elevations) and a map of where
the cameras stand, seen from above.

Mouse
    left-drag           orbit the camera (azimuth, elevation)
    wheel               move closer / further
    click a view        jump to that view

Keys
    [ / ]           narrower / wider field of view
    W               wireframe of the projected triangles on top of the silhouette
    Space           turntable: spin the camera round the car
    R               reset the camera
    Esc             quit

Run:  python 2026-11-12-differentiable_rendering.py [--obj assets/car.obj]
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame  # noqa: E402

from differentiable_rendering import Camera, Image, Object3D, traditional_renderer  # noqa: E402

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
PANE_BG = (255, 255, 255)
MONO = 'menlo,consolas,dejavusansmono,couriernew,monospace'

TARGET = np.array([0.0, 0.0, 0.6])                  # the camera looks here (car centre)
HOME = dict(az=35.0, el=15.0, dist=8.0, fov=40.0)
VIEWS = [(az, el) for el in (10.0, 40.0) for az in range(0, 360, 45)]
THUMB_DIST, THUMB_FOV = 8.0, 40.0
EL_RANGE, DIST_RANGE, FOV_RANGE = (-80.0, 85.0), (3.5, 30.0), (10.0, 100.0)
SPIN_DEG_PER_S = 40.0


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
    return Object3D(np.array(verts), np.array(faces))


def camera_centre(az, el, dist):
    a, e = math.radians(az), math.radians(el)
    return TARGET + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def look_at(centre, target=TARGET, up=(0.0, 0.0, 1.0)):
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


def make_camera(az, el, dist, fov, w, h):
    return Camera(intrinsics(w, h, fov), look_at(camera_centre(az, el, dist)))


def silhouette(obj, cam, w, h):
    return traditional_renderer(obj, cam, Image(w, h))


def to_surface(image):
    rgb = np.ascontiguousarray(np.clip(image.data, 0, 255).astype(np.uint8))
    return pygame.image.frombuffer(rgb.tobytes(), (image.width, image.height), 'RGB')


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
        self.obj = load_obj(args.obj)
        self.obj_name = Path(args.obj).name
        self.cam_state = dict(HOME)
        self.wire = False
        self.spin = False
        self.drag = None
        self.thumbs = [to_surface(silhouette(self.obj, make_camera(az, el, THUMB_DIST, THUMB_FOV, TW, TH), TW, TH))
                       for az, el in VIEWS]
        self.thumb_rects = [pygame.Rect(SBX + (k % TCOLS) * (TW + TGAP),
                                        TITLE_H + 20 + (k // TCOLS) * (TH + TLABEL + TGAP), TW, TH)
                            for k in range(len(VIEWS))]

    # ---- camera ---------------------------------------------------------------
    def camera(self):
        s = self.cam_state
        return make_camera(s['az'], s['el'], s['dist'], s['fov'], VW, VH)

    def orbit(self, dx, dy):
        s = self.cam_state
        s['az'] = (s['az'] - 0.4 * dx) % 360
        s['el'] = float(np.clip(s['el'] + 0.4 * dy, *EL_RANGE))

    def zoom(self, closer):
        s = self.cam_state
        s['dist'] = float(np.clip(s['dist'] * (0.9 if closer else 1 / 0.9), *DIST_RANGE))

    def widen(self, wider):
        s = self.cam_state
        s['fov'] = float(np.clip(s['fov'] + (5 if wider else -5), *FOV_RANGE))

    # ---- events ---------------------------------------------------------------
    def handle(self, ev):
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
            return False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_w:
                self.wire = not self.wire
            elif ev.key == pygame.K_SPACE:
                self.spin = not self.spin
            elif ev.key == pygame.K_r:
                self.cam_state, self.spin = dict(HOME), False
            elif ev.key == pygame.K_LEFTBRACKET:
                self.widen(False)
            elif ev.key == pygame.K_RIGHTBRACKET:
                self.widen(True)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
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
        image = silhouette(self.obj, cam, VW, VH)
        self.screen.blit(to_surface(image), (VX, VY))
        if self.wire:
            p = project(self.obj, cam) + (VX, VY)
            for tri in self.obj.faces:
                pygame.draw.lines(self.screen, ACCENT, True, p[tri].tolist(), 1)
        pygame.draw.rect(self.screen, LINE, (VX, VY, VW, VH), 1)
        cover = (image.data[..., 0] < 128).mean()
        self.text(f'{self.obj_name}: {len(self.obj.vertices)} vertices, {len(self.obj.faces)} triangles'
                  f'   silhouette covers {100 * cover:.1f}% of the image', (VX + 8, VY + VH - 22), self.small, MUTED)

    def draw_info(self, cam):
        s = self.cam_state
        self.text(f"azimuth {s['az']:5.1f} deg   elevation {s['el']:5.1f} deg   "
                  f"distance {s['dist']:4.1f} m   fov {s['fov']:3.0f} deg", (IX, IY), self.bold)
        K, E = cam.intrinsic_matrix, cam.extrinsic_matrix
        self.text('K (intrinsics)', (IX, IY + 26), self.small, MUTED)
        self.text('[R | t] (extrinsics, world -> camera)', (IX + 230, IY + 26), self.small, MUTED)
        for i in range(3):
            self.text('  '.join(f'{v:7.1f}' for v in K[i]), (IX, IY + 44 + 18 * i))
            self.text('  '.join(f'{v:6.3f}' for v in E[i]), (IX + 230, IY + 44 + 18 * i))
        self.text('drag: orbit   wheel: distance   [ ]: fov   W: wireframe   Space: spin   R: reset',
                  (IX, WIN_H - PAD - 16), self.small, FAINT)

    def nearest_view(self):
        s = self.cam_state
        d = [abs((s['az'] - az + 180) % 360 - 180) + abs(s['el'] - el) for az, el in VIEWS]
        k = int(np.argmin(d))
        return k if d[k] < 1e-6 else None

    def draw_thumbs(self):
        self.text('Views (click to open)', (SBX, TITLE_H), self.bold)
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
        reach = max(THUMB_DIST, s['dist']) + 0.8
        scale = (MAP_H / 2 - 10) / reach
        cx, cy = rect.center

        def to_map(p):
            return (cx - (p[1] - TARGET[1]) * scale, cy - (p[0] - TARGET[0]) * scale)

        for tri in self.obj.faces:
            pygame.draw.polygon(self.screen, FAINT, [to_map(v) for v in self.obj.vertices[tri]])
        pygame.draw.circle(self.screen, LINE, (cx, cy), int(THUMB_DIST * math.cos(math.radians(10)) * scale), 1)
        for az, el in VIEWS:
            pygame.draw.circle(self.screen, MUTED, to_map(camera_centre(az, el, THUMB_DIST)), 3)
        c = camera_centre(s['az'], s['el'], s['dist'])
        fwd = (TARGET - c)[:2] / max(np.linalg.norm((TARGET - c)[:2]), 1e-9)
        half = math.radians(s['fov']) / 2
        for sgn in (-1, 1):
            rot = np.array([[math.cos(sgn * half), -math.sin(sgn * half)], [math.sin(sgn * half), math.cos(sgn * half)]])
            tip = c[:2] + rot @ fwd * 1.6
            pygame.draw.line(self.screen, ACCENT, to_map(c), to_map(tip), 2)
        pygame.draw.circle(self.screen, ACCENT, to_map(c), 6)
        self.text('front', (cx - 18, rect.y + 4), self.small, FAINT)

    def draw(self):
        self.screen.fill(BG)
        self.text('Tutorial 9: silhouettes from traditional_renderer', (PAD, 8), self.bold)
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
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--obj', default=str(HERE / 'assets' / 'car.obj'), help='mesh to render')
    App(ap.parse_args()).run()


if __name__ == '__main__':
    main()
