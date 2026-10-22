"""SDF demo for Tutorial 7 (2026-10-22, Volumetric Fusion: SDFs and marching cubes).

Four tabs, one after the other:

Tab 1, Torus (slide 173): write torus_sdf (T1) and fit (T2) in sdf_student.py.
    Save, and the demo reloads it. F fits your torus to the depth points of six
    cameras: watch its pose and its three scales change, step by step.
Tab 2, CSG: build union, intersect and subtract from min / max magnets (C1-C3),
    then the mug from a body, a cavity and a handle (C4).
Tab 3, Mug: your fit() again, now on your mug: 7 sizes and a pose, fitted to
    the depth maps of a real mug (assets/mug.obj).
Tab 4, Mesh (slides 189-191): sample your mug's SDF on a grid and run marching
    cubes. When it meets a sign pattern it has no triangles for, it stops: draw
    them in the cube on the right (press on a point, drag through a second,
    release on a third). Once they are right the cube stays on that case
    until you continue (E); click a drawn case in the list to open it again.

Viewer
    left-drag           orbit the camera            wheel   zoom
    tab 3: Shift + left-drag turns the mug, right-drag moves it

Keys
    1-4             tabs
    F               fit (tabs 1, 3)          Space   pause / resume a fit or the march
    R               reset to the start (tabs 1, 3) / clear the mesh (tab 4)
    H               tab 3: turn the mug half way round
    X               cut-away view (tabs 2, 3)
    G / E           tab 4: sample the grid / extract the mesh (E also continues after a case is drawn)
    I               tab 4: vertices at edge midpoints / interpolated (slide 189)
    C               reset the camera
    A               fill the released answer for the step under the mouse
    Backspace       remove the magnet under the mouse / the last triangle
    Delete          empty the row under the mouse / the cube's triangles
    Esc             drop what you are dragging / stop a fit / quit

Run:  python 2026-10-22-sdf_demo.py          (--answers fills every released answer)
"""
import argparse
import importlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame  # noqa: E402
import torch  # noqa: E402

import sdf_lib as L  # noqa: E402

PAD = 12
TITLE_H = 32
VX, VY, VW, VH = PAD, TITLE_H, 720, 500          # viewer
GX, GY, GW, GH = PAD, VY + VH + 6, VW, 84       # plot
MX, MY, MW = PAD, GY + GH + 6, VW               # mode bar
SBX = VX + VW + PAD                               # sidebar
SBW = 504
WIN_W = SBX + SBW + PAD
WIN_H = 720
SBY, SBH = 8, WIN_H - 16
LO_RES, HI_RES = (240, 167), (480, 333)          # sphere tracing while moving / when still

BG = (248, 250, 252)          # #f8fafc
FG = (15, 23, 42)             # #0f172a
MUTED = (100, 116, 139)
FAINT = (148, 163, 184)
TRACK = (226, 232, 240)
LINE = (203, 213, 225)
ACCENT = (37, 99, 235)        # #2563eb
ACCENT_BG = (219, 234, 254)
PANE_BG = (255, 255, 255)
VIEW_BG = (241, 245, 249)
SURF = (96, 165, 250)         # fitted surface, #60a5fa
GOOD = (22, 163, 74)
GOOD_BG = (220, 252, 231)
BAD = (220, 38, 38)
BAD_BG = (254, 226, 226)
AMBER = (217, 119, 6)
AMBER_BG = (254, 243, 199)
GLOW = (250, 204, 21)
WHITE = (255, 255, 255)
INSIDE = (220, 38, 38)        # red / green = inside / outside (slide 189)
OUTSIDE = (22, 163, 74)
HOLE = (249, 115, 22)         # cells waiting for a missing case
SOLID = (139, 92, 246)        # success violet, as in Tutorial 6
CUT = (148, 163, 184)
CONFETTI = [SURF, SOLID, AMBER, GOOD, GLOW, (217, 70, 239)]
VIEW_COLS = [(59, 130, 246), (217, 70, 239), (20, 184, 166), (234, 88, 12), (132, 204, 22), (99, 102, 241)]
PART_COL = {'a': (59, 130, 246), 'b': (234, 88, 12), 'body': (96, 165, 250), 'cavity': (245, 158, 11),
            'handle': (217, 70, 239)}
MONO = 'menlo,consolas,dejavusansmono,couriernew,monospace'

MAG_H = 26
POP_SECS, POP_STAGGER = 0.32, 0.05
SNAP_SECS = 1.2
FIT_STEPS, FIT_LR = 300, 0.05
MARCH_DELAY = 0.05            # s per slab
H_STEPS = [0.6, 0.5, 0.4, 0.3]
TABS = [('torus', '1 Torus'), ('csg', '2 CSG'), ('mug', '3 Mug'), ('mesh', '4 Mesh')]
HINTS = {
    'torus': 'F fit  Space pause  R reset · edit sdf_student.py and save · drag orbit, wheel zoom, C camera',
    'csg': 'drag magnets into the rows · right-click removes · X cut-away · A answer · drag orbit, wheel zoom',
    'mug': 'F fit  Space pause  R reset  H half turn · Shift-drag turns the mug, right-drag moves it · X cut-away',
    'mesh': 'G sample  E extract  Space pause  I midpoints / interpolated  R clear · A answer · drag orbit',
}
OUT_MESH = HERE / 'mug_mesh.obj'


class FitStopped(Exception):
    pass


def darker(c, f=0.75):
    return tuple(int(v * f) for v in c)


def lighter(c, f=0.55):
    return tuple(int(v + (255 - v) * f) for v in c)


def lighter_arr(cols, f):
    return cols + (255 - cols) * f


def point_in_tri(q, P):
    a, b, c = P
    d1 = (q[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (q[1] - b[1])
    d2 = (q[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (q[1] - c[1])
    d3 = (q[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (q[1] - a[1])
    neg = d1 < 0 or d2 < 0 or d3 < 0
    pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (neg and pos)


def dashed_rect(surf, color, rect, dash=4):
    r = pygame.Rect(rect)
    for x in range(r.left, r.right, dash * 2):
        pygame.draw.line(surf, color, (x, r.top), (min(x + dash, r.right), r.top))
        pygame.draw.line(surf, color, (x, r.bottom - 1), (min(x + dash, r.right), r.bottom - 1))
    for y in range(r.top, r.bottom, dash * 2):
        pygame.draw.line(surf, color, (r.left, y), (r.left, min(y + dash, r.bottom)))
        pygame.draw.line(surf, color, (r.right - 1, y), (r.right - 1, min(y + dash, r.bottom)))


def shade(base, nrm, cam, ambient=0.3):
    """Lambert + a little specular, light from over the viewer's left shoulder."""
    light = -0.35 * cam.right + 0.55 * cam.up - 0.75 * cam.fwd
    light /= np.linalg.norm(light)
    half = light - cam.fwd
    half /= np.linalg.norm(half)
    k = ambient + (1 - ambient) * np.clip(nrm @ light, 0, 1)
    spec = np.clip(nrm @ half, 0, 1) ** 40 * 0.3
    return np.clip(base * k[:, None] + 255 * spec[:, None], 0, 255)


# ---------------------------------------------------------------- the app

class App:
    def __init__(self, args):
        pygame.init()
        pygame.display.set_caption('Tutorial 7 — Signed Distance Functions')
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.font = pygame.font.SysFont(MONO, 14)
        self.bold = pygame.font.SysFont(MONO, 14, bold=True)
        self.small = pygame.font.SysFont(MONO, 12)
        self.big = pygame.font.SysFont(MONO, 16, bold=True)
        self.clock = pygame.time.Clock()
        self.args = args
        self.tab = 'torus'
        self.msg, self.msg_col = '', FG
        self.particles = []
        self.pops = {}
        self.sparked = set()
        self.success = {}                     # tab -> dict(t0, text, ...)
        self.hit = {}
        self.drag = None
        self.cam_drag = None
        self.last_t = time.time()
        self.last_move = 0.0
        self.version = 0                      # bumps whenever anything drawn by sphere tracing changes
        self._trace_key = None
        self.fitting = None
        self.reset_camera()
        self.cut = False
        self.focus = None                     # tab 4: (centre, dist) the camera glides to
        # data
        self.torus_obs, self.torus_center = L.torus_scene(args.seed)
        self.mug_obs, self.mug_center, _ = L.mug_scene(args.seed)
        self.losses = {'torus': [], 'mug': []}
        # student code
        self.student = importlib.import_module('sdf_student')
        self.watch = self.code_mtimes()
        self.last_watch = time.time()
        self.status = L.self_test(self.student)
        # tab 2: the magnet rows
        self.rows = {'C1': [], 'C2': [], 'C3': [], 'C4': []}
        self.checked = {k: None for k in self.rows}
        self.shown_row = 'C1'
        self.released = L.released_answers()
        self.released_t = time.time()
        if args.answers:
            for k in self.rows:
                if k in self.released:
                    self.rows[k] = list(self.released[k])
        # tab 3
        self.mug_params = dict(L.MUG_START)
        self.mug_pose = [0.0, np.zeros(3)]    # yaw (deg), translation
        self.fit_pose, self.fit_shape = True, True
        self.slider_drag = None
        # tab 4
        self.h = 0.4
        self.grid = None
        self.march = None
        self.march_state = 'idle'             # idle / sampled / running / paused / stuck / solved / done
        self.march_t = 0.0
        self.table = L.given_table()
        self.drawn = {}                       # class -> (cfg, triangles) drawn by the student
        self.skipped = set()
        self.edit_cfg = None                  # the sign pattern open in the cube editor
        self.edit_ok = False                  # its triangles are right
        self.partial = {}                     # unfinished drawings of patterns not open right now
        self.cube_zoom = 1.0
        self.cube_tris = []                   # the editor's triangles
        self.cube_path = []                   # edges picked so far in a press-drag-release
        self.cube_click = False
        self.cube_view = [-35.0, 22.0]
        self.cube_msg = ('', MUTED)
        self.cube_bad = ()
        self.cube_bad_t = 0.0
        self.interpolate = True
        self._mesh_key = None
        self.build_models()
        for k in self.rows:                   # already right at start-up (--answers): no fanfare
            self.checked[k] = self.row_correct(k) or None
        self.say('Tab 1: write T1 (torus_sdf) in sdf_student.py and save; the torus appears here.', MUTED)
        self.report_status(initial=True)

    # ------------------------------------------------------------ student code

    def code_mtimes(self):
        files = [HERE / 'sdf_student.py'] + sorted((HERE / 'answers').glob('s*.py'))
        return {str(f): f.stat().st_mtime for f in files if f.exists()}

    def check_reload(self):
        if time.time() - self.last_watch < 0.5 or self.fitting:
            return
        self.last_watch = time.time()
        now = self.code_mtimes()
        if now == self.watch:
            return
        self.watch = now
        for name in [m for m in sys.modules if m.startswith('answers.')]:
            del sys.modules[name]
        importlib.invalidate_caches()
        try:
            importlib.reload(self.student)
        except Exception as e:     # noqa: BLE001 - shown to the student
            self.say(f'sdf_student.py did not load: {type(e).__name__}: {e}', BAD)
            return
        self.status = L.self_test(self.student)
        self.build_models()
        self.report_status()

    def report_status(self, initial=False):
        fails = [(t, n, d) for t, n, s, d in self.status if s == 'fail']
        if fails:
            self.say('{} {}(): {}'.format(*fails[0]), BAD)
        elif not initial:
            todo = [t for t, _, s, _ in self.status if s == 'todo']
            self.say('reloaded sdf_student.py' + (f'; still to do: {" ".join(todo)}' if todo else
                                                   '; T1 and T2 pass their self-tests'), ACCENT)

    def status_of(self, tag):
        for t, _, s, d in self.status:
            if t == tag:
                return s, d
        return 'todo', ''

    def torus_fn(self):
        """The student's torus_sdf if it runs (even when its self-test fails: then the viewer shows why), else None."""
        s, _ = self.status_of('T1')
        if s in ('ok', 'answer'):
            return self.student.torus_sdf
        if s == 'todo':
            return None
        try:
            fn = self.student.torus_sdf
            out = fn(torch.zeros(4, 3), torch.tensor(3.0), torch.tensor(1.0))
            return fn if isinstance(out, torch.Tensor) and out.shape == (4,) else None
        except Exception:     # noqa: BLE001
            return None

    def build_models(self):
        """(Re)build the torus and the mug from the student's code and rows, keeping their parameters."""
        self.version += 1
        fn = self.torus_fn()
        old = getattr(self, 'torus_model', None)
        if fn is None:
            self.torus_model = None
        else:
            if old is not None:
                with torch.no_grad():
                    rv, t, s = old.rotvec.numpy().tolist(), old.t.numpy().tolist(), old.log_s.exp().numpy().tolist()
            else:
                rv, t, s = (0, 0, 0), (0, 0, 0), (1, 1, 1)
            self.torus_model = L.torus_model(fn, rv, t, s)
        self.build_mug()

    def ops(self):
        """{union: tree, ...} for the rows C1-C3 that parse."""
        out = {}
        for key, name in L.OP_ROWS:
            tree, _ = L.parse_op(self.rows[key])
            if tree is not None:
                out[name] = tree
        return out

    def mug_tree(self):
        """(tree, error) for C4 together with the operations it needs."""
        tree, err = L.parse_mug(self.rows['C4'])
        if tree is None:
            return None, 'finish the mug (C4) in tab 2 first'
        ops = self.ops()
        need = [f for _, f in L._calls(tree) if f not in ops]
        if need:
            key = {n: k for k, n in L.OP_ROWS}[need[0]]
            return None, f'the mug uses {need[0]}, but {key} does not run yet (tab 2)'
        return tree, None

    def build_mug(self):
        fn = self.torus_fn()
        tree, err = self.mug_tree()
        if fn is None:
            self.mug_model, self.mug_err = None, 'the handle is your torus: T1 (tab 1) has to work first'
            return
        if tree is None:
            self.mug_model, self.mug_err = None, err
            return
        self.mug_model = L.MugSDF(fn, self.ops(), tree, self.mug_params,
                                  rotvec=(0.0, 0.0, math.radians(self.mug_pose[0])), t=tuple(self.mug_pose[1]))
        self.mug_err = None
        self.version += 1

    def sync_mug(self):
        """Copy the mug model's parameters back (after a fit) so rebuilds keep them."""
        m = self.mug_model
        if m is None:
            return
        with torch.no_grad():
            self.mug_params = {k: float(v) for k, v in m.p.items()}
            R = m.R().numpy()
            self.mug_pose = [math.degrees(math.atan2(R[1, 0], R[0, 0])), m.t.numpy().copy()]

    # ------------------------------------------------------------ fitting (tabs 1 and 3)

    def start_fit(self):
        if self.fitting:
            return
        if self.tab == 'torus':
            model, obs = self.torus_model, self.torus_obs
            if model is None:
                self.say('T1 torus_sdf has to work first (python sdf_student.py)', AMBER)
                return
            params = list(model.parameters())
        elif self.tab == 'mug':
            model, obs = self.mug_model, self.mug_obs
            if model is None:
                self.say(self.mug_err, AMBER)
                return
            for p in model.pose_params():
                p.requires_grad_(self.fit_pose)
            for p in model.shape_params():
                p.requires_grad_(self.fit_shape)
            if not (self.fit_pose or self.fit_shape):
                self.say('tick pose or shape (or both): there is nothing to fit', AMBER)
                return
            params = [p for p in model.parameters() if p.requires_grad]
        else:
            return
        if self.status_of('T2')[0] == 'todo':
            self.say('T2 fit is a TODO in sdf_student.py', AMBER)
            return
        tab = self.tab
        self.success.pop(tab, None)
        self.losses[tab] = []
        self.fitting = dict(tab=tab, model=model, paused=False, steps=FIT_STEPS, params=params)
        self.say(f'fitting: your fit(), {FIT_STEPS} steps, lr {FIT_LR} · Space pauses, Esc stops', ACCENT)
        why = None
        try:
            self.student.fit(model, obs.fit_points(), steps=FIT_STEPS, lr=FIT_LR, report=self.report)
        except FitStopped:
            why = 'stopped'
        except NotImplementedError as e:
            why = f'{e}'
        except Exception as e:     # noqa: BLE001 - shown to the student
            why = f'your fit() stopped with {type(e).__name__}: {e}'
        finally:
            self.fitting = None
            for p in model.parameters():
                p.requires_grad_(True)
            self.version += 1
        if tab == 'mug':
            self.sync_mug()
        if why and why != 'stopped':
            self.say(why, BAD)
            return
        self.finish_fit(tab, model, obs, stopped=why == 'stopped')
        if not self.losses[tab]:
            self.say('your fit() never called report(i, loss), so the viewer could not follow it: ' + self.msg,
                     AMBER)

    def report(self, i, loss):
        """Called by the student's fit() after every step: draw this moment, handle the mouse and keys."""
        f = self.fitting
        if f is None:
            return
        try:
            loss = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
        except (TypeError, ValueError, RuntimeError):
            loss = float('nan')
        self.losses[f['tab']].append(loss)
        self.version += 1
        while True:
            for ev in pygame.event.get():
                self.handle_during_fit(ev)
            self.update()
            self.draw()
            self.clock.tick(60)
            if not self.fitting['paused']:
                break

    def handle_during_fit(self, ev):
        if ev.type == pygame.QUIT:
            raise SystemExit
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE:
                self.fitting['paused'] = not self.fitting['paused']
                self.say('paused · Space resumes, Esc stops' if self.fitting['paused'] else 'fitting', ACCENT)
            elif ev.key in (pygame.K_ESCAPE, pygame.K_r):
                raise FitStopped
            elif ev.key == pygame.K_c:
                self.reset_camera()
            elif ev.key == pygame.K_x:
                self.toggle_cut()
            return
        if ev.type == pygame.MOUSEBUTTONDOWN and pygame.Rect(VX, VY, VW, VH).collidepoint(ev.pos):
            if ev.button == 1:
                self.cam_drag = ('orbit', ev.pos)
            elif ev.button in (4, 5):
                self.zoom(ev.button == 4)
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            for rect, action in self.hit.get('buttons', []):
                if rect.collidepoint(ev.pos) and action in (self.toggle_pause, self.stop_fit):
                    action()
        elif ev.type == pygame.MOUSEBUTTONUP:
            self.cam_drag = None
        elif ev.type == pygame.MOUSEMOTION and self.cam_drag:
            self.orbit(ev.pos)

    def toggle_pause(self):
        if self.fitting:
            self.fitting['paused'] = not self.fitting['paused']
        elif self.tab == 'mesh':
            self.toggle_march()

    def stop_fit(self):
        if self.fitting:
            raise FitStopped

    def finish_fit(self, tab, model, obs, stopped=False):
        bound = self.bound(tab)
        try:
            ch = L.chamfer(model, obs, bound=bound)
        except Exception as e:     # noqa: BLE001
            self.say(f'could not measure the fit: {type(e).__name__}: {e}', BAD)
            return
        self.chamfer = getattr(self, 'chamfer', {})
        self.chamfer[tab] = ch
        word = 'stopped' if stopped else 'done'
        if ch < L.SUCCESS_CM and not stopped:
            what = 'torus' if tab == 'torus' else 'mug'
            self.celebrate(tab, f'✓ {what}: {ch * 10:.1f} mm from the depth points')
            self.say(f'{word}: the surface is {ch * 10:.1f} mm from the depth points on average', GOOD)
        else:
            hint = '' if stopped else (' · start it closer and fit again (Shift-drag turns it)' if tab == 'mug'
                                       else ' · check T2, or fit again')
            self.say(f'{word}: the surface is {ch * 10:.1f} mm from the depth points on average{hint}',
                     AMBER if ch < 0.5 else BAD)

    def reset_fit(self):
        self.success.pop(self.tab, None)
        if self.tab == 'torus':
            self.torus_model = None
            self.build_models()
            self.losses['torus'] = []
            self.say('the torus is back at its start: no rotation, at the origin, scale (1, 1, 1)', MUTED)
        elif self.tab == 'mug':
            self.mug_params = dict(L.MUG_START)
            self.mug_pose = [0.0, np.zeros(3)]
            self.build_mug()
            self.losses['mug'] = []
            self.say('the mug is back at its start sizes and pose', MUTED)
        elif self.tab == 'mesh':
            self.clear_mesh()

    # ------------------------------------------------------------ success

    def celebrate(self, tab, text):
        self.success[tab] = dict(t0=time.time(), text=text)
        self.version += 1
        rng = np.random.default_rng()
        clip = pygame.Rect(VX, VY, VW, VH)
        for _ in range(110):
            x = VX + VW / 2 + rng.normal(0, 90)
            self.particles.append(dict(
                kind='confetti', x=x, y=VY + VH * 0.55, vx=rng.normal(0, 170), vy=-rng.uniform(280, 560),
                life=rng.uniform(1.6, 2.6), col=CONFETTI[rng.integers(len(CONFETTI))],
                w=rng.uniform(4, 7), h=rng.uniform(8, 12), spin=rng.uniform(-9, 9), a=rng.uniform(0, 6.3),
                clip=clip))
        for p in self.particles:
            p.setdefault('max', p['life'])

    def sparks(self, x, y, n=16):
        rng = np.random.default_rng()
        for k in range(n):
            ang = 2 * math.pi * k / n + rng.uniform(-0.2, 0.2)
            v = rng.uniform(110, 210)
            life = rng.uniform(0.45, 0.7)
            self.particles.append(dict(kind='spark', x=x, y=y, vx=v * math.cos(ang), vy=v * math.sin(ang),
                                       life=life, max=life, col=GOOD if k % 2 else GLOW, r=rng.uniform(2.5, 4),
                                       clip=None))

    def update_particles(self, dt):
        alive = []
        for p in self.particles:
            p['life'] -= dt
            if p['life'] <= 0:
                continue
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            if p['kind'] == 'confetti':
                p['vy'] += 650 * dt
                p['vx'] *= 1 - 1.2 * dt
                p['a'] += p['spin'] * dt
            else:
                p['vx'] *= 1 - 4 * dt
                p['vy'] *= 1 - 4 * dt
            alive.append(p)
        self.particles = alive

    def draw_particles(self):
        scr = self.screen
        clip = scr.get_clip()
        for p in self.particles:
            scr.set_clip(p['clip'] or clip)
            if p['kind'] == 'confetti':
                c, s_ = math.cos(p['a']), math.sin(p['a'])
                w, h = p['w'] / 2, p['h'] / 2 * abs(math.cos(p['a'] * 0.7))
                pts = [(p['x'] + dx * c - dy * s_, p['y'] + dx * s_ + dy * c)
                       for dx, dy in ((-w, -h), (w, -h), (w, h), (-w, h))]
                pygame.draw.polygon(scr, p['col'], pts)
            else:
                r = p['r'] * (p['life'] / p['max'])
                pygame.draw.circle(scr, p['col'], (int(p['x']), int(p['y'])), max(1, int(r)))
        scr.set_clip(clip)

    def pop(self, key):
        self.pops[key] = time.time()
        self.sparked.discard(key)

    def pop_scale(self, key, ordinal):
        t0 = self.pops.get(key)
        if t0 is None:
            return 1.0
        t = time.time() - t0 - POP_STAGGER * ordinal
        if 0 <= t < POP_SECS:
            return 1 + 0.32 * math.sin(math.pi * t / POP_SECS)
        return 1.0

    def draw_banner(self, tab):
        s = self.success[tab]
        f = min(1.0, (time.time() - s['t0']) / 0.5)
        t = self.big.render(s['text'], True, WHITE)
        w, h = t.get_width() + 28, t.get_height() + 12
        x, y = VX + (VW - w) // 2, VY + 12 - int(20 * (1 - f))
        pygame.draw.rect(self.screen, GOOD, (x, y, w, h), border_radius=h // 2)
        self.screen.blit(t, (x + 14, y + 6))
        sub = s.get('sub')
        if sub:
            r = self.small.render(sub, True, darker(GOOD))
            self.screen.blit(r, (VX + (VW - r.get_width()) // 2, y + h + 4))

    def success_ease(self, tab, secs=SNAP_SECS):
        s = self.success.get(tab)
        if not s:
            return 0.0
        f = min(1.0, (time.time() - s['t0']) / secs)
        return 1 - (1 - f) ** 3

    def say(self, msg, col=FG):
        self.msg, self.msg_col = msg, col

    # ------------------------------------------------------------ camera

    def cam_tab(self):
        return self.tab

    def view_center(self):
        if self.tab == 'torus':
            return self.torus_center
        if self.tab == 'csg':
            return np.array([1.5, 0.0, 4.75]) if self.shown_row == 'C4' else np.array([0.15, 0.0, 0.45])
        if self.tab == 'mesh' and self.focus is not None:
            return self.focus[0]
        return self.mug_center

    def view_cam(self, w=VW, h=VH):
        yaw, pitch, dist = self.cams[self.tab]
        if self.tab == 'mesh' and self.focus is not None:
            dist = self.focus[1]
        if self.tab == 'csg' and self.shown_row != 'C4':
            dist *= 0.55                               # the two spheres are smaller than the mug
        if self.tab == 'csg' and 'csg' in self.success:
            yaw += 360 * self.success_ease('csg', 2.4)
        c = self.view_center()
        y, p = math.radians(yaw), math.radians(pitch)
        pos = c + dist * np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
        return L.Camera(pos, c, w, h, 40.0)

    def reset_camera(self):
        self.cams = {'torus': [35.0, 55.0, 30.0], 'csg': [-60.0, 22.0, 32.0], 'mug': [-60.0, 22.0, 30.0],
                     'mesh': [-60.0, 28.0, 30.0]}
        self.moved()

    def orbit(self, pos):
        mode, last = self.cam_drag
        dx, dy = pos[0] - last[0], pos[1] - last[1]
        self.cam_drag = (mode, pos)
        if mode == 'orbit':
            c = self.cams[self.tab]
            c[0] -= dx * 0.4
            c[1] = max(-85.0, min(85.0, c[1] + dy * 0.4))
            self.moved()
        elif mode == 'turn' and self.tab == 'mug':
            self.mug_pose[0] += dx * 0.5
            self.build_mug()
            self.success.pop('mug', None)
        elif mode == 'move' and self.tab == 'mug':
            cam = self.view_cam()
            s = self.cams['mug'][2] / cam.focal
            right = cam.right * np.array([1, 1, 0])
            fwd = np.cross([0, 0, 1.0], cam.right)
            self.mug_pose[1] = self.mug_pose[1] + (right * dx - fwd * dy) * s
            self.build_mug()
            self.success.pop('mug', None)

    def zoom(self, closer):
        if self.tab == 'mesh' and self.focus is not None:     # close up on a stuck cell
            c, d = self.focus
            self.focus = (c, max(2.0, min(80.0, d / 1.1 if closer else d * 1.1)))
        else:
            c = self.cams[self.tab]
            c[2] = max(8.0, min(80.0, c[2] / 1.1 if closer else c[2] * 1.1))
        self.moved()

    def moved(self):
        self.last_move = time.time()

    def toggle_cut(self):
        if self.tab in ('csg', 'mug'):
            self.cut = not self.cut
            self.say('cut-away: the near half is cut off (grey = the cut)' if self.cut else 'cut-away off', MUTED)

    # ------------------------------------------------------------ what the viewer traces

    def bound(self, tab):
        if tab == 'torus':
            m = self.torus_model
            if m is None:
                return None
            with torch.no_grad():
                return m.t.numpy().copy(), (L.TORUS_R + L.TORUS_r) * float(m.log_s.exp().max()) + 0.3
        if tab == 'csg':
            return (np.array([1.5, 0, 4.75]), 9.0) if self.shown_row == 'C4' else (np.array([0.15, 0, 0.45]), 4.0)
        if tab in ('mug', 'mesh') and self.mug_model is not None:
            return self.mug_model.bound()
        return None

    def ops_spheres(self, x):
        (ca, ra), (cb, rb) = L.OPS_DEMO['a'], L.OPS_DEMO['b']
        return {'a': L.sphere_sdf(x, ca, ra), 'b': L.sphere_sdf(x, cb, rb)}

    def csg_mug(self):
        """Tab 2's mug: your C4 with your C1-C3, at the start sizes, no pose."""
        tree, err = self.mug_tree()
        fn = self.torus_fn()
        if tree is None:
            return None, err
        if fn is None:
            return None, 'the handle is your torus: T1 (tab 1) has to work first'
        key = (tuple(self.rows['C1']), tuple(self.rows['C2']), tuple(self.rows['C3']), tuple(self.rows['C4']), id(fn))
        if getattr(self, '_csg_mug_key', None) != key:
            self._csg_mug_key = key
            self._csg_mug = L.MugSDF(fn, self.ops(), tree)
        return self._csg_mug, None

    def scene_sdf(self):
        """(sdf, parts, base colour, note) for the current tab, or (None, None, None, why not)."""
        tab = self.tab
        if tab == 'torus':
            if self.torus_model is None:
                s, d = self.status_of('T1')
                why = 'T1 torus_sdf is a TODO: write it in sdf_student.py and save' if s == 'todo' else f'T1: {d}'
                return None, None, None, why
            s, d = self.status_of('T1')
            return self.torus_model, None, SURF, (f'T1 fails its self-test: {d}' if s == 'fail' else None)
        if tab == 'csg':
            if self.shown_row == 'C4':
                m, err = self.csg_mug()
                if m is None:
                    return None, None, None, err
                return m, m.parts, None, None
            tree, err = L.parse_op(self.rows[self.shown_row])
            if tree is None:
                both = lambda x: torch.minimum(*self.ops_spheres(x).values())  # noqa: E731
                return both, self.ops_spheres, None, f'{self.shown_row} does not run yet: here are a and b'
            return (lambda x: L.evaluate(tree, self.ops_spheres(x))), self.ops_spheres, None, None
        if tab in ('mug', 'mesh'):
            if self.mug_model is None:
                return None, None, None, self.mug_err
            m = self.mug_model
            return m, m.parts, None, None
        return None, None, None, None

    @torch.no_grad()
    def traced(self):
        """The sphere-traced image of the current tab (cached): (surface, depth, cam) or None."""
        moving = self.fitting is not None or time.time() - self.last_move < 0.3 or bool(self.particles) or \
            (self.tab in self.success and time.time() - self.success[self.tab]['t0'] < 2.5)
        res = LO_RES if moving else HI_RES
        cam = self.view_cam()
        ease = round(self.success_ease(self.tab), 2)
        key = (self.tab, self.shown_row, self.version, tuple(np.round(cam.pos, 4)), res, self.cut, ease,
               self.tab == 'mesh' and self.grid is not None)
        if key == self._trace_key:
            return self._trace
        sdf, parts, base, note = self.scene_sdf()
        self.view_note = note
        if sdf is None:
            self._trace_key, self._trace = key, None
            return None
        small = cam.resized(*res)
        bound = self.bound(self.tab)
        cut_n = None
        f = sdf
        if self.cut and self.tab in ('csg', 'mug'):
            c = self.view_center() if self.tab == 'csg' else bound[0]
            n = (cam.pos - c) * np.array([1, 1, 0])
            cut_n = n / max(np.linalg.norm(n), 1e-9)
            cn, cc = torch.as_tensor(cut_n, dtype=torch.float32), torch.as_tensor(c, dtype=torch.float32)
            f = lambda x: torch.maximum(sdf(x), (x - cc) @ cn)  # noqa: E731
        d, cos = small.rays()
        try:
            t = L.sphere_trace(f, small.pos, d, 1.0, 120.0, bound=bound)
        except Exception as e:     # noqa: BLE001 - the student's SDF failed
            self.view_note = f'your SDF failed: {type(e).__name__}: {e}'
            self._trace_key, self._trace = key, None
            return None
        hit = np.isfinite(t)
        X = small.pos + d[hit] * t[hit, None]
        nrm = L.sdf_normals(f, X)
        if parts is not None:
            xt = torch.as_tensor(X, dtype=torch.float32)
            pv = parts(xt)
            names = list(pv)
            which = torch.stack([pv[k].abs() for k in names], 1).argmin(1).numpy()
            cols = np.array([PART_COL[k] for k in names], float)[which]
        else:
            cols = np.tile(np.array(base, float), (len(X), 1))
        if cut_n is not None and len(X):
            c = self.view_center() if self.tab == 'csg' else bound[0]
            on_cut = np.abs((X - c) @ cut_n) < 3e-3 * (1 + 0.05 * t[hit])
            cols[on_cut] = CUT
        if ease > 0:
            cols = cols * (1 - ease) + np.array(SOLID, float) * ease
        rgb = np.tile(np.array(VIEW_BG, np.uint8), (res[0] * res[1], 1))
        rgb[hit] = shade(cols, nrm, small).astype(np.uint8)
        img = rgb.reshape(res[1], res[0], 3)
        surf = pygame.transform.smoothscale(pygame.surfarray.make_surface(img.swapaxes(0, 1)), (VW, VH))
        depth = (t * cos).reshape(res[1], res[0])
        self._trace_key, self._trace = key, (surf, depth, cam)
        return self._trace

    # ------------------------------------------------------------ points, cameras

    def obs(self):
        return self.torus_obs if self.tab == 'torus' else self.mug_obs

    def model(self):
        return self.torus_model if self.tab == 'torus' else self.mug_model

    def shown_points(self):
        """Up to 9 000 of the depth points (a fixed sample) and their camera."""
        obs = self.obs()
        key = id(obs)
        if getattr(self, '_shown_key', None) != key:
            n = min(9000, len(obs.points))
            idx = np.sort(np.random.default_rng(1).choice(len(obs.points), n, replace=False))
            self._shown_key, self._shown = key, idx
        return obs.points[self._shown], obs.view[self._shown]

    @torch.no_grad()
    def point_colors(self, P, view):
        cols = np.array(VIEW_COLS, float)[view % len(VIEW_COLS)]
        m = self.model()
        if m is None:
            return cols, P
        key = (self.tab, self.version)
        if getattr(self, '_res_key', None) != key:
            try:
                f = m(torch.as_tensor(P, dtype=torch.float32)).numpy()
            except Exception:     # noqa: BLE001
                f = np.zeros(len(P))
            self._res_key, self._res = key, f
        f = self._res
        tint = np.clip((np.abs(f) - 0.1) / 0.6, 0, 1)[:, None]
        cols = cols * (1 - tint) + np.array(BAD, float) * tint
        ease = self.success_ease(self.tab)
        if ease > 0:
            nk = (self.tab, 'nrm', self.version)
            if getattr(self, '_nrm_key', None) != nk:
                self._nrm_key, self._nrm = nk, L.sdf_normals(m, P)
            P = P - (ease * f)[:, None] * self._nrm
            cols = cols * (1 - ease) + np.array(lighter(SOLID, 0.5), float) * ease
        return cols, P

    def splat(self, surf, X, cols, cam, depth=None, size=2):
        """Draw points as small squares straight into the pixels; hidden behind the traced surface."""
        if len(X) == 0:
            return
        uv, z = cam.project(X)
        x, y = uv[:, 0].astype(int), uv[:, 1].astype(int)
        ok = (z > 0.1) & (x >= 0) & (x < VW - 1) & (y >= 0) & (y < VH - 1)
        if depth is not None:
            h, w = depth.shape
            dz = depth[np.clip(y * h // VH, 0, h - 1), np.clip(x * w // VW, 0, w - 1)]
            ok &= z <= dz + 0.25
        x, y, z, c = x[ok], y[ok], z[ok], cols[ok].astype(np.uint8)
        order = np.argsort(-z)                    # far first, near points win
        x, y, c = x[order], y[order], c[order]
        arr = pygame.surfarray.pixels3d(surf)
        for dx in range(size):
            for dy in range(size):
                arr[x + dx, y + dy] = c
        del arr

    def draw_frusta(self, cam, obs):
        scr = self.screen
        for k, c in enumerate(obs.cams):
            col = VIEW_COLS[k % len(VIEW_COLS)]
            corners = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                d = c.fwd + a * (c.W / 2 / c.focal) * c.right + b * (c.H / 2 / c.focal) * c.up
                corners.append(c.pos + 2.6 * d)
            pts, z = cam.project(np.array([c.pos] + corners))
            if (z < 0.5).any():
                continue
            pts = pts + [VX, VY]
            for q in pts[1:]:
                pygame.draw.aaline(scr, col, pts[0], q)
            pygame.draw.lines(scr, col, True, pts[1:].tolist(), 2)
            lab = self.small.render(str(k + 1), True, col)
            scr.blit(lab, pts[0] + [4, -14])

    # ------------------------------------------------------------ the viewer

    def draw_viewer(self):
        scr = self.screen
        rect = pygame.Rect(VX, VY, VW, VH)
        clip = scr.get_clip()
        scr.set_clip(rect)
        if self.tab == 'mesh':
            self.draw_mesh_view()
        else:
            tr = self.traced()
            surf = tr[0].copy() if tr else self._blank()
            cam = self.view_cam()
            if self.tab in ('torus', 'mug'):
                P, view = self.shown_points()
                cols, P = self.point_colors(P, view)
                self.splat(surf, P, cols, cam, tr[1] if tr else None)
            scr.blit(surf, (VX, VY))
            if self.tab in ('torus', 'mug') and not self.success.get(self.tab):
                self.draw_frusta(cam, self.obs())
        self.draw_gizmo()
        scr.set_clip(clip)
        self.draw_view_labels()
        if self.tab in self.success:
            self.draw_banner(self.tab)
        pygame.draw.rect(scr, LINE, rect, 1)

    def _blank(self):
        s = pygame.Surface((VW, VH))
        s.fill(VIEW_BG)
        return s

    def draw_view_labels(self):
        scr = self.screen
        note = getattr(self, 'view_note', None)
        if self.tab == 'mesh':
            note = self.mesh_note()
        y = VY + 8
        if self.tab in self.success:
            y = VY + 70                                # below the banner
        elif self.tab == 'csg':
            for k in (('a', 'b') if self.shown_row != 'C4' else ('body', 'cavity', 'handle')):
                pygame.draw.circle(scr, PART_COL[k], (VX + 14, y + 8), 5)
                scr.blit(self.small.render(k, True, FG), (VX + 24, y + 1))
                y += 18
            shown = 'mug (C4)' if self.shown_row == 'C4' else dict(L.OP_ROWS)[self.shown_row] + f'(a, b)  ({self.shown_row})'
            scr.blit(self.small.render(f'showing: {shown} · click a row to show it', True, MUTED), (VX + 10, y + 2))
            y += 20
        elif self.tab in ('torus', 'mug') and not self.success.get(self.tab):
            scr.blit(self.small.render('depth points, coloured by camera; red = far from your surface', True, MUTED),
                     (VX + 10, y))
            y += 18
        if note:
            for k, line in enumerate(self.wrap(note, VW - 60, self.font)[:3]):
                r = self.font.render(line, True, AMBER)
                x, yy = VX + (VW - r.get_width()) // 2, y + 8 + 20 * k
                pygame.draw.rect(scr, AMBER_BG, (x - 8, yy - 2, r.get_width() + 16, 20), border_radius=4)
                scr.blit(r, (x, yy))
        hint = 'drag: orbit · wheel: zoom'
        if self.tab == 'mug':
            hint += ' · Shift+drag: turn the mug · right-drag: move it'
        scr.blit(self.small.render(hint, True, FAINT), (VX + VW - 8 - self.small.size(hint)[0], VY + VH - 18))

    def draw_gizmo(self):
        scr = self.screen
        o = np.array([VX + 40, VY + VH - 40])
        cam = self.view_cam()
        for k, (col, name) in enumerate(((BAD, 'x'), (GOOD, 'y'), (ACCENT, 'z'))):
            e = np.eye(3)[k]
            v = np.array([e @ cam.right, -(e @ cam.up)]) * 22
            pygame.draw.line(scr, col, o, o + v, 2)
            scr.blit(self.small.render(name, True, col), o + v * 1.25 - 4)
        # 1 cm scale bar at the view centre's depth
        s = cam.focal / self.cams[self.tab][2] if not (self.tab == 'mesh' and self.focus) else cam.focal / self.focus[1]
        x0, y0 = VX + VW - 30 - s, VY + VH - 34
        pygame.draw.line(scr, FG, (x0, y0), (x0 + s, y0), 2)
        lab = self.small.render('1 cm', True, FG)
        scr.blit(lab, (x0 + s / 2 - lab.get_width() / 2, y0 - 16))

    # ------------------------------------------------------------ tab 4: grid and march

    def mug_box(self):
        """World-space box around the mug (from its sizes and pose), plus a margin of one grid step."""
        m = self.mug_model
        P = m.numbers()
        lo = np.array([-P['rb'], -P['rb'], 0.0])
        hi = np.array([P['rb'] + P['Rh'] + P['rh'], P['rb'], P['h']])
        corners = np.array([[(lo, hi)[i >> k & 1][k] for k in range(3)] for i in range(8)])
        with torch.no_grad():
            W = corners @ m.R().numpy().T + m.t.numpy()
        return W.min(0) - self.h, W.max(0) + self.h

    def sample_grid(self):
        if self.fitting:
            return
        if self.mug_model is None:
            self.say(self.mug_err, AMBER)
            return
        lo, hi = self.mug_box()
        with torch.no_grad():
            self.grid = L.Grid(self.mug_model, lo, hi, self.h)
        self.march, self.march_state, self.focus = None, 'sampled', None
        self.success.pop('mesh', None)
        self.close_case()
        g = self.grid
        crossing = int((g.cls > 0).sum())
        self.say(f'sampled your mug at {np.prod(g.n + 1):,} grid points ({self.h * 10:g} mm apart): '
                 f'{crossing:,} of {np.prod(g.n):,} cells have a sign change · E extracts', ACCENT)
        self.version += 1

    def clear_mesh(self):
        self.march, self.focus = None, None
        self.march_state = 'sampled' if self.grid is not None else 'idle'
        self.success.pop('mesh', None)
        self.close_case()
        for k in list(self.skipped):
            self.unskip(k)
        self.say('mesh cleared · E extracts again (the classes you drew stay drawn)', MUTED)

    def extract(self):
        if self.grid is None:
            self.sample_grid()
            if self.grid is None:
                return
        if self.march_state == 'solved':
            self.continue_march()
            return
        if self.march is not None and self.march_state in ('running', 'stuck'):
            return
        if self.march is None or self.march_state == 'done':
            for k in list(self.skipped):
                self.unskip(k)
            self.march = L.March(self.grid, self.table, self.interpolate)
            self.success.pop('mesh', None)
        self.march_state, self.march_t = 'running', time.time()
        self.say('marching through the cells, one slab at a time, from the bottom up', MUTED)

    def toggle_march(self):
        if self.march_state == 'running':
            self.march_state = 'paused'
            self.say('march paused · Space resumes', MUTED)
        elif self.march_state == 'paused':
            self.march_state = 'running'
        elif self.march_state == 'solved':
            self.continue_march()
        elif self.march_state in ('sampled', 'idle', 'done'):
            self.extract()

    def update_march(self, now):
        if self.march_state != 'running' or now < self.march_t:
            return
        r = self.march.step()
        self.march_t = now + MARCH_DELAY
        if r == 'stuck':
            self.march_state = 'stuck'
            cfg = self.stuck_cfg()
            k = L.CLASS_OF[cfg]
            self.open_case(cfg)
            self.say(f'stuck at case {k} ({L.CLASS_INFO[k]}): no triangles for it yet · draw them in the cube', AMBER)
            i, j, kk = self.march.stuck
            c = self.grid.X[i, j, kk] + self.grid.h / 2
            self.focus = (c, 7.0)
        elif r == 'done':
            self.finish_mesh()

    def stuck_cfg(self):
        i, j, k = self.march.stuck
        return int(self.grid.cfg[i, j, k])

    def continue_march(self):
        """After a case is right: march on (the editor closes until the next missing case)."""
        if self.march_state != 'solved':
            return
        self.close_case()
        self.march_state, self.march_t, self.focus = 'running', time.time(), None
        self.say('marching on', MUTED)

    # which sign pattern the cube editor shows

    def open_case(self, cfg, tris=None):
        """Show cfg in the cube editor, with tris or its unfinished drawing (kept per pattern)."""
        if self.edit_cfg is not None and not self.edit_ok:
            self.partial[self.edit_cfg] = list(self.cube_tris)
        self.edit_cfg = cfg
        self.cube_tris = list(tris) if tris is not None else self.partial.pop(cfg, [])
        self.cube_path, self.cube_click, self.cube_bad = [], False, ()
        ok, msg, _ = L.check_triangles(cfg, self.cube_tris) if self.cube_tris else (None, '', ())
        self.edit_ok = ok is True
        if self.edit_ok:
            self.cube_msg = ('✓ right. Change it if you like: it is checked again after every change', GOOD)
        elif self.cube_tris:
            self.cube_msg = (msg, BAD if ok is False else MUTED)
        else:
            self.cube_msg = ('press on a crossing point, drag through a second, release on a third', MUTED)

    def close_case(self):
        if self.edit_cfg is not None and not self.edit_ok and self.cube_tris:
            self.partial[self.edit_cfg] = list(self.cube_tris)
        self.edit_cfg, self.edit_ok = None, False
        self.cube_tris, self.cube_path, self.cube_click = [], [], False

    def open_class(self, k):
        """A click on a case in the list: open the stuck case, or one you have drawn, in the editor."""
        if self.march_state == 'stuck' and L.CLASS_OF[self.stuck_cfg()] == k:
            if self.edit_cfg != self.stuck_cfg():
                self.open_case(self.stuck_cfg())
            return
        if k not in self.drawn:
            what = 'given: the program has it already' if k in L.GIVEN_CLASSES or k == 0 else \
                'not drawn yet: the march opens it when it meets it'
            self.say(f'case {k} is {what}', MUTED)
            return
        src, tris = self.drawn[k]
        cfg = src if L.CLASS_OF[src] == k else L.class_representative(k)
        self.open_case(cfg, L.map_triangles(tris, src, cfg))
        self.say(f'case {k}: your drawing · change it and it is checked again', MUTED)

    def finish_mesh(self):
        self.march_state, self.focus = 'done', None
        V, F = self.march.mesh()
        wt = L.watertight(F)
        if self.skipped:
            holes = sum(int((self.grid.cls == k).sum()) for k in self.skipped)
            self.say(f'done, with {holes} holes: case{"s" if len(self.skipped) > 1 else ""} '
                     f'{", ".join(map(str, sorted(self.skipped)))} skipped · R clears, then draw them', AMBER)
            return
        L.write_obj(OUT_MESH, V, F, 'Tutorial 7: your mug, fitted (tab 3) and meshed by your marching cubes (tab 4).')
        self.celebrate('mesh', f'✓ {len(F):,} triangles, {"watertight" if wt else "with gaps"} · saved {OUT_MESH.name}')
        self.success['mesh']['sub'] = 'I: midpoints / interpolated · G samples again · drag to look around'
        self.say(f'marching cubes done: {len(V):,} vertices, {len(F):,} triangles, saved to {OUT_MESH.name}', GOOD)

    def skip_case(self):
        if self.march_state != 'stuck':
            return
        k = L.CLASS_OF[self.stuck_cfg()]
        self.skipped.add(k)
        for cfg in range(256):
            if L.CLASS_OF[cfg] == k and cfg not in self.table:
                self.table[cfg] = []
        self.partial.pop(self.stuck_cfg(), None)
        self.close_case()
        self.march_state, self.march_t, self.focus = 'running', time.time(), None
        self.say(f'case {k} skipped: its cells stay empty, so the mesh will have holes there', AMBER)

    def unskip(self, k):
        self.skipped.discard(k)
        for cfg in range(256):
            if L.CLASS_OF[cfg] == k and self.table.get(cfg) == []:
                del self.table[cfg]

    def solve_case(self, cfg, tris):
        """The student's triangles for cfg are right: copy them to every sign pattern of the class."""
        k = L.CLASS_OF[cfg]
        classes = {k, L.MIRROR_OF.get(k, k)} | {a for a, b in L.MIRROR_OF.items() if b == k}
        n = 0
        for c2 in range(256):
            if L.CLASS_OF[c2] in classes:
                self.table[c2] = L.map_triangles(tris, cfg, c2)
                n += 1
        for c in classes:
            self.drawn[c] = (cfg, list(tris))
            self.pop(f'X{c}')
        for c2 in list(self.partial):
            if L.CLASS_OF[c2] in classes:
                del self.partial[c2]
        extra = ' (and its mirror image, case 14)' if 11 in classes and k != 14 else \
            (' (and its mirror image, case 11)' if k == 14 else '')
        then = ''
        if self.march_state == 'stuck' and L.CLASS_OF[self.stuck_cfg()] in classes:
            self.march_state = 'solved'
            then = ' · Continue (E) marches on'
        elif self.march is not None and self.march.tris:
            then = ' · R then E extracts again with it'
        self.say(f'✓ case {k}{extra}: your {len(tris)} triangle{"s" if len(tris) > 1 else ""} fill '
                 f'{n} of the 256 sign patterns{then}', GOOD)
        self.cube_msg = ('✓ every inside corner is cut off from every outside corner', GOOD)
        self.edit_ok = True
        self.pop('cube')

    def mesh_note(self):
        if self.mug_model is None:
            return self.mug_err
        if self.grid is None:
            return 'G samples your mug (tab 3) on a grid'
        return None

    def mesh_arrays(self):
        m = self.march
        if m is None:
            return None
        key = (id(m), sum(len(t) for t in m.tris), m.interpolate)
        if key != self._mesh_key:
            self._mesh_key = key
            self._mesh = m.mesh()
        return self._mesh

    def draw_mesh_view(self):
        scr = self.screen
        cam = self.view_cam()
        surf = self._blank()
        g = self.grid
        if g is None:
            tr = self.traced()
            if tr:
                surf = tr[0].copy()
                surf.set_alpha(110)
                base = self._blank()
                base.blit(surf, (0, 0))
                surf = base
            scr.blit(surf, (VX, VY))
            return
        # signs at the grid points near the surface, above the slab the march has reached
        kmin = self.march.k if self.march is not None else 0
        stuck = self.march_state in ('stuck', 'solved')
        key = (id(g), kmin, stuck)
        if getattr(self, '_dots_key', None) != key:
            near = np.abs(g.F) < g.h * 0.9
            near[:, :, :kmin] = False
            if stuck:                                  # only the slab the march is in
                near[:, :, kmin + 2:] = False
            X = g.X[near]
            cols = np.where((g.F[near] < 0)[:, None], np.array(INSIDE), np.array(OUTSIDE)).astype(float)
            self._dots_key, self._dots = key, (X, lighter_arr(cols, 0.25))
        X, cols = self._dots
        if self.march_state != 'done':
            self.splat(surf, X, cols, cam, size=2)
        scr.blit(surf, (VX, VY))
        mesh = self.mesh_arrays()
        if mesh is not None and len(mesh[1]):
            self.draw_triangles(mesh[0], mesh[1], cam)
        if self.march is not None and self.march_state in ('running', 'paused'):
            self.draw_slab(cam)
        if self.march_state in ('stuck', 'solved'):
            self.draw_stuck(cam)

    def draw_triangles(self, V, F, cam, base=None):
        uv, z = cam.project(V)
        uv = uv + [VX, VY]
        A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        nrm = np.cross(B - A, C - A)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        ctr = (A + B + C) / 3
        facing = ((cam.pos - ctr) * nrm).sum(1) > 0
        F, nrm, zc = F[facing], nrm[facing], z[F[facing]].mean(1)
        ease = self.success_ease('mesh')
        col = np.array(base or SURF, float) * (1 - ease) + np.array(SOLID, float) * ease
        cols = shade(np.tile(col, (len(F), 1)), nrm, cam, ambient=0.45).astype(int)
        order = np.argsort(-zc)
        tri = uv[F[order]].tolist()
        cl = cols[order].tolist()
        draw = pygame.draw.polygon
        scr = self.screen
        for p, c in zip(tri, cl):
            draw(scr, c, p)

    def draw_slab(self, cam):
        g = self.grid
        k = min(self.march.k, g.n[2])
        z = g.lo[2] + k * g.h
        lo, hi = g.lo, g.lo + g.n * g.h
        sq = np.array([[lo[0], lo[1], z], [hi[0], lo[1], z], [hi[0], hi[1], z], [lo[0], hi[1], z]])
        uv, d = cam.project(sq)
        if (d > 0.1).all():
            pygame.draw.lines(self.screen, ACCENT, True, (uv + [VX, VY]).tolist(), 2)

    def cell_corners(self, ijk):
        i, j, k = ijk
        return np.array([self.grid.X[i + dx, j + dy, k + dz] for dx, dy, dz in L.CORNERS])

    def draw_stuck(self, cam):
        scr = self.screen
        cfg = self.stuck_cfg()
        k = L.CLASS_OF[cfg]
        solved = self.march_state == 'solved'
        hole = GOOD if solved else HOLE                # green: filled once the march goes on
        # every other cell of this case: an orange square (a hole in the mesh)
        holes = self.march.holes(k)
        if len(holes):
            ctr = self.grid.X[tuple(holes.T)] + self.grid.h / 2
            uv, z = cam.project(ctr)
            over = pygame.Surface((VW, VH), pygame.SRCALPHA)
            for (x, y), zz in zip(uv[:600].tolist(), z[:600].tolist()):
                if zz > 0.5:
                    s = max(4, min(10, int(cam.focal * self.grid.h / zz * 0.5)))
                    pygame.draw.rect(over, (*hole, 110), (x - s // 2, y - s // 2, s, s))
                    pygame.draw.rect(over, (*hole, 230), (x - s // 2, y - s // 2, s, s), 1)
            scr.blit(over, (VX, VY))
        # the stuck cell: a pulsing wire cube with its corner signs
        Cw = self.cell_corners(self.march.stuck)
        uv, z = cam.project(Cw)
        uv = uv + [VX, VY]
        pulse = 0.5 + 0.5 * math.sin(time.time() * 6)
        wc = tuple(int(hole[i] * (1 - 0.4 * pulse * (not solved))) for i in range(3))
        for a, b in L.EDGES:
            pygame.draw.line(scr, wc, uv[a], uv[b], 3)
        # the triangles drawn so far, in the viewer too
        tris = self.cube_tris if self.edit_cfg == cfg else self.table.get(cfg) or self.partial.get(cfg, [])
        for t in tris:
            P = [uv[list(L.EDGES[e])].mean(0) for e in t]
            pygame.draw.polygon(scr, lighter(SURF, 0.2), P)
            pygame.draw.polygon(scr, darker(SURF), P, 1)
        for i in range(8):
            col = INSIDE if cfg >> i & 1 else OUTSIDE
            pygame.draw.circle(scr, WHITE, uv[i], 7)
            pygame.draw.circle(scr, col, uv[i], 6)


    # ------------------------------------------------------------ tab 2: magnet rows

    def row_tree(self, key):
        return L.parse_mug(self.rows[key]) if key == 'C4' else L.parse_op(self.rows[key])

    def ops_correct(self):
        return all(L.check_op(name, self.row_tree(key)[0]) for key, name in L.OP_ROWS)

    def row_correct(self, key):
        tree, err = self.row_tree(key)
        if tree is None:
            return False
        if key == 'C4':
            return L.check_mug(tree, None) and self.ops_correct()
        return L.check_op(dict(L.OP_ROWS)[key], tree)

    def row_note(self, key):
        """(text, colour) under a row: what it does, or a hint."""
        toks = self.rows[key]
        if not toks:
            return '', MUTED
        tree, err = self.row_tree(key)
        if tree is None:
            return f'? {err}', AMBER
        if key != 'C4':
            return f'→ {L.describe_op(tree)}', GOOD if self.checked[key] else MUTED
        if err:
            return f'→ {err}', AMBER
        tree_ok = L.check_mug(tree, None)
        if tree_ok and not self.ops_correct():
            bad = [k for k, n in L.OP_ROWS if not L.check_op(n, self.row_tree(k)[0])]
            return f'→ the order is right; {", ".join(bad)} not yet', AMBER
        if tree_ok:
            return '→ a mug: body and handle joined, then the cavity carved out of both', GOOD
        return f'→ {self.mug_diagnosis()}', AMBER

    @torch.no_grad()
    def mug_diagnosis(self):
        m, err = self.csg_mug()
        if m is None:
            return err
        P = L.MUG_START
        probes = {'inside the handle, in the cup': [P['rb'] - P['Rh'], 0, P['zh']],
                  'the middle of the cup': [0, 0, P['h'] * 0.6], 'the base': [0, 0, P['b'] / 2],
                  'the outer handle': [P['rb'] + P['Rh'], 0, P['zh']], 'the wall': [-(P['rb'] - P['w'] / 2), 0, 3]}
        f = {k: float(m(torch.tensor([v], dtype=torch.float32))[0]) for k, v in probes.items()}
        if f['the middle of the cup'] < 0:
            return 'the cup is solid: nothing is carved out of the middle'
        if f['inside the handle, in the cup'] < 0:
            return 'the handle pokes into the cup (X shows the inside)'
        if f['the base'] > 0 or f['the wall'] > 0:
            return 'the body is missing (or carved away)'
        if f['the outer handle'] > 0:
            return 'the handle is missing'
        return 'not the mug yet: see the viewer'

    def changed(self, key):
        """A row changed: re-check it (and C4, which uses C1-C3), rebuild the mug, pop if now right."""
        self.version += 1
        keys = [key] + (['C4'] if key != 'C4' else [])
        for k in keys:
            was = self.checked[k]
            ok = self.row_correct(k)
            self.checked[k] = True if ok else None
            if ok and was is not True:
                self.pop(k)
                if k == 'C4':
                    self.celebrate('csg', '✓ mug SDF')
                    self.success['csg']['sub'] = 'X cuts it open · tab 3 fits it to a real mug'
                    self.shown_row = 'C4'
                    self.say('✓ C4: your mug SDF is right. Tab 3 fits it to the depth maps of a real mug.', GOOD)
                else:
                    self.say(f'✓ {k}: {dict(L.OP_ROWS)[k]} is right', GOOD)
            if not ok and k == 'C4':
                self.success.pop('csg', None)
        self.build_mug()

    def fill_answer(self, key):
        if key not in self.released:
            self.say(f'no released answer for {key} yet', AMBER)
            return
        self.rows[key] = list(self.released[key])
        self.shown_row = key
        self.changed(key)

    def magnet_at(self, pos):
        for rect, where in self.hit.get('magnets', []):
            if rect.collidepoint(pos):
                return where
        return None

    def row_at(self, pos):
        for rect, key in self.hit.get('lines', []):
            if rect.inflate(6, 10).collidepoint(pos):
                return key
        return None

    def insert_index(self, key, x):
        cs = [r.centerx for r, (k, _) in self.hit.get('magnets', []) if k == key]
        return sum(1 for c in cs if c < x)

    def magnet_width(self, m):
        return max(18, self.small.size(m)[0] + 9)

    def draw_magnet(self, m, x, y, lifted=False, scale=1.0, good=False):
        w = self.magnet_width(m)
        r = pygame.Rect(x, y, w, MAG_H)
        d = r
        if scale != 1.0:
            d = pygame.Rect(0, 0, int(w * scale), int(MAG_H * scale))
            d.center = r.center
        if lifted:
            pygame.draw.rect(self.screen, LINE, d.move(3, 3), border_radius=3)
        symbol = m in ('a', 'b', ',', ')', '−', '+')
        bg = GOOD_BG if good else (233, 241, 255) if symbol else WHITE
        pygame.draw.rect(self.screen, bg, d, border_radius=3)
        pygame.draw.rect(self.screen, lighter(GOOD, 0.2) if good else FAINT, d, 1, border_radius=3)
        col = PART_COL.get(m)
        t = self.small.render(m, True, darker(col, 0.8) if col else FG)
        self.screen.blit(t, (d.centerx - t.get_width() // 2, d.centery - t.get_height() // 2))
        return r

    def take_magnet(self, key, j):
        m = self.rows[key].pop(j)
        self.changed(key)
        return m

    def put_magnet(self, key, j, m):
        allowed = L.MUG_MAGNETS if key == 'C4' else L.OP_MAGNETS
        if m not in allowed:
            self.say(f'{m} goes in {"C4, the mug" if key != "C4" else "C1–C3, the operations"}', AMBER)
            return False
        self.rows[key].insert(j, m)
        self.shown_row = key
        self.changed(key)
        return True

    # ------------------------------------------------------------ tab 3: sliders

    def set_slider(self, name, x, rect):
        lo, hi = L.MUG_RANGE[name]
        f = min(1.0, max(0.0, (x - rect.x) / rect.w))
        self.mug_params[name] = round(lo + f * (hi - lo), 2)
        self.success.pop('mug', None)
        self.build_mug()

    # ------------------------------------------------------------ tab 4: the cube editor

    def cube_project(self, rect):
        """2D positions of the 8 corners and 12 edge midpoints in the editor, and their depth."""
        yaw, pitch = map(math.radians, self.cube_view)
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        right = np.array([-sy, cy, 0.0])
        fwd = -np.array([cp * cy, cp * sy, sp])
        up = np.cross(right, fwd)
        P = L.CORNERS - 0.5
        s = min(rect.w, rect.h) * 0.52 * self.cube_zoom
        cx, cy_ = rect.centerx, rect.centery + 6
        persp = 1 / (1 + 0.12 * (P @ fwd))
        corners = np.stack([cx + s * (P @ right) * persp, cy_ - s * (P @ up) * persp], 1)
        depth = P @ fwd
        mids = np.array([(corners[a] + corners[b]) / 2 for a, b in L.EDGES])
        mid_depth = np.array([(depth[a] + depth[b]) / 2 for a, b in L.EDGES])
        return corners, depth, mids, mid_depth

    def edge_at(self, pos, radius=12):
        rect = self.hit.get('editor')
        if rect is None or not rect.collidepoint(pos):
            return None
        _, _, mids, _ = self.cube_project(rect)
        d = np.linalg.norm(mids - np.array(pos), axis=1)
        k = int(np.argmin(d))
        return k if d[k] < radius else None

    def triangle_at(self, pos):
        rect = self.hit.get('editor')
        if rect is None:
            return None
        _, _, mids, md = self.cube_project(rect)
        best = None
        for i, t in enumerate(self.cube_tris):
            P = mids[list(t)]
            if point_in_tri(np.array(pos, float), P):
                z = md[list(t)].mean()
                if best is None or z < best[0]:
                    best = (z, i)
        return None if best is None else best[1]

    def cube_add_edge(self, e):
        cfg = self.edit_cfg
        if e not in L.crossing_edges(cfg):
            self.cube_bad, self.cube_bad_t = (e,), time.time()
            self.cube_msg = ('both corners of this edge are on the same side: the surface does not cross it', BAD)
            return False
        if e in self.cube_path:
            return False
        self.cube_path.append(e)
        if len(self.cube_path) == 3:
            self.add_triangle(tuple(self.cube_path))
            self.cube_path, self.cube_click = [], False
        return True

    def add_triangle(self, t):
        if any(set(t) == set(u) for u in self.cube_tris):
            self.cube_msg = ('that triangle is already there', AMBER)
            return
        self.cube_tris.append(t)
        self.cube_changed()

    def cube_changed(self):
        cfg = self.edit_cfg
        if cfg is None:
            return
        ok, msg, bad = L.check_triangles(cfg, self.cube_tris)
        if ok is True:
            self.solve_case(cfg, list(self.cube_tris))
            return
        self.edit_ok = False
        if L.CLASS_OF[cfg] in self.drawn:
            msg += ' (until it is right again, the march uses your last right drawing)'
        if ok is None:
            n = len(self.cube_tris)
            self.cube_msg = (f'{n} triangle{"s" if n != 1 else ""} · {msg}', MUTED)
            self.cube_bad = ()
        else:
            self.cube_msg = (msg, BAD)
            self.cube_bad, self.cube_bad_t = bad, time.time()

    def cube_answer(self):
        cfg = self.edit_cfg
        if cfg is None:
            self.say('open a case first (a click on it in the list)', AMBER)
            return
        k = L.CLASS_OF[cfg]
        tris = L.released_triangles(self.released.get('X'), k, cfg)
        if tris is None:
            self.say(f'no released answer for case {k} yet', AMBER)
            return
        self.cube_tris, self.cube_path = tris, []
        self.cube_changed()

    def cube_mouse_down(self, ev):
        rect = self.hit['editor']
        if ev.button in (4, 5):                        # wheel: zoom the cube
            f = 1.1 if ev.button == 4 else 1 / 1.1
            self.cube_zoom = max(0.6, min(1.5, self.cube_zoom * f))
            return
        e = self.edge_at(ev.pos)
        if ev.button == 3:
            i = self.triangle_at(ev.pos)
            if self.cube_path:
                self.cube_path, self.cube_click = [], False
            elif i is not None:
                del self.cube_tris[i]
                self.cube_changed()
            return
        if ev.button != 1:
            return
        if e is not None:
            if self.cube_click and self.cube_path:
                self.cube_add_edge(e)
            else:
                self.cube_path, self.cube_click = [], False
                if self.cube_add_edge(e):
                    self.drag = dict(what=('cube', None), pos=ev.pos, start=e)
            return
        self.cube_path, self.cube_click = [], False
        if rect.collidepoint(ev.pos):
            self.drag = dict(what=('cube_orbit', None), pos=ev.pos)

    def cube_mouse_up(self, ev):
        d = self.drag
        if d['what'][0] != 'cube':
            return
        if len(self.cube_path) in (1, 2):
            self.cube_click = True                     # finish with clicks
            self.cube_msg = (f'{3 - len(self.cube_path)} more point{"s" if len(self.cube_path) == 1 else ""} '
                             f'for this triangle (click, or right-click to cancel)', MUTED)

    def cube_mouse_motion(self, ev):
        d = self.drag
        if d['what'][0] == 'cube_orbit':
            dx, dy = ev.pos[0] - d['pos'][0], ev.pos[1] - d['pos'][1]
            d['pos'] = ev.pos
            self.cube_view[0] -= dx * 0.5
            self.cube_view[1] = max(-80.0, min(80.0, self.cube_view[1] + dy * 0.5))
            return
        d['pos'] = ev.pos
        e = self.edge_at(ev.pos, radius=10)
        if e is not None and e not in self.cube_path and e in L.crossing_edges(self.edit_cfg):
            self.cube_add_edge(e)
            if not self.cube_path:
                self.drag = None

    def draw_editor(self, rect):
        scr = self.screen
        cfg = self.edit_cfg
        need = set(L.crossing_edges(cfg))
        used = {e for t in self.cube_tris for e in t}
        pygame.draw.rect(scr, (248, 250, 252), rect, border_radius=6)
        pygame.draw.rect(scr, LINE, rect, 1, border_radius=6)
        corners, depth, mids, md = self.cube_project(rect)
        # back edges first, faint; then triangles; then front edges
        order = sorted(range(12), key=lambda k: -md[k])
        for k in order:
            a, b = L.EDGES[k]
            back = md[k] > 0.1
            pygame.draw.line(scr, LINE if back else FAINT, corners[a], corners[b], 1 if back else 2)
        tri_s = pygame.Surface(rect.size, pygame.SRCALPHA)
        for t in sorted(self.cube_tris, key=lambda t: -md[list(t)].mean()):
            P = mids[list(t)] - [rect.x, rect.y]
            pygame.draw.polygon(tri_s, (*SURF, 150), P.tolist())
            pygame.draw.polygon(tri_s, (*darker(SURF), 255), P.tolist(), 2)
        scr.blit(tri_s, rect.topleft)
        bad_on = time.time() - self.cube_bad_t < 1.5
        if bad_on and len(self.cube_bad) == 2:
            pygame.draw.line(scr, BAD, mids[self.cube_bad[0]], mids[self.cube_bad[1]], 4)
        # the path being drawn
        if self.cube_path:
            pts = [mids[e].tolist() for e in self.cube_path]
            if self.drag and self.drag['what'][0] == 'cube':
                pts.append(list(self.drag['pos']))
            elif self.cube_click:
                pts.append(list(pygame.mouse.get_pos()))
            if len(pts) > 1:
                pygame.draw.lines(scr, ACCENT, False, pts, 2)
        # corners: red inside, green outside
        for i in sorted(range(8), key=lambda i: -depth[i]):
            col = INSIDE if cfg >> i & 1 else OUTSIDE
            r = 9 if depth[i] < 0 else 7
            pygame.draw.circle(scr, WHITE, corners[i], r + 2)
            pygame.draw.circle(scr, col, corners[i], r)
        # edge points: big where the surface crosses, hollow until used
        pop = self.pop_scale('cube', 0)
        for k in range(12):
            x, y = mids[k]
            if k in need:
                r = int(8 * pop)
                if k in used or k in self.cube_path:
                    pygame.draw.circle(scr, ACCENT, (x, y), r)
                else:
                    pygame.draw.circle(scr, WHITE, (x, y), r)
                    pygame.draw.circle(scr, HOLE, (x, y), r, 3)
            else:
                pygame.draw.circle(scr, FAINT, (x, y), 3)
            if bad_on and k in self.cube_bad and len(self.cube_bad) == 1:
                pygame.draw.circle(scr, BAD, (x, y), 11, 3)
            if bad_on and len(self.cube_bad) > 2 and k in self.cube_bad:
                pygame.draw.circle(scr, AMBER, (x, y), 11, 2)
        scr.blit(self.small.render('drag empty space to turn the cube', True, FAINT), (rect.x + 8, rect.bottom - 18))

    # ------------------------------------------------------------ sidebar

    def wrap(self, text, width, font=None):
        font = font or self.small
        lines, cur = [], ''
        for w in text.split(' '):
            nxt = f'{cur} {w}'.strip()
            if cur and font.size(nxt)[0] > width:
                lines.append(cur)
                cur = w
            else:
                cur = nxt
        return lines + [cur]

    def button(self, label, action, x, y, ready=True, on=False):
        t = self.bold.render(label, True, WHITE if ready else MUTED)
        r = pygame.Rect(x, y, t.get_width() + 22, 30)
        pygame.draw.rect(self.screen, (darker(ACCENT) if on else ACCENT) if ready else TRACK, r, border_radius=6)
        self.screen.blit(t, (r.x + 11, r.y + 7))
        self.hit['buttons'].append((r, action))
        return r.right + 8

    def chip_box(self, text, fg, bg, x, y, key=None, left=True):
        t = self.small.render(text, True, fg)
        w = t.get_width() + 12
        r = pygame.Rect(x if left else x - w, y, w, 20)
        pygame.draw.rect(self.screen, bg, r, border_radius=10)
        self.screen.blit(t, (r.x + 6, r.y + 3))
        if key is not None and key in self.pops and key not in self.sparked:
            self.sparked.add(key)
            self.sparks(r.centerx, r.centery)
        pop = self.pop_scale(key, 0) if key is not None else 1.0
        if pop != 1.0:
            g = int(8 * (pop - 1) / 0.32)
            pygame.draw.rect(self.screen, GOOD, r.inflate(g, g), 2, border_radius=12)
        return r

    def header(self, text, x, y, sub=None):
        self.screen.blit(self.bold.render(text, True, MUTED), (x, y))
        if sub:
            self.screen.blit(self.small.render(sub, True, FAINT), (x + self.bold.size(text)[0] + 10, y + 2))

    def draw_sidebar(self):
        scr = self.screen
        pygame.draw.rect(scr, PANE_BG, (SBX, SBY, SBW, SBH), border_radius=6)
        pygame.draw.rect(scr, LINE, (SBX, SBY, SBW, SBH), 1, border_radius=6)
        self.hit = {'tabs': [], 'buttons': [], 'steps': [], 'palette': [], 'magnets': [], 'lines': [],
                    'sliders': [], 'editor': None}
        x = SBX + 10
        for tab, label in TABS:
            done = tab in self.success
            label += ' ✓' if done else ''
            w = self.bold.size(label)[0] + 22
            r = pygame.Rect(x, SBY + 8, w, 28)
            on = self.tab == tab
            pygame.draw.rect(scr, ACCENT if on else TRACK, r, border_radius=5)
            t = self.bold.render(label, True, WHITE if on else (GOOD if done else FG))
            scr.blit(t, (r.x + 11, r.y + 6))
            self.hit['tabs'].append((r, tab))
            x += w + 6
        {'torus': self.draw_torus_tab, 'csg': self.draw_csg_tab, 'mug': self.draw_mug_tab,
         'mesh': self.draw_mesh_tab}[self.tab]()

    def code_chip(self, tag, x, y):
        s, d = self.status_of(tag)
        text, fg, bg = {'ok': ('✓ passes', GOOD, GOOD_BG), 'answer': ('released answer', ACCENT, ACCENT_BG),
                        'todo': ('TODO', AMBER, AMBER_BG), 'fail': ('✗ fails', BAD, BAD_BG)}[s]
        if s == 'ok' and tag not in self.pops and not getattr(self, '_popped_' + tag, False):
            setattr(self, '_popped_' + tag, True)
            if not self.args.answers:
                self.pop(tag)
        return self.chip_box(text, fg, bg, x, y, key=tag if s == 'ok' else None)

    def draw_torus_tab(self):
        scr = self.screen
        x0, y = SBX + 12, SBY + 48
        right = SBX + SBW - 12
        self.header('YOUR CODE', x0, y, 'sdf_student.py · save and the demo reloads it')
        y += 24
        for tag, sig, what in (('T1', 'torus_sdf(p, R, r)', 'the distance to a torus around the z axis'),
                               ('T2', 'fit(model, points, …)', 'Adam on the mean of model(points)²')):
            area = pygame.Rect(x0, y, right - x0, 62)
            pygame.draw.rect(scr, (240, 253, 244) if self.status_of(tag)[0] in ('ok', 'answer') else (248, 250, 252),
                             area, border_radius=4)
            scr.blit(self.bold.render(tag, True, FG), (x0 + 6, y + 6))
            scr.blit(self.font.render(sig, True, FG), (x0 + 40, y + 6))
            r = self.code_chip(tag, x0 + 6, y + 34)
            s, d = self.status_of(tag)
            text = what if s in ('ok', 'todo', 'answer') else d
            for k, line in enumerate(self.wrap(text, right - r.right - 14)[:2]):
                scr.blit(self.small.render(line, True, BAD if s == 'fail' else MUTED), (r.right + 8, y + 28 + 15 * k))
            y += 68
        y += 6
        self.header('DEPTH MAPS', x0, y, 'six cameras, known poses · near = dark')
        y += 22
        self.draw_depth_maps(self.torus_obs, x0, y, right)
        y += 92
        self.header('YOUR TORUS', x0, y, 'R = 3 cm, r = 1 cm, then placed:')
        y += 22
        scr.blit(self.small.render('p = S⁻¹ Rᵀ (x − t)   ·   9 numbers to fit: R (3), t (3), s (3)', True, MUTED), (x0, y))
        y += 22
        m = self.torus_model
        if m is None:
            scr.blit(self.small.render('the numbers appear once T1 works', True, FAINT), (x0, y))
        else:
            n = m.pose_numbers()
            rows = [('t', '  '.join(f'{v:+.2f}' for v in n['t']) + '  cm'),
                    ('R', f'{n["deg"]:.1f}° about ({n["axis"][0]:.2f}, {n["axis"][1]:.2f}, {n["axis"][2]:.2f})'),
                    ('s', '  '.join(f'{v:.2f}' for v in n['s']) + '   (x, y, z of the torus)')]
            for lab, val in rows:
                scr.blit(self.bold.render(lab, True, FG), (x0, y))
                scr.blit(self.font.render(val, True, FG), (x0 + 24, y))
                y += 20
        y += 6
        self.draw_fit_status('torus', x0, y)
        self.fit_buttons()

    def draw_fit_status(self, tab, x0, y):
        losses = self.losses[tab]
        if self.fitting and self.fitting['tab'] == tab:
            text = f'step {len(losses)} / {self.fitting["steps"]}   loss {losses[-1]:.4g}' if losses else 'starting…'
            self.screen.blit(self.font.render(text, True, ACCENT), (x0, y))
        elif tab in getattr(self, 'chamfer', {}):
            ch = self.chamfer[tab]
            col = GOOD if ch < L.SUCCESS_CM else AMBER
            self.screen.blit(self.font.render(f'last fit: {ch * 10:.1f} mm from the depth points '
                                              f'(goal < {L.SUCCESS_CM * 10:g} mm)', True, col), (x0, y))

    def fit_buttons(self):
        x0, by = SBX + 12, SBY + SBH - 74
        fitting = self.fitting is not None
        paused = fitting and self.fitting['paused']
        bx = self.button('Fit  F', self.start_fit, x0, by, not fitting)
        bx = self.button('Resume  ␣' if paused else 'Pause  ␣', self.toggle_pause, bx, by, fitting)
        bx = self.button('Stop  Esc', self.stop_fit, bx, by, fitting)
        self.button('Reset  R', self.reset_fit, bx, by, not fitting)
        self.screen.blit(self.small.render(f'F runs your fit(model, points, steps={FIT_STEPS}, lr={FIT_LR})', True,
                                           FAINT), (x0, by + 38))

    def draw_depth_maps(self, obs, x0, y, right):
        key = id(obs)
        if getattr(self, '_thumbs_key', None) != key:
            thumbs = []
            for dm in obs.depths:
                ok = np.isfinite(dm)
                lo, hi = (dm[ok].min(), dm[ok].max()) if ok.any() else (0, 1)
                v = np.where(ok, (dm - lo) / max(hi - lo, 1e-6), 1.0)
                rgb = np.stack([30 + 160 * v, 64 + 150 * v, 175 + 70 * v], -1)
                rgb[~ok] = 255
                s = pygame.surfarray.make_surface(rgb.astype(np.uint8).swapaxes(0, 1))
                thumbs.append(pygame.transform.smoothscale(s, (72, 72)))
            self._thumbs_key, self._thumbs = key, thumbs
        for k, t in enumerate(self._thumbs):
            x = x0 + k * 79
            self.screen.blit(t, (x, y))
            col = VIEW_COLS[k % len(VIEW_COLS)]
            pygame.draw.rect(self.screen, col, (x - 1, y - 1, 74, 74), 2)
            self.screen.blit(self.small.render(str(k + 1), True, col), (x + 3, y + 74))

    def draw_csg_tab(self):
        scr = self.screen
        x0, y = SBX + 12, SBY + 48
        right = SBX + SBW - 12
        self.header('MAGNETS', x0, y, 'drag into the rows; reuse any magnet')
        y += 22
        for group, label in ((L.OP_MAGNETS, 'C1–C3'), (L.MUG_MAGNETS, 'C4')):
            scr.blit(self.small.render(label, True, MUTED), (x0, y + 6))
            x = x0 + 46
            for m in group:
                w = self.magnet_width(m)
                if x + w > right:
                    x, y = x0 + 46, y + MAG_H + 4
                r = self.draw_magnet(m, x, y)
                self.hit['palette'].append((r, m))
                x += w + 5
            y += MAG_H + 8
        y += 4
        self.header('OPERATIONS', x0, y, 'a and b are two SDFs (the spheres in the viewer)')
        y += 22
        for key, name in L.OP_ROWS:
            y = self.draw_row_block(key, f'{name}(a, b) =', x0, y, right, 146)
        y += 6
        self.header('THE MUG', x0, y, 'body, cavity: cylinders · handle: your torus')
        y += 22
        y = self.draw_row_block('C4', 'mug =', x0, y, right, 58)
        y += 4
        for line in self.wrap('Click a row to show it in the viewer. The plot under the viewer shows the '
                              'distances along the line through the shapes.', right - x0):
            scr.blit(self.small.render(line, True, FAINT), (x0, y))
            y += 15

    def draw_row_block(self, key, label, x0, y, right, label_w):
        scr = self.screen
        area = pygame.Rect(x0, y - 4, right - x0, MAG_H + 34)
        good = self.checked[key] is True
        shown = self.shown_row == key
        pygame.draw.rect(scr, (240, 253, 244) if good else (248, 250, 252), area, border_radius=4)
        if shown:
            pygame.draw.rect(scr, ACCENT if not good else lighter(GOOD, 0.2), area, 2, border_radius=4)
        self.hit['steps'].append((area, key))
        scr.blit(self.bold.render(key, True, FG), (x0 + 6, y + 5))
        scr.blit(self.font.render(label, True, FG), (x0 + 34, y + 5))
        row = pygame.Rect(x0 + 34 + label_w, y, right - x0 - 40 - label_w, MAG_H + 2)
        toks = self.rows[key]
        dragging = self.drag is not None and self.drag['what'][0] == 'mag'
        over = dragging and row.inflate(6, 10).collidepoint(self.drag['pos'])
        pygame.draw.rect(scr, (255, 247, 230) if over else WHITE, row, border_radius=4)
        if not toks:
            dashed_rect(scr, ACCENT if dragging else FAINT, row)
        else:
            pygame.draw.rect(scr, LINE, row, 1, border_radius=4)
        self.hit['lines'].append((row, key))
        x = row.x + 3
        for j, m in enumerate(toks):
            r = self.draw_magnet(m, x, row.y + 1, scale=self.pop_scale(key, j + 1), good=good)
            self.hit['magnets'].append((r, (key, j)))
            x = r.right + 3
        if over:
            k = self.insert_index(key, self.drag['pos'][0])
            rs = [rr for rr, (kk, _) in self.hit['magnets'] if kk == key]
            cx = rs[k - 1].right + 1 if k > 0 else row.x + 2
            pygame.draw.line(scr, ACCENT, (cx, row.y + 2), (cx, row.bottom - 2), 3)
        note, col = self.row_note(key)
        if self.checked[key] is True:
            self.chip_box('✓ correct', GOOD, GOOD_BG, x0 + 6, y + MAG_H + 6, key=key)
        elif key in self.released:
            scr.blit(self.small.render('A answer', True, ACCENT), (x0 + 6, y + MAG_H + 8))
        if note:
            while self.small.size(note)[0] > right - x0 - 96 and len(note) > 4:
                note = note[:-2].rstrip() + '…'
            scr.blit(self.small.render(note, True, col), (x0 + 90, y + MAG_H + 8))
        return y + MAG_H + 38

    def draw_mug_tab(self):
        scr = self.screen
        x0, y = SBX + 12, SBY + 48
        right = SBX + SBW - 12
        self.header('YOUR MUG', x0, y, 'your C1–C4 and T1, with 7 sizes and a pose')
        y += 22
        if self.mug_model is None:
            for line in self.wrap(self.mug_err, right - x0):
                scr.blit(self.font.render(line, True, AMBER), (x0, y))
                y += 18
            y += 6
        n = self.mug_model.numbers() if self.mug_model is not None else dict(self.mug_params)
        bar_x, bar_w = x0 + 128, 230
        for name, label in L.MUG_PARAMS:
            scr.blit(self.small.render(label, True, FG), (x0, y + 3))
            lo, hi = L.MUG_RANGE[name]
            v = float(n[name])
            r = pygame.Rect(bar_x, y + 6, bar_w, 8)
            pygame.draw.rect(scr, TRACK, r, border_radius=4)
            f = (min(max(v, lo), hi) - lo) / (hi - lo)
            pygame.draw.rect(scr, lighter(ACCENT, 0.4), (r.x, r.y, int(r.w * f), r.h), border_radius=4)
            pygame.draw.circle(scr, ACCENT, (int(r.x + r.w * f), r.centery), 7)
            scr.blit(self.font.render(f'{v:5.2f} cm', True, FG), (bar_x + bar_w + 14, y))
            self.hit['sliders'].append((r.inflate(10, 14), name))
            y += 26
        y += 4
        if self.mug_model is not None:
            scr.blit(self.small.render('pose', True, FG), (x0, y + 3))
            ax, deg = n['axis'], n['deg']
            turn = deg * (1 if ax[2] >= 0 else -1)
            t = n['t']
            scr.blit(self.font.render(f'turned {turn:+.1f}° · t ({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) cm', True, FG),
                     (bar_x, y))
            y += 26
        bx = x0
        for label, flag in (('fit the pose', 'fit_pose'), ('fit the sizes', 'fit_shape')):
            on = getattr(self, flag)
            box = pygame.Rect(bx, y + 2, 18, 18)
            pygame.draw.rect(scr, ACCENT if on else WHITE, box, border_radius=3)
            pygame.draw.rect(scr, ACCENT if on else FAINT, box, 1, border_radius=3)
            if on:
                pygame.draw.lines(scr, WHITE, False, [(box.x + 4, box.y + 9), (box.x + 8, box.y + 13),
                                                       (box.x + 14, box.y + 5)], 2)
            t = self.font.render(label, True, FG)
            scr.blit(t, (box.right + 6, y + 2))
            hit = pygame.Rect(bx, y, box.w + 10 + t.get_width(), 22)
            self.hit['buttons'].append((hit, (lambda f=flag: self.toggle_flag(f))))
            bx = hit.right + 22
        self.button('Half turn  H', self.half_turn, right - 128, y - 4, self.mug_model is not None and not self.fitting)
        y += 34
        self.header('DEPTH MAPS', x0, y, 'of assets/mug.obj · near = dark')
        y += 22
        self.draw_depth_maps(self.mug_obs, x0, y, right)
        y += 96
        self.draw_fit_status('mug', x0, y)
        self.fit_buttons()

    def toggle_flag(self, flag):
        if not self.fitting:
            setattr(self, flag, not getattr(self, flag))

    def half_turn(self):
        if self.mug_model is None or self.fitting:
            return
        self.sync_mug()
        self.mug_pose[0] += 180.0
        self.build_mug()
        self.success.pop('mug', None)
        self.say('turned the mug half way round: its handle now points away from the handle\'s depth points', MUTED)

    def draw_mesh_tab(self):
        scr = self.screen
        x0, y = SBX + 12, SBY + 48
        right = SBX + SBW - 12
        self.header('GRID', x0, y, 'your mug from tab 3, sampled at the grid points')
        y += 24
        scr.blit(self.small.render('spacing', True, FG), (x0, y + 4))
        x = x0 + 64
        for h in H_STEPS:
            on = abs(self.h - h) < 1e-9
            t = self.small.render(f'{h * 10:g} mm', True, WHITE if on else FG)
            r = pygame.Rect(x, y, t.get_width() + 14, 22)
            pygame.draw.rect(scr, ACCENT if on else TRACK, r, border_radius=11)
            scr.blit(t, (r.x + 7, r.y + 4))
            self.hit['buttons'].append((r, (lambda h=h: self.set_h(h))))
            x = r.right + 6
        y += 30
        busy = self.march_state in ('running', 'paused', 'stuck', 'solved')
        bx = self.button('Sample grid  G', self.sample_grid, x0, y, self.mug_model is not None and not busy)
        bx = self.button('Extract  E', self.extract, bx, y, self.grid is not None and not busy)
        self.button('Interpolated  I' if self.interpolate else 'Midpoints  I', self.toggle_interp, bx, y, True)
        y += 40
        self.header('CASES', x0, y, '256 sign patterns, 15 cases (slide 191)')
        y += 22
        counts = self.grid.class_counts() if self.grid is not None else {}
        stuck_k = L.CLASS_OF[self.stuck_cfg()] if self.march_state == 'stuck' else None
        open_k = L.CLASS_OF[self.edit_cfg] if self.edit_cfg is not None else None
        self.hit['cases'] = []
        cw, ch = 93, 36
        for i, k in enumerate(range(15)):
            cx, cy = x0 + (i % 5) * (cw + 4), y + (i // 5) * (ch + 4)
            r = pygame.Rect(cx, cy, cw, ch)
            if k == 0:
                state, fg, bg = 'empty', MUTED, TRACK
            elif k in self.skipped:
                state, fg, bg = 'skipped', BAD, BAD_BG
            elif k == stuck_k:
                state, fg, bg = 'draw it', WHITE, HOLE
            elif k in self.drawn:
                state, fg, bg = '✓ drawn', GOOD, GOOD_BG
            elif k in L.GIVEN_CLASSES:
                state, fg, bg = 'given', MUTED, (241, 245, 249)
            else:
                state, fg, bg = 'TODO', AMBER, AMBER_BG
            pop = self.pop_scale(f'X{k}', 0)
            rr = r.inflate(int(10 * (pop - 1) / 0.32), int(10 * (pop - 1) / 0.32))
            pygame.draw.rect(scr, bg, rr, border_radius=5)
            if k == open_k or (k in L.MIRROR_OF and L.MIRROR_OF[k] == open_k):
                pygame.draw.rect(scr, ACCENT, rr.inflate(4, 4), 2, border_radius=7)
            self.hit['cases'].append((r, k))
            scr.blit(self.bold.render(str(k), True, fg if k != stuck_k else WHITE), (r.x + 6, r.y + 3))
            scr.blit(self.small.render(state, True, fg), (r.x + 6, r.y + 19))
            if f'X{k}' in self.pops and f'X{k}' not in self.sparked:
                self.sparked.add(f'X{k}')
                self.sparks(r.centerx, r.centery, 12)
            if counts.get(k) and k:
                c = self.small.render(f'{counts[k]}', True, darker(fg) if k != stuck_k else WHITE)
                scr.blit(c, (r.right - c.get_width() - 5, r.y + 4))
        y += 3 * (ch + 4) + 2
        scr.blit(self.small.render('click a ✓ case to open it · number: its cells in this grid', True, FAINT), (x0, y))
        y += 22
        if self.edit_cfg is not None:
            self.draw_stuck_panel(x0, y, right)
        else:
            self.draw_mesh_info(x0, y, right)

    def set_h(self, h):
        if self.march_state in ('running', 'paused', 'stuck', 'solved'):
            self.say('clear the mesh first (R)', AMBER)
            return
        self.h = h
        if self.grid is not None:
            self.sample_grid()

    def toggle_interp(self):
        self.interpolate = not self.interpolate
        if self.march is not None:
            self.march.interpolate = self.interpolate
        self.say('vertices on the zero crossing of the linear interpolant (slide 189)' if self.interpolate else
                 'vertices at the edge midpoints: the mesh gets blocky', MUTED)

    def draw_stuck_panel(self, x0, y, right):
        """The cube editor for the open case, with what the case is and what to do."""
        scr = self.screen
        cfg = self.edit_cfg
        k = L.CLASS_OF[cfg]
        stopped = self.march_state in ('stuck', 'solved')
        is_stuck = stopped and cfg == self.stuck_cfg()
        rect = pygame.Rect(x0, y, 236, SBY + SBH - y - 10)
        self.hit['editor'] = rect
        self.draw_editor(rect)
        tx = rect.right + 12
        tw = right - tx
        ty = y + 2
        scr.blit(self.big.render(f'case {k}' + (' ✓' if self.edit_ok else ''), True,
                                 GOOD if self.edit_ok else HOLE), (tx, ty))
        ty += 24
        for line in self.wrap(L.CLASS_INFO[k] + (' inside' if bin(cfg).count('1') <= 4 else ' outside'), tw):
            scr.blit(self.small.render(line, True, FG), (tx, ty))
            ty += 15
        ty += 4
        if is_stuck:
            n = len(self.march.holes(k))
            where = 'green: filled when you continue' if self.edit_ok else 'orange in the viewer'
            note = f'{n} cell{"s" if n != 1 else ""} of this grid need{"s" if n == 1 else ""} it ({where})'
        else:
            n = sum(1 for c in range(256) if L.CLASS_OF[c] == k)
            note = f'one of the {n} sign patterns of case {k}'
        for line in self.wrap(note, tw):
            scr.blit(self.small.render(line, True, MUTED), (tx, ty))
            ty += 15
        ty += 6
        legend = [(INSIDE, 'inside (f < 0)'), (OUTSIDE, 'outside (f > 0)'), (HOLE, 'the surface crosses')]
        for col, text in legend:
            pygame.draw.circle(scr, col, (tx + 6, ty + 7), 5, 0 if col != HOLE else 2)
            scr.blit(self.small.render(text, True, FG), (tx + 16, ty))
            ty += 16
        ty += 6
        msg, col = self.cube_msg
        for line in self.wrap(msg, tw)[:6]:
            scr.blit(self.small.render(line, True, col), (tx, ty))
            ty += 15
        by = rect.bottom - 70
        bx = self.button('Clear  Del', self.cube_clear, tx, by, bool(self.cube_tris))
        if self.march_state == 'solved':
            self.button('Continue  E', self.continue_march, bx, by, True)
        elif is_stuck:
            self.button('Skip', self.skip_case, bx, by, True)
        bx = tx
        if 'X' in self.released and not self.edit_ok:
            bx = self.button('Answer  A', self.cube_answer, bx, by + 36, True)
        if stopped and not is_stuck:
            self.button(f'Back to {L.CLASS_OF[self.stuck_cfg()]}', self.back_to_stuck, bx, by + 36, True)
        elif not stopped:
            self.button('Close', self.close_case, bx, by + 36, True)

    def back_to_stuck(self):
        self.open_case(self.stuck_cfg()) if self.march_state == 'stuck' else \
            self.open_case(self.stuck_cfg(), self.table.get(self.stuck_cfg()))

    def cube_clear(self):
        self.cube_tris, self.cube_path = [], []
        self.cube_changed()

    def draw_mesh_info(self, x0, y, right):
        scr = self.screen
        lines = []
        if self.grid is None:
            lines = ['G samples your mug\'s SDF at every grid point.',
                     'E then marches through the cells. In each cell the signs at the 8 corners say which edges '
                     'the surface crosses; the triangles for that sign pattern come from the table of cases.',
                     'Your table is almost empty: when the march meets a case it has no triangles for, it stops '
                     'and you draw them.']
        elif self.march is None:
            lines = ['Red dots: grid points inside the mug (f < 0); green: outside. Only points near the surface '
                     'are drawn.', 'E starts marching cubes.']
        elif self.march_state == 'done':
            V, F = self.mesh_arrays()
            lines = [f'{len(V):,} vertices, {len(F):,} triangles.',
                     f'Saved as {OUT_MESH.name}.' if 'mesh' in self.success else 'Not saved (it has holes).',
                     'Try I (midpoints), or another grid spacing (then G and E).']
        else:
            lines = [f'slab {self.march.k + 1} of {self.grid.n[2]}', 'Space pauses the march.']
        for text in lines:
            for line in self.wrap(text, right - x0):
                scr.blit(self.small.render(line, True, MUTED), (x0, y))
                y += 15
            y += 6

    # ------------------------------------------------------------ title, plot, mode bar

    def draw_title(self):
        scr = self.screen
        names = {'torus': 'fit a torus to depth maps', 'csg': 'build a mug with CSG',
                 'mug': 'fit your mug to depth maps', 'mesh': 'marching cubes'}
        t = self.big.render(f'SDF · {dict(TABS)[self.tab][2:]}: {names[self.tab]}', True, FG)
        scr.blit(t, (VX, 8))
        state, col = '', MUTED
        if self.fitting:
            f = self.fitting
            state = f'{"paused" if f["paused"] else "fitting"} · step {len(self.losses[f["tab"]])} / {f["steps"]}'
            col = ACCENT
        elif self.tab == 'mesh' and self.march is not None:
            state = {'running': 'marching', 'paused': 'paused', 'stuck': 'stuck: a case is missing',
                     'solved': '✓ case drawn · E continues', 'done': 'done'}.get(self.march_state, '')
            if self.march_state in ('running', 'paused', 'stuck', 'solved'):
                state += f' · slab {self.march.k + 1} / {self.grid.n[2]}'
            col = HOLE if self.march_state == 'stuck' else (GOOD if self.march_state in ('done', 'solved') else ACCENT)
        if state:
            r = self.font.render(state, True, col)
            scr.blit(r, (VX + VW - r.get_width(), 10))

    def plot_frame(self, title):
        scr = self.screen
        pygame.draw.rect(scr, PANE_BG, (GX, GY, GW, GH))
        pygame.draw.rect(scr, LINE, (GX, GY, GW, GH), 1)
        scr.blit(self.small.render(title, True, MUTED), (GX + 8, GY + 4))

    def draw_plot(self):
        if self.tab in ('torus', 'mug'):
            self.draw_loss_plot(self.losses[self.tab])
        elif self.tab == 'csg':
            self.draw_profile()
        else:
            self.draw_mesh_plot()

    def draw_loss_plot(self, E):
        scr = self.screen
        self.plot_frame('loss per step: mean of f(x)² over the depth points (cm², log scale)')
        E = np.array([e for e in E if math.isfinite(e) and e > 0])
        if len(E) == 0:
            scr.blit(self.small.render('F runs your fit(); the loss appears here', True, FAINT), (GX + 8, GY + 26))
            return
        x0, x1, y0, y1 = GX + 70, GX + GW - 14, GY + 22, GY + GH - 14
        lo, hi = np.log10(E.min()), np.log10(E.max())
        if hi - lo < 0.5:
            lo, hi = (lo + hi) / 2 - 0.25, (lo + hi) / 2 + 0.25
        nmax = max(FIT_STEPS, len(E))
        xs = x0 + (np.arange(len(E)) + 1) / nmax * (x1 - x0)
        ys = y1 - (np.log10(E) - lo) / (hi - lo) * (y1 - y0)
        for v in (lo, hi):
            yy = y1 - (v - lo) / (hi - lo) * (y1 - y0)
            pygame.draw.line(scr, TRACK, (x0, yy), (x1, yy))
            scr.blit(self.small.render(f'{10 ** v:.3g}', True, MUTED), (GX + 8, yy - 7))
        if len(E) > 1:
            pygame.draw.lines(scr, ACCENT, False, list(zip(xs, ys)), 2)
        last = self.small.render(f'step {len(E)}: {E[-1]:.3g}  (rms {math.sqrt(E[-1]) * 10:.2f} mm)', True, FG)
        scr.blit(last, (min(xs[-1] + 6, x1 - last.get_width()), ys[-1] - 16 if ys[-1] > y0 + 16 else ys[-1] + 4))

    @torch.no_grad()
    def draw_profile(self):
        scr = self.screen
        if self.shown_row == 'C4':
            P = L.MUG_START
            self.plot_frame(f'distances along the x axis at the handle\'s height (z = {P["zh"]:g} cm)')
            s = np.linspace(-6, 8.5, 300)
            X = torch.as_tensor(np.stack([s, 0 * s, 0 * s + P['zh']], 1), dtype=torch.float32)
            m, _ = self.csg_mug()
            fn = self.torus_fn() or L.reference_torus
            parts = L.mug_parts(X, {k: torch.tensor(v) for k, v in P.items()}, fn)
            curves = [(parts[k].numpy(), PART_COL[k], 1) for k in L.PART_NAMES]
            if m is not None:
                curves.append((m(X).numpy(), FG, 3))
        else:
            name = dict(L.OP_ROWS)[self.shown_row]
            self.plot_frame(f'distances along the line through both centres: a, b and {name}(a, b)')
            (ca, _), (cb, _) = L.OPS_DEMO['a'], L.OPS_DEMO['b']
            ca, cb = np.array(ca), np.array(cb)
            d = (cb - ca) / np.linalg.norm(cb - ca)
            s = np.linspace(-5.5, 5.5, 300)
            X = torch.as_tensor((ca + cb) / 2 + s[:, None] * d, dtype=torch.float32)
            ab = self.ops_spheres(X)
            curves = [(ab['a'].numpy(), PART_COL['a'], 1), (ab['b'].numpy(), PART_COL['b'], 1)]
            tree, _ = L.parse_op(self.rows[self.shown_row])
            if tree is not None:
                curves.append((L.evaluate(tree, ab).numpy(), FG, 3))
        x0, x1, y0, y1 = GX + 40, GX + GW - 14, GY + 22, GY + GH - 8
        vmax = 2.5
        to_y = lambda v: y1 - (np.clip(v, -vmax, vmax) + vmax) / (2 * vmax) * (y1 - y0)  # noqa: E731
        xs = x0 + (s - s[0]) / (s[-1] - s[0]) * (x1 - x0)
        pygame.draw.rect(scr, (254, 242, 242), (x0, float(to_y(0)), x1 - x0, y1 - float(to_y(0))))
        pygame.draw.line(scr, FAINT, (x0, float(to_y(0))), (x1, float(to_y(0))))
        scr.blit(self.small.render('0', True, MUTED), (GX + 26, float(to_y(0)) - 7))
        scr.blit(self.small.render('inside', True, lighter(BAD, 0.3)), (x1 - 52, y1 - 14))
        for v, col, w in curves:
            pygame.draw.lines(scr, col, False, list(zip(xs.tolist(), to_y(v).astype(float).tolist())), w)

    def draw_mesh_plot(self):
        scr = self.screen
        self.plot_frame('the march, slab by slab from the bottom (z)')
        if self.grid is None:
            scr.blit(self.small.render('G samples the grid', True, FAINT), (GX + 8, GY + 26))
            return
        n = self.grid.n[2]
        x0, x1, y0 = GX + 10, GX + GW - 10, GY + 28
        w = (x1 - x0) / n
        done = self.march.k if self.march is not None else 0
        for k in range(n):
            crossing = int((self.grid.cls[:, :, k] > 0).sum())
            h = 6 + min(30, crossing / 12)
            col = GOOD if k < done or self.march_state == 'done' else TRACK
            if self.march_state in ('stuck', 'solved') and k == self.march.k:
                col = HOLE if self.march_state == 'stuck' else lighter(GOOD, 0.4)
            pygame.draw.rect(scr, col, (x0 + k * w + 1, y0 + 36 - h, max(1, w - 2), h))
        mesh = self.mesh_arrays()
        tris = 0 if mesh is None else len(mesh[1])
        scr.blit(self.small.render(f'bar height = cells with a sign change · {tris:,} triangles so far', True, MUTED),
                 (x0, GY + GH - 16))

    def draw_modebar(self):
        scr = self.screen
        pygame.draw.rect(scr, PANE_BG, (MX, MY, MW, WIN_H - MY - 8))
        pygame.draw.rect(scr, LINE, (MX, MY, MW, WIN_H - MY - 8), 1)
        x = MX + 10
        if self.tab in ('torus', 'mug'):
            E = self.losses[self.tab]
            obs = self.obs()
            text = f'rms f = {math.sqrt(E[-1]) * 10:.2f} mm' if E and math.isfinite(E[-1]) else 'no fit yet'
            text += f'  ·  {len(obs.fit_idx):,} of {len(obs.points):,} depth points used for fitting'
        elif self.tab == 'csg':
            text = '   '.join(f'{k} {"✓" if self.checked[k] else "·"}' for k in self.rows)
        else:
            text = 'no grid yet' if self.grid is None else \
                f'grid {self.grid.n[0]} × {self.grid.n[1]} × {self.grid.n[2]} cells, {self.h * 10:g} mm'
        scr.blit(self.bold.render(text, True, FG), (x, MY + 8))
        if self.msg:
            m = self.font.render(self.msg, True, self.msg_col)
            if m.get_width() > MW - 20:
                m = self.small.render(self.msg, True, self.msg_col)
            scr.blit(m, (x, MY + 34))
        scr.blit(self.small.render(HINTS[self.tab], True, FAINT), (x, MY + 62))

    def draw_drag(self):
        if not self.drag or self.drag['what'][0] != 'mag':
            return
        m = self.drag['what'][1]
        x, y = self.drag['pos']
        self.draw_magnet(m, x - self.magnet_width(m) // 2, y - MAG_H // 2, lifted=True)

    # ------------------------------------------------------------ input

    def set_tab(self, tab):
        if self.fitting:
            return
        self.tab = tab
        self.drag = None
        self.moved()
        intro = {'torus': ('T1 and T2 are in sdf_student.py; F fits your torus to the depth points', MUTED),
                 'csg': ('Build union, intersect and subtract from the magnets, then the mug', MUTED),
                 'mug': ('F fits your mug to the depth maps of a real mug; place it first if you like', MUTED),
                 'mesh': ('G samples your mug on a grid, E runs marching cubes', MUTED)}[tab]
        if tab in ('mug', 'mesh') and self.mug_model is None:
            intro = (self.mug_err, AMBER)
        self.say(*intro)

    def step_at(self, pos):
        for rect, key in self.hit.get('steps', []):
            if rect.collidepoint(pos):
                return key
        return None

    def mouse_down(self, ev):
        pos = ev.pos
        for rect, tab in self.hit.get('tabs', []):
            if rect.collidepoint(pos) and ev.button == 1:
                self.set_tab(tab)
                return
        for rect, action in self.hit.get('buttons', []):
            if rect.collidepoint(pos) and ev.button == 1:
                action()
                return
        for rect, name in self.hit.get('sliders', []):
            if rect.collidepoint(pos) and ev.button == 1 and not self.fitting:
                self.sync_mug()
                self.slider_drag = (name, rect.inflate(-10, -14))
                self.set_slider(name, pos[0], self.slider_drag[1])
                return
        if self.tab == 'mesh':
            for rect, k in self.hit.get('cases', []):
                if rect.collidepoint(pos) and ev.button == 1:
                    self.open_class(k)
                    return
            if self.hit.get('editor') is not None and self.hit['editor'].collidepoint(pos):
                self.cube_mouse_down(ev)
                return
        if self.tab == 'csg':
            where = self.magnet_at(pos)
            if where is not None:
                self.shown_row = where[0]
                if ev.button == 3:
                    self.take_magnet(*where)
                elif ev.button == 1:
                    m = self.take_magnet(*where)
                    self.drag = dict(what=('mag', m), src=where, pos=pos)
                return
            for rect, m in self.hit.get('palette', []):
                if rect.collidepoint(pos) and ev.button == 1:
                    self.drag = dict(what=('mag', m), src=None, pos=pos)
                    return
            key = self.step_at(pos)
            if key is not None and ev.button == 1:
                self.shown_row = key
                self.moved()
                return
        if pygame.Rect(VX, VY, VW, VH).collidepoint(pos):
            mods = pygame.key.get_mods()
            if ev.button == 1 and mods & pygame.KMOD_SHIFT and self.tab == 'mug':
                self.sync_mug()
                self.cam_drag = ('turn', pos)
            elif ev.button == 3 and self.tab == 'mug':
                self.sync_mug()
                self.cam_drag = ('move', pos)
            elif ev.button == 1:
                self.cam_drag = ('orbit', pos)
            elif ev.button in (4, 5):
                self.zoom(ev.button == 4)

    def mouse_up(self, ev):
        self.slider_drag = None
        if self.cam_drag:
            self.cam_drag = None
            return
        if not self.drag:
            return
        if self.drag['what'][0] in ('cube', 'cube_orbit'):
            self.cube_mouse_up(ev)
            self.drag = None
            return
        d, self.drag = self.drag, None
        m = d['what'][1]
        key = self.row_at(ev.pos)
        if key is None:
            if d['src'] is not None:
                self.say('removed', MUTED)
            return
        if self.msg_col == AMBER:
            self.msg = ''
        if not self.put_magnet(key, self.insert_index(key, ev.pos[0]), m) and d['src'] is not None:
            self.rows[d['src'][0]].insert(d['src'][1], m)      # snap back
            self.changed(d['src'][0])

    def mouse_motion(self, ev):
        if self.slider_drag:
            self.set_slider(self.slider_drag[0], ev.pos[0], self.slider_drag[1])
            return
        if self.drag:
            if self.drag['what'][0] in ('cube', 'cube_orbit'):
                self.cube_mouse_motion(ev)
            else:
                self.drag['pos'] = ev.pos
            return
        if self.cam_drag:
            self.orbit(ev.pos)

    def key_down(self, ev):
        k = ev.key
        mouse = pygame.mouse.get_pos()
        if k == pygame.K_ESCAPE:
            if self.drag:
                d, self.drag = self.drag, None
                if d['what'][0] == 'mag' and d['src'] is not None:
                    self.rows[d['src'][0]].insert(d['src'][1], d['what'][1])
                    self.changed(d['src'][0])
                return
            if self.cube_path:
                self.cube_path, self.cube_click = [], False
                return
            raise SystemExit
        if pygame.K_1 <= k <= pygame.K_4:
            self.set_tab(TABS[k - pygame.K_1][0])
        elif k == pygame.K_f:
            self.start_fit()
        elif k == pygame.K_SPACE:
            self.toggle_pause()
        elif k == pygame.K_r:
            self.reset_fit()
        elif k == pygame.K_h and self.tab == 'mug':
            self.half_turn()
        elif k == pygame.K_x:
            self.toggle_cut()
        elif k == pygame.K_c:
            self.reset_camera()
        elif k == pygame.K_g and self.tab == 'mesh':
            if self.march_state in ('running', 'paused', 'stuck', 'solved'):
                self.say('clear the mesh first (R)', AMBER)
            else:
                self.sample_grid()
        elif k == pygame.K_e and self.tab == 'mesh':
            self.extract()
        elif k == pygame.K_i and self.tab == 'mesh':
            self.toggle_interp()
        elif k == pygame.K_a:
            if self.tab == 'mesh':
                self.cube_answer()
            elif self.tab == 'csg':
                key = self.step_at(mouse)
                if key is None:
                    self.say('point the mouse at a row first', AMBER)
                else:
                    self.fill_answer(key)
        elif k == pygame.K_BACKSPACE:
            if self.tab == 'mesh' and self.edit_cfg is not None:
                if self.cube_path:
                    self.cube_path.pop()
                elif self.cube_tris:
                    self.cube_tris.pop()
                    self.cube_changed()
            elif self.tab == 'csg':
                where, key = self.magnet_at(mouse), self.row_at(mouse)
                if where is not None:
                    self.take_magnet(*where)
                elif key is not None and self.rows[key]:
                    self.take_magnet(key, len(self.rows[key]) - 1)
        elif k == pygame.K_DELETE:
            if self.tab == 'mesh' and self.edit_cfg is not None:
                self.cube_clear()
            elif self.tab == 'csg':
                key = self.step_at(mouse)
                if key is not None:
                    self.rows[key] = []
                    self.changed(key)

    # ------------------------------------------------------------ main loop

    def update(self):
        now = time.time()
        self.update_particles(min(0.05, now - self.last_t))
        self.last_t = now
        if not self.fitting:
            self.update_march(now)
            self.check_reload()
        if now - self.released_t > 2:
            self.poll_answers()

    def poll_answers(self):
        self.released_t = time.time()
        new = L.released_answers()
        fresh = sorted(set(new) - set(self.released))
        self.released = new
        if fresh:
            where = {'C1': 'tab 2', 'C2': 'tab 2', 'C3': 'tab 2', 'C4': 'tab 2', 'X': 'tab 4 (the cube)'}
            names = ', '.join(('cases' if f == 'X' else f) for f in fresh)
            self.say(f'answer for {names} released: in {where[fresh[0]]}, hover it and press A', ACCENT)

    def draw(self):
        self.screen.fill(BG)
        self.draw_title()
        self.draw_viewer()
        self.draw_plot()
        self.draw_modebar()
        self.draw_sidebar()
        self.draw_particles()
        self.draw_drag()
        pygame.display.flip()

    def handle(self, ev):
        if ev.type == pygame.QUIT:
            raise SystemExit
        if ev.type == pygame.MOUSEBUTTONDOWN:
            self.mouse_down(ev)
        elif ev.type == pygame.MOUSEBUTTONUP:
            self.mouse_up(ev)
        elif ev.type == pygame.MOUSEMOTION:
            self.mouse_motion(ev)
        elif ev.type == pygame.KEYDOWN:
            self.key_down(ev)

    def loop(self):
        try:
            while True:
                for ev in pygame.event.get():
                    self.handle(ev)
                self.update()
                self.draw()
                self.clock.tick(60)
        except SystemExit:
            pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seed', type=int, default=478)
    ap.add_argument('--answers', action='store_true', help='fill every row that has a released answer')
    args = ap.parse_args()
    App(args).loop()


if __name__ == '__main__':
    main()
