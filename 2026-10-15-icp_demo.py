"""ICP demo for Tutorial 6 (2026-10-15, Point Cloud Registration).

Two scans of the Stanford bunny: Q (blue) stays put, P (magenta) has to be
moved onto it. Build the algorithm on the right by dragging blocks, Scratch
style, and watch each piece work in the 3D view on the left.

Tab 1, Align (known pairs, slides 100-104): drag variable tiles into the empty
slots. A tile only fits a slot of its shape (cloud, point, 3x3 matrix).
Tab 2, ICP (slide 109): write the loop's four lines yourself with word and
symbol magnets (drop a magnet where it should go in a line; right-click or
drag it out to remove it). Each line says under it what it will do.
A step runs as soon as all its slots are filled. C (or a click on the chip)
checks the step under the mouse; A fills it with a released answer.

Viewer
    left-drag           orbit the camera            wheel   zoom
    Shift + left-drag   turn P by hand              right-drag   move P by hand

Keys
    Tab             switch Align / ICP
    T / O / L       Align tab: translate, rotate, align
    Space           ICP tab: run / pause the loop
    Right           ICP tab: one pass of the loop
    .               ICP tab: the next block only (block mode)
    1-4             scenes: known pairs, two scans, far start, two views
    R / N           reset P to its start / new samples
    G               random start, turned by theta0 ( [ ] change theta0 )
    V               ghost of P at its true place
    C / A           check / fill answer for the step under the mouse
    Backspace       empty the slot / remove the magnet under the mouse
    Delete          empty the whole step under the mouse
    Esc             drop what you are dragging / quit

Run:  python 2026-10-15-icp_demo.py          (--answers fills every released answer)
"""
import argparse
import io
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

import icp_lib as L  # noqa: E402

PAD = 12
TITLE_H = 32
VX, VY, VW, VH = PAD, TITLE_H, 720, 500          # viewer
GX, GY, GW, GH = PAD, VY + VH + 6, VW, 84       # E per pass plot
MX, MY, MW = PAD, GY + GH + 6, VW               # mode bar
SBX = VX + VW + PAD                               # sidebar
SBW = 504
WIN_W = SBX + SBW + PAD
WIN_H = 720
SBY, SBH = 8, WIN_H - 16

BG = (248, 250, 252)          # #f8fafc
FG = (15, 23, 42)             # #0f172a
MUTED = (100, 116, 139)       # #64748b
FAINT = (148, 163, 184)       # #94a3b8
TRACK = (226, 232, 240)       # #e2e8f0
LINE = (203, 213, 225)        # #cbd5e1
ACCENT = (37, 99, 235)        # #2563eb
ACCENT_BG = (219, 234, 254)
PANE_BG = (255, 255, 255)
VIEW_BG = (241, 245, 249)     # #f1f5f9
COL_Q = (59, 130, 246)        # target, blue (slide 98)
COL_P = (217, 70, 239)        # source, magenta (slide 98)
GOOD = (22, 163, 74)
GOOD_BG = (220, 252, 231)
BAD = (220, 38, 38)
BAD_BG = (254, 226, 226)
AMBER = (217, 119, 6)
AMBER_BG = (254, 243, 199)
GLOW = (250, 204, 21)         # running block, #facc15
WHITE = (255, 255, 255)
MONO = 'menlo,consolas,dejavusansmono,couriernew,monospace'

KIND_COL = {'cloud': (15, 189, 140), 'point': (255, 140, 26), 'matrix': (153, 102, 255)}
CONTROL = (255, 171, 25)
EVENTS = (255, 191, 0)
MYBLOCK = (255, 102, 128)
VARIABLE = (255, 140, 26)

TILE_W, TILE_H = 50, 26
SUCCESS_DEG, SUCCESS_CM = 2.0, 0.2        # true error below both: aligned (celebrate)
SOLID = (139, 92, 246)        # violet: blue + magenta merged, the solid bunny
SOLID_SECS = 1.2              # clouds -> solid surface
CONFETTI = [COL_Q, COL_P, SOLID, AMBER, GOOD, GLOW]
POP_SECS = 0.32               # one tile's pop
POP_STAGGER = 0.05            # delay from one tile to the next
MAG_H = 26                    # word / symbol magnets
LINE_PITCH = 48               # one line of the loop plus its note
ROW_H = 36
ANIM_STEP = 0.35              # s per half (translate, rotate) of a move
ANIM_RUN = 0.10
BLOCK_DELAY = {'run': 0.06, 'step': 0.30}
THETA_STEPS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180]
HINT_ALIGN = 'T translate  O rotate  L align · C check  A answer · 1-4 scene  R reset  G random start  V truth'
HINT_ICP = 'Space run  → one pass  . one line · C check  A answer · 1-4 scene  R reset  G random start  V truth'


# ---------------------------------------------------------------- math labels

class TexCache:
    """Math labels rendered once with matplotlib's mathtext (pygame fonts lack μ_P, ᵀ, ‖)."""

    def __init__(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        self._figure = Figure
        self.cache = {}

    def get(self, tex, size=14, color=FG):
        key = (tex, size, color)
        if key not in self.cache:
            fig = self._figure(figsize=(1, 1), dpi=100)
            fig.text(0, 0, tex, fontsize=size, color='#%02x%02x%02x' % color, math_fontfamily='dejavusans')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, transparent=True, bbox_inches='tight', pad_inches=0.02)
            buf.seek(0)
            self.cache[key] = pygame.image.load(buf, 'tex.png').convert_alpha()
        return self.cache[key]


# ---------------------------------------------------------------- shapes

def darker(c, f=0.75):
    return tuple(int(v * f) for v in c)


def lighter(c, f=0.55):
    return tuple(int(v + (255 - v) * f) for v in c)


def tile_shape(surf, kind, rect, fill, outline=None, width=0):
    """Cloud = pill, point = hexagon (pointed ends), matrix = square corners."""
    r = pygame.Rect(rect)
    if kind == 'point':
        h = r.height // 2
        pts = [(r.left, r.centery), (r.left + h, r.top), (r.right - h, r.top), (r.right, r.centery),
               (r.right - h, r.bottom - 1), (r.left + h, r.bottom - 1)]
        if fill is not None:
            pygame.draw.polygon(surf, fill, pts)
        if outline is not None:
            pygame.draw.polygon(surf, outline, pts, width or 1)
        return
    rad = r.height // 2 if kind == 'cloud' else 3
    if fill is not None:
        pygame.draw.rect(surf, fill, r, border_radius=rad)
    if outline is not None:
        pygame.draw.rect(surf, outline, r, width or 1, border_radius=rad)


def stack_block(surf, color, rect, hat=False):
    """A command block. A hat block (the start of a script) gets a rounder top."""
    r = pygame.Rect(rect)
    top = 12 if hat else 4
    radii = dict(border_top_left_radius=top, border_top_right_radius=top,
                 border_bottom_left_radius=4, border_bottom_right_radius=4)
    pygame.draw.rect(surf, color, r, **radii)
    pygame.draw.rect(surf, darker(color), r, 1, **radii)


def cross(surf, color, c, r=8, width=3):
    x, y = c
    pygame.draw.line(surf, color, (x - r, y - r), (x + r, y + r), width)
    pygame.draw.line(surf, color, (x - r, y + r), (x + r, y - r), width)


def dashed_rect(surf, color, rect, dash=4):
    r = pygame.Rect(rect)
    for x in range(r.left, r.right, dash * 2):
        pygame.draw.line(surf, color, (x, r.top), (min(x + dash, r.right), r.top))
        pygame.draw.line(surf, color, (x, r.bottom - 1), (min(x + dash, r.right), r.bottom - 1))
    for y in range(r.top, r.bottom, dash * 2):
        pygame.draw.line(surf, color, (r.left, y), (r.left, min(y + dash, r.bottom)))
        pygame.draw.line(surf, color, (r.right - 1, y), (r.right - 1, min(y + dash, r.bottom)))


def heat(v, vmax):
    """Diverging colour for a matrix entry: blue negative, red positive."""
    a = 0 if vmax <= 0 else max(-1.0, min(1.0, v / vmax))
    if a >= 0:
        return lighter(BAD, 1 - a * 0.85)
    return lighter(COL_Q, 1 + a * 0.85)


# ---------------------------------------------------------------- the app

class App:
    def __init__(self, args):
        pygame.init()
        pygame.display.set_caption('Tutorial 6 — Iterative Closest Point')
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.font = pygame.font.SysFont(MONO, 14)
        self.bold = pygame.font.SysFont(MONO, 14, bold=True)
        self.small = pygame.font.SysFont(MONO, 12)
        self.big = pygame.font.SysFont(MONO, 16, bold=True)
        self.tex = TexCache()
        self.clock = pygame.time.Clock()

        self.fill = L.empty_fill()
        self.checked = {s.key: None for s in L.STEPS}
        self.released = L.released_answers()
        self.released_t = time.time()
        if args.answers:
            for d in self.released.values():
                self.fill.update({k: [list(x) if isinstance(x, list) else x for x in v] for k, v in d.items()})
        for step in L.STEPS:                  # already right at start-up (--answers): no fanfare
            if L.step_filled(step, self.fill) and L.check_step(step, self.fill):
                self.checked[step.key] = True
        self.fv = 0                           # fill version (cache key)
        self.solid = None                      # the success render, once P is aligned
        self.particles = []                    # confetti and sparks
        self.pops = {}                         # step -> time its blanks popped
        self.sparked = set()
        self.last_t = time.time()
        pts, nrm = L.load_bunny()
        self.solid_pts, self.solid_nrm = pts, nrm
        self.solid_mix = np.random.default_rng(0).integers(0, 2, len(pts))
        self.tab = 'align'
        self.seed = args.seed
        self.n = L.N_DEFAULT
        self.theta0 = 60
        self.random_starts = 0
        self.ghost = False
        self.msg, self.msg_col, self.msg_t = '', FG, 0.0
        self.errors = {}                      # line key / ('loop', i) -> message
        self.glow, self.glow_until = set(), 0.0
        self.drag = None
        self.cam_drag = None
        self.anim = None
        self.run = None                        # the ICP script's execution state
        self.run_done = None                   # why the last run stopped
        self.hit = {}
        self.yaw, self.pitch, self.zoom = -35.0, 18.0, 1.0
        self.load_scene(args.scene)
        self.say('Drag the magenta bunny onto the blue one (Shift+drag turns it, right-drag moves it).', MUTED)

    # ------------------------------------------------------------ scene and pose

    def load_scene(self, key, start_deg=None, rng_axis=False):
        self.scene_key = key
        self.scene = L.make_scene(key, self.seed, self.n, start_deg, rng_axis)
        self.center = (self.scene.Q.mean(0) + self.scene.P0.mean(0)) / 2
        self.reset_pose()

    def reset_pose(self):
        self.solid = None
        self.A, self.b = np.eye(3), np.zeros(3)
        self.anim = None
        self.reset_icp(self.scene.P0.copy())

    def reset_icp(self, P):
        self.st = L.IcpState(P=P)
        self.run, self.run_done = None, None
        self.errors = {k: v for k, v in self.errors.items() if not isinstance(k, tuple)}
        self.pv = getattr(self, 'pv', 0) + 1

    @property
    def P(self):
        return self.st.P

    def set_P(self, P, R, t, keep_icp=False):
        """P moved by p -> R p + t (tracked for the true error)."""
        self.solid = None
        self.A, self.b = R @ self.A, R @ self.b + t
        if keep_icp:
            self.st.P = P
            self.pv += 1
        else:
            self.reset_icp(P)

    def align_pairs(self):
        """The pairs the Align tab uses: known in scene 1, closest points otherwise."""
        if self.scene.spec.known_pairs:
            return np.arange(len(self.P)) % len(self.scene.Q)
        key = ('pairs', self.pv)
        if getattr(self, '_pairs_key', None) != key:
            self._pairs_key, self._pairs = key, L.closest_points(self.P, self.scene.Q)
        return self._pairs

    def align_env(self):
        """Run the Align script on the current pairs (cached). Returns env or None; sets errors."""
        key = (self.pv, self.fv)
        if getattr(self, '_env_key', None) == key:
            return self._env
        self._env_key = key
        self.errors = {k: v for k, v in self.errors.items() if isinstance(k, tuple)}
        try:
            self._env = L.run_align(self.fill, self.P, self.scene.Q[self.align_pairs()])
        except L.EvalError as e:
            self._env = None
            self.errors[e.where] = str(e)
        return self._env

    def shown_pairs(self):
        if self.tab == 'icp':
            return self.st.idx
        return self.align_pairs()

    # ------------------------------------------------------------ moves (animated, slide 105)

    def start_move(self, R, t, quick=False, keep_icp=False, then=None):
        P0 = self.P.copy()
        mu = P0.mean(0)
        d = (R @ mu + t) - mu
        if L.is_rotation(R):
            axis, deg = L.axis_angle(R)
            rot = lambda lam: L.rotation(axis, lam * deg) if deg > 1e-9 else np.eye(3)  # noqa: E731
        else:
            rot = lambda lam: (1 - lam) * np.eye(3) + lam * R  # noqa: E731
        self.anim = dict(P0=P0, mu=mu, d=d, rot=rot, R=R, t=t, t0=time.time(),
                         dur=ANIM_RUN if quick else ANIM_STEP, keep_icp=keep_icp, then=then)

    def anim_P(self):
        a = self.anim
        lam = (time.time() - a['t0']) / a['dur']
        if lam < 1:                                   # translate
            return a['P0'] + min(lam, 1) * a['d'], False
        lam2 = min(lam - 1, 1)                        # then rotate about the moved centroid
        M = a['rot'](lam2)
        P = (a['P0'] - a['mu']) @ M.T + a['mu'] + a['d']
        return P, lam >= 2

    def finish_anim(self):
        a = self.anim
        self.anim = None
        P = a['P0'] @ a['R'].T + a['t']
        self.set_P(P, a['R'], a['t'], keep_icp=a['keep_icp'])
        if a['then']:
            a['then']()
        if not a['keep_icp']:                         # a move from the Align tab's buttons
            self.maybe_celebrate()

    def drawn_P(self):
        if self.anim:
            P, done = self.anim_P()
            return P
        return self.P

    # ------------------------------------------------------------ Align tab actions

    def need(self, steps, what):
        missing = [k for k in steps if not L.step_filled(L.STEP[k], self.fill)]
        if missing:
            self.say(f'{what} needs {", ".join(missing)} filled in first', AMBER)
            return False
        if self.anim:
            return False
        env = self.align_env()
        if env is None:
            self.say('fix the line marked in red first', BAD)
            return False
        return env

    def do_translate(self):
        env = self.need(['I1'], 'Translate')
        if env is False:
            return
        t = env['muQ'] - env['muP']
        self.glow_steps({'I1'})
        self.start_move(np.eye(3), t)
        self.say(f'translate by μ_Q − μ_P = {np.linalg.norm(t) * 10:.1f} mm (slide 105, first half)', MUTED)

    def do_rotate(self):
        env = self.need(['I1', 'I2', 'I3'], 'Rotate')
        if env is False:
            return
        R, mu = env['R'], env['muP']
        self.glow_steps({'I1', 'I2', 'I3'})
        self.try_move(R, mu - R @ mu, 'rotate P about μ_P by R' + self.r_desc(R))

    def do_align(self):
        env = self.need(['I1', 'I2', 'I3', 'I4'], 'Align')
        if env is False:
            return
        self.glow_steps({'I1', 'I2', 'I3', 'I4'})
        self.try_move(env['R'], env['t'], 'p ← R p + t' + self.r_desc(env['R']))

    def try_move(self, R, t, text):
        try:
            L.check_move(self.P, R, t)
        except L.EvalError as e:
            self.say(str(e), BAD)
            return
        self.start_move(R, t)
        self.say(text, BAD if not L.is_rotation(R) else MUTED)

    @staticmethod
    def r_desc(R):
        if L.is_rotation(R):
            return f'  (turns {L.angle_of(R):.1f}°)'
        if R.shape == (3, 3) and abs(np.linalg.det(R) + 1) < 1e-3 and np.allclose(R.T @ R, np.eye(3), atol=1e-3):
            return '  — R is a mirror (det R = −1)!'
        return '  — R is not a rotation (RᵀR ≠ I): watch the bunny get squashed'

    # ------------------------------------------------------------ success

    def maybe_celebrate(self):
        """P is where it belongs (true error below 2 mm and 2°): merge the clouds into a solid bunny."""
        rot, rms = L.pose_error(self.scene, self.A, self.b)
        if rot is None or rot > SUCCESS_DEG or rms > SUCCESS_CM:
            return
        self.solid = dict(t0=time.time(), rot=rot, rms=rms)
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
                w, h = p['w'] / 2, p['h'] / 2 * abs(math.cos(p['a'] * 0.7))   # tumbling
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
        """Scale of the ordinal-th tile of a step that has just been filled in correctly."""
        t0 = self.pops.get(key)
        if t0 is None:
            return 1.0
        t = time.time() - t0 - POP_STAGGER * ordinal
        if 0 <= t < POP_SECS:
            return 1 + 0.32 * math.sin(math.pi * t / POP_SECS)
        return 1.0

    def step_bg(self, key, base):
        """A correct step is tinted green; it flashes a deeper green as it pops."""
        if self.checked.get(key) is not True:
            return base
        t0 = self.pops.get(key)
        f = 0.0 if t0 is None else max(0.0, 1 - (time.time() - t0) / 0.8)
        calm, flash = (240, 253, 244), (187, 247, 208)
        return tuple(int(calm[i] + (flash[i] - calm[i]) * f) for i in range(3))

    def draw_solid(self):
        """The success render: both clouds merge into one shaded bunny made of the full scan."""
        scr = self.screen
        f = min(1.0, (time.time() - self.solid['t0']) / SOLID_SECS)
        ease = 1 - (1 - f) ** 3
        Rc = self.cam_matrix()
        nc = self.solid_nrm @ Rc.T
        keep = nc[:, 2] > 0.05                        # surfels facing the camera
        s, z = self.project(self.solid_pts[keep])
        nc = nc[keep]
        light = np.array([-0.45, 0.6, 0.66])
        light /= np.linalg.norm(light)
        half = light + np.array([0, 0, 1.0])
        half /= np.linalg.norm(half)
        k = 0.28 + 0.72 * np.clip(nc @ light, 0, 1)
        spec = np.clip(nc @ half, 0, 1) ** 40 * 0.25
        final = np.clip(np.array(SOLID)[None, :] * k[:, None] + 255 * spec[:, None], 0, 255)
        start = np.where(self.solid_mix[keep][:, None] == 0, np.array(COL_Q), np.array(COL_P))
        cols = (start * (1 - ease) + final * ease).astype(int)
        r = 1.5 + (max(3.0, 0.27 * self.scale()) - 1.5) * ease
        order = np.argsort(z)
        ri = max(1, int(round(r)))
        cols, xy = cols.tolist(), s.astype(int).tolist()
        for i in order:
            pygame.draw.circle(scr, cols[i], xy[i], ri)
        # a ring that pops out of the bunny as the clouds merge
        if f < 0.6:
            (cx, cy), = self.project(self.scene.Q.mean(0))[0]
            g = f / 0.6
            pygame.draw.circle(scr, lighter(SOLID, 0.3 + 0.6 * g), (int(cx), int(cy)), int(40 + 260 * g),
                               max(1, int(6 * (1 - g))))

    def draw_banner(self):
        f = min(1.0, (time.time() - self.solid['t0']) / 0.5)
        text = f"✓ aligned: {self.solid['rms'] * 10:.1f} mm, {self.solid['rot']:.1f}° from the truth"
        t = self.big.render(text, True, WHITE)
        w, h = t.get_width() + 28, t.get_height() + 12
        x, y = VX + (VW - w) // 2, VY + 12 - int(20 * (1 - f))
        pygame.draw.rect(self.screen, GOOD, (x, y, w, h), border_radius=h // 2)
        self.screen.blit(t, (x + 14, y + 6))
        sub = self.small.render('R resets P · Shift-drag or right-drag P to try again', True, darker(GOOD))
        self.screen.blit(sub, (VX + (VW - sub.get_width()) // 2, y + h + 4))

    def glow_steps(self, steps, secs=0.9):
        self.glow = {('step', s) for s in steps}
        self.glow_until = time.time() + secs

    # ------------------------------------------------------------ ICP tab: running the loop

    def loop_ready(self):
        if not L.step_filled(L.STEP['I5'], self.fill):
            self.say('write every line of the ICP script first (start value, condition, four lines)', AMBER)
            return False
        (init, e1), (cond, e2) = L.parse_stop(self.fill)
        if init is None:
            self.say(f'the start of E cannot run yet: {e1}', AMBER)
            return False
        if cond is None:
            self.say(f'the condition cannot run yet: {e2}', AMBER)
            return False
        for i, (act, err) in enumerate(L.parse_loop(self.fill)):
            if act is None:
                self.say(f'line {i + 1} cannot run yet: {err}', AMBER)
                return False
        return True

    def new_run(self, mode):
        """when ▶ clicked: start from the top of the script (the start value of E)."""
        if self.run_done is not None or self.st.passes > 0:
            self.reset_icp(self.P)
        self.solid = None
        self.run = dict(mode=mode, pc='init', t_next=0.0)

    def toggle_run(self):
        if self.tab != 'icp':
            self.tab = 'icp'
        if not self.loop_ready():
            return
        if self.run is None:
            self.new_run('run')
            self.say('running', MUTED)
        elif self.run['mode'] == 'run':
            self.run['mode'] = 'paused'
            self.say('paused (Space continues, → finishes this pass, . runs one line)', MUTED)
        else:
            self.run['mode'] = 'run'

    def step_pass(self):
        self.tab = 'icp'
        if not self.loop_ready() or self.anim:
            return
        if self.run is None:
            self.new_run('step')
        else:
            self.run['mode'] = 'step'

    def step_block(self):
        self.tab = 'icp'
        if not self.loop_ready() or self.anim:
            return
        if self.run is None:
            self.new_run('block')
        self.run['mode'] = 'block'
        self.exec_next()

    def stop_run(self, why, col=MUTED):
        self.run, self.run_done = None, why
        self.say(why, col)

    def exec_next(self):
        """Run the next line of the ICP script: the start of E, the condition, or a line of the loop."""
        run, st = self.run, self.st
        pc = run['pc']
        self.glow = {('loop', pc)}
        self.glow_until = float('inf')
        self.errors.pop(('loop', pc), None)
        if pc == 'init':
            init = L.parse_stop(self.fill)[0][0]
            st.E, st.E_prev = L.START_VALUES[init.value], float('inf')
            run['pc'] = 'cond'
            if run['mode'] == 'block':
                self.say(f'{L.line_text(self.fill["init"])}: {init.says}', MUTED)
            run['t_next'] = time.time() + BLOCK_DELAY.get(run['mode'], 0)
            return
        if pc == 'cond':
            cond = L.parse_stop(self.fill)[1][0]
            ok, why = L.cond_holds(cond, st, L.TAU_DEFAULT)
            if ok and st.passes >= L.MAX_PASSES:
                self.stop_run(f'stopped by the safety limit after {st.passes} passes, '
                              f'with the condition still true: {why}', AMBER)
                self.glow_until = time.time() + 1.5
                return
            if not ok:
                if st.passes == 0:
                    self.stop_run(f'the loop never ran: {why}', AMBER)
                else:
                    self.stop_run(f'stopped after {st.passes} passes: {why}', GOOD)
                    self.maybe_celebrate()
                self.glow_until = time.time() + 1.5
                return
            run['pc'] = 0
            if run['mode'] == 'block':
                self.say(f'repeat: {why}', GOOD)
            run['t_next'] = time.time() + BLOCK_DELAY.get(run['mode'], 0)
            return
        act, err = L.parse_loop(self.fill)[pc]
        try:
            if act is None:
                raise L.EvalError(err or 'this line is empty', ('loop', pc))
            r = L.run_action(act, st, self.scene.Q, self.fill, where=('loop', pc))
        except L.EvalError as e:
            self.errors[('loop', pc)] = str(e)
            self.stop_run(f'stopped: {e}', BAD)
            self.glow_until = time.time() + 1.5
            return
        self.pv += 1
        run['pc'] = pc + 1
        if isinstance(r, tuple):
            self.start_move(r[1], r[2], quick=run['mode'] == 'run', keep_icp=True,
                            then=self._after_block)
            if run['mode'] == 'block':
                self.say('move P: translate, then rotate (slide 105)', MUTED)
            return
        if run['mode'] == 'block':
            self.say(r, MUTED)
        self._after_block()

    def _after_block(self):
        run, st = self.run, self.st
        if run is None:
            return
        if run['pc'] == L.LOOP_SLOTS:
            run['pc'] = 'cond'                        # back to the top of the loop
            st.passes += 1
            st.errors.append(st.E)
            if run['mode'] in ('step', 'block'):
                E = '∞' if not math.isfinite(st.E) else f'{st.E * 100:.3g} mm²'
                self.say(f'pass {st.passes} done, E = {E}; next: the condition', MUTED)
                if run['mode'] == 'step':
                    run['mode'] = 'paused'
        run['t_next'] = time.time() + BLOCK_DELAY.get(run['mode'], 0)

    def update(self):
        now = time.time()
        self.update_particles(min(0.05, now - self.last_t))
        self.last_t = now
        if self.anim:
            P, done = self.anim_P()
            if done:
                self.finish_anim()
        if self.run and not self.anim and self.run['mode'] in ('run', 'step') and now >= self.run['t_next']:
            self.exec_next()
        if self.glow and now > self.glow_until and not (self.run and self.run['mode'] != 'paused'):
            if self.glow_until != float('inf') or self.run is None:
                self.glow = set()
        if now - self.released_t > 2:
            self.poll_answers()

    def poll_answers(self):
        self.released_t = time.time()
        new = L.released_answers()
        fresh = sorted(set(new) - set(self.released))
        self.released = new
        if fresh:
            self.say(f'answer for {", ".join(fresh)} released: hover the step and press A', ACCENT)

    # ------------------------------------------------------------ steps: check, answer, clear

    def step_at(self, pos):
        for rect, key in self.hit.get('steps', []):
            if rect.collidepoint(pos):
                return key
        return None

    def check(self, key):
        step = L.STEP[key]
        if not L.step_filled(step, self.fill):
            self.say(f'{key}: fill every slot before checking', AMBER)
            return
        ok = L.check_step(step, self.fill)
        self.checked[key] = ok
        if key == 'I5' and not ok:
            body, stop = L.check_loop(self.fill)
            if body:
                self.say('I5 ✗ not yet: the four lines inside the loop are right; look again at how E starts '
                         'and when the loop repeats (slide 109)', BAD)
                return
            if stop:
                self.say('I5 ✗ not yet: how E starts and when the loop repeats are right; look again at the '
                         'four lines inside the loop', BAD)
                return
        if ok and key == 'I5' and any(a and a.variant == 'sum' for a, _ in L.parse_loop(self.fill)):
            self.say('I5 ✓ correct; the sum has the same minimum as the mean (E just scales with the number of pairs)', GOOD)
        elif ok and key == 'I2' and tuple(self.fill['W']) in L.EQUIVALENT_W:
            self.say("I2 ✓ correct, and the same W as Σ q′p′ᵀ: the centred points sum to zero", GOOD)
        elif ok:
            self.say(f'{key} ✓ correct', GOOD)
        else:
            self.say(f'{key} ✗ not yet — look at what it does in the viewer', BAD)

    def fill_answer(self, key):
        if key not in self.released:
            self.say(f'no released answer for {key} yet', AMBER)
            return
        self.fill.update({k: [list(x) if isinstance(x, list) else x for x in v]
                          for k, v in self.released[key].items()})
        self.changed(key)
        self.say(f'{key} filled with the released answer', ACCENT)

    def clear_step(self, key):
        for k in L.step_slots(L.STEP[key]):
            if k in ('init', 'cond'):
                self.fill[k] = []
            else:
                self.fill[k] = [[] for _ in self.fill[k]] if k == 'loop' else [None] * len(self.fill[k])
        self.changed(key)

    def changed(self, key):
        self.fv += 1
        was = self.checked[key]
        self.checked[key] = None
        step = L.STEP[key]
        if L.step_filled(step, self.fill) and L.check_step(step, self.fill):
            self.check(key)                           # sets ✓ and says so
            if was is not True:
                self.pop(key)
        if key == 'I5':
            self.errors = {k: v for k, v in self.errors.items() if not isinstance(k, tuple)}

    @staticmethod
    def step_of_slot(slot):
        return L.LINE[slot[1]].step

    # ------------------------------------------------------------ input

    def slot_at(self, pos):
        for rect, slot in self.hit.get('slots', []):
            if rect.inflate(8, 8).collidepoint(pos):
                return slot
        return None

    def get_slot(self, slot):
        return self.fill[slot[1]][slot[2]]

    def put_slot(self, slot, val):
        self.fill[slot[1]][slot[2]] = val
        self.changed(self.step_of_slot(slot))

    def slot_kind(self, slot):
        return L.LINE[slot[1]].kinds[slot[2]]

    # magnets in the loop's lines
    def magnet_at(self, pos):
        for rect, where in self.hit.get('magnets', []):
            if rect.collidepoint(pos):
                return where
        return None

    def line_at(self, pos):
        for rect, li in self.hit.get('lines', []):
            if rect.inflate(6, 10).collidepoint(pos):
                return li
        return None

    def insert_index(self, li, x):
        """Where a magnet dropped at x goes in line li: before the first magnet right of x."""
        cs = [r.centerx for r, (lj, _) in self.hit.get('magnets', []) if lj == li]
        return sum(1 for c in cs if c < x)

    def toks(self, li):
        """The magnets of a line of the ICP script: 'init', 'cond', or line 0-3 of the loop."""
        return self.fill[li] if li in ('init', 'cond') else self.fill['loop'][li]

    def take_magnet(self, li, j):
        m = self.toks(li).pop(j)
        self.changed('I5')
        return m

    def put_magnet(self, li, j, m):
        self.toks(li).insert(j, m)
        self.changed('I5')

    def mouse_down(self, ev):
        pos = ev.pos
        for rect, tab in self.hit.get('tabs', []):
            if rect.collidepoint(pos) and ev.button == 1:
                self.tab = tab
                return
        for rect, action in self.hit.get('buttons', []):
            if rect.collidepoint(pos) and ev.button == 1:
                action()
                return
        for rect, key in self.hit.get('chips', []):
            if rect.collidepoint(pos) and ev.button == 1:
                self.check(key)
                return
        slot = self.slot_at(pos)
        if slot is not None:
            val = self.get_slot(slot)
            if ev.button == 3 and val is not None:
                self.put_slot(slot, None)
            elif ev.button == 1 and val is not None:
                self.put_slot(slot, None)
                self.drag = dict(what=('tile', val), src=slot, pos=pos)
            return
        where = self.magnet_at(pos)
        if where is not None:
            if ev.button == 3:
                self.take_magnet(*where)
            elif ev.button == 1:
                m = self.take_magnet(*where)
                self.drag = dict(what=('mag', m), src=('mag',) + where, pos=pos)
            return
        for rect, what in self.hit.get('palette', []):
            if rect.collidepoint(pos) and ev.button == 1:
                self.drag = dict(what=what, src=None, pos=pos)
                return
        if pygame.Rect(VX, VY, VW, VH).collidepoint(pos):
            mods = pygame.key.get_mods()
            if self.anim or (self.run and self.run['mode'] != 'paused'):
                if ev.button in (1, 3) and (mods & pygame.KMOD_SHIFT or ev.button == 3):
                    self.say('wait for P to stop moving', AMBER)
                    return
            if ev.button == 1 and mods & pygame.KMOD_SHIFT:
                self.cam_drag = ('turn', pos)
            elif ev.button == 3:
                self.cam_drag = ('move', pos)
            elif ev.button == 1:
                self.cam_drag = ('orbit', pos)
            elif ev.button == 4:
                self.zoom = min(4.0, self.zoom * 1.1)
            elif ev.button == 5:
                self.zoom = max(0.3, self.zoom / 1.1)

    def mouse_up(self, ev):
        if self.cam_drag:
            if self.cam_drag[0] in ('turn', 'move'):
                self.say('moved P by hand', MUTED)
            self.cam_drag = None
            return
        if not self.drag:
            return
        d, self.drag = self.drag, None
        kind, name = d['what']
        if kind == 'mag':
            li = self.line_at(ev.pos)
            if li is None:
                if d['src'] is not None:
                    self.say('removed', MUTED)
                return
            if self.msg_col == AMBER:                 # an old warning no longer applies
                self.msg = ''
            self.put_magnet(li, self.insert_index(li, ev.pos[0]), name)
            return
        slot = self.slot_at(ev.pos)
        if slot is None:
            if d['src'] is not None:
                self.say('removed', MUTED)
            return
        want, have = self.slot_kind(slot), L.tile_kind(name)
        if want != have:
            self.say(f'{L.TILES[name][1]} is {L.KIND_NAME[have]}; this slot needs {L.KIND_NAME[want]}', AMBER)
            if d['src'] is not None:
                self.put_slot(d['src'], name)        # snap back
            return
        if self.msg_col == AMBER:
            self.msg = ''
        self.put_slot(slot, name)

    def mouse_motion(self, ev):
        if self.drag:
            self.drag['pos'] = ev.pos
            return
        if not self.cam_drag:
            return
        mode, last = self.cam_drag
        dx, dy = ev.pos[0] - last[0], ev.pos[1] - last[1]
        self.cam_drag = (mode, ev.pos)
        if mode == 'orbit':
            self.yaw += dx * 0.4
            self.pitch = max(-89.0, min(89.0, self.pitch + dy * 0.4))
            return
        Rc = self.cam_matrix()
        right, up = Rc[0], Rc[1]
        if mode == 'turn':
            Rd = L.rotation(up, dx * 0.5) @ L.rotation(right, dy * 0.5)
            mu = self.P.mean(0)
            self.set_P((self.P - mu) @ Rd.T + mu, Rd, mu - Rd @ mu)
        else:
            s = self.scale()
            t = (right * dx - up * dy) / s
            self.set_P(self.P + t, np.eye(3), t)

    def key_down(self, ev):
        k = ev.key
        mouse = pygame.mouse.get_pos()
        if k == pygame.K_ESCAPE:
            if self.drag:
                d, self.drag = self.drag, None
                if d['src'] is not None and d['src'][0] == 'mag':
                    self.put_magnet(d['src'][1], d['src'][2], d['what'][1])
                elif d['src'] is not None:
                    self.put_slot(d['src'], d['what'][1])
                return
            raise SystemExit
        if k == pygame.K_TAB:
            self.tab = 'icp' if self.tab == 'align' else 'align'
        elif k == pygame.K_t:
            self.tab = 'align'
            self.do_translate()
        elif k == pygame.K_o:
            self.tab = 'align'
            self.do_rotate()
        elif k == pygame.K_l:
            self.tab = 'align'
            self.do_align()
        elif k == pygame.K_SPACE:
            self.toggle_run()
        elif k == pygame.K_RIGHT:
            self.step_pass()
        elif k == pygame.K_PERIOD:
            self.step_block()
        elif pygame.K_1 <= k <= pygame.K_4:
            self.load_scene(k - pygame.K_0)
            s = self.scene.spec
            self.say(f'scene {s.key}, {s.name}: {s.blurb}', MUTED)
        elif k == pygame.K_r:
            self.reset_pose()
            self.say('P is back at its start pose', MUTED)
        elif k == pygame.K_n:
            self.seed = int(np.random.default_rng().integers(1000, 9999))
            self.load_scene(self.scene_key)
            self.say(f'new samples, seed {self.seed}', MUTED)
        elif k == pygame.K_g:
            self.random_starts += 1
            seed_was = self.seed
            self.seed = seed_was * 1000 + self.random_starts
            self.load_scene(self.scene_key, start_deg=self.theta0, rng_axis=True)
            self.seed = seed_was
            self.say(f'random start: P turned {self.theta0}° about a random axis', MUTED)
        elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            i = THETA_STEPS.index(self.theta0) + (1 if k == pygame.K_RIGHTBRACKET else -1)
            self.theta0 = THETA_STEPS[max(0, min(len(THETA_STEPS) - 1, i))]
            self.say(f'θ₀ = {self.theta0}° for G', MUTED)
        elif k == pygame.K_v:
            self.ghost = not self.ghost
        elif k in (pygame.K_c, pygame.K_a, pygame.K_DELETE):
            key = self.step_at(mouse)
            if key is None:
                self.say('point the mouse at a step first', AMBER)
            elif k == pygame.K_c:
                self.check(key)
            elif k == pygame.K_a:
                self.fill_answer(key)
            else:
                self.clear_step(key)
        elif k == pygame.K_BACKSPACE:
            slot, where, li = self.slot_at(mouse), self.magnet_at(mouse), self.line_at(mouse)
            if slot is not None:
                self.put_slot(slot, None)
            elif where is not None:
                self.take_magnet(*where)
            elif li is not None and self.toks(li):
                self.take_magnet(li, len(self.toks(li)) - 1)

    def say(self, msg, col=FG):
        self.msg, self.msg_col, self.msg_t = msg, col, time.time()

    # ------------------------------------------------------------ 3D view

    def cam_matrix(self):
        """Rows: the camera's right, up and towards-viewer axes in world coordinates."""
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
        return Rx @ Ry

    def scale(self):
        return 19.0 * self.zoom

    def project(self, X):
        Y = (np.atleast_2d(X) - self.center) @ self.cam_matrix().T
        f = 90.0
        persp = f / np.maximum(f - Y[:, 2], 1.0)
        s = self.scale()
        sx = VX + VW / 2 + Y[:, 0] * s * persp
        sy = VY + VH / 2 - Y[:, 1] * s * persp
        return np.stack([sx, sy], 1), Y[:, 2]

    def draw_viewer(self):
        scr = self.screen
        pygame.draw.rect(scr, VIEW_BG, (VX, VY, VW, VH))
        clip = scr.get_clip()
        scr.set_clip(pygame.Rect(VX, VY, VW, VH))
        if self.solid:
            self.draw_solid()
            self.draw_gizmo()
            scr.set_clip(clip)
            self.draw_banner()
            pygame.draw.rect(scr, LINE, (VX, VY, VW, VH), 1)
            return
        Q, P = self.scene.Q, self.drawn_P()
        sq, zq = self.project(Q)
        sp, zp = self.project(P)
        # pair lines (under the points)
        idx = self.shown_pairs()
        if idx is not None and not self.anim and len(idx) == len(P):
            col = (180, 190, 205)
            for a, b in zip(sp, sq[idx]):
                pygame.draw.aaline(scr, col, a, b)
        if self.ghost:
            sg, _ = self.project(self.scene.P_true)
            gc = lighter(COL_P, 0.7)
            for x, y in sg[::2]:
                pygame.draw.circle(scr, gc, (int(x), int(y)), 1)
        pts = np.concatenate([sq, sp])
        z = np.concatenate([zq, zp])
        who = np.concatenate([np.zeros(len(sq), int), np.ones(len(sp), int)])
        lo, hi = z.min(), z.max()
        near = (z - lo) / max(hi - lo, 1e-9)
        order = np.argsort(z)
        cols = [COL_Q, COL_P]
        for i in order:
            c = cols[who[i]]
            fade = 0.6 * (1 - near[i])
            c = (int(c[0] + (VIEW_BG[0] - c[0]) * fade), int(c[1] + (VIEW_BG[1] - c[1]) * fade),
                 int(c[2] + (VIEW_BG[2] - c[2]) * fade))
            pygame.draw.circle(scr, c, (int(pts[i, 0]), int(pts[i, 1])), 3 if near[i] > 0.55 else 2)
        # centroids (layer of I1)
        env = self.align_env() if self.tab == 'align' else self.st.env
        if env is not None and not self.anim:
            for name, col in (('muQ', COL_Q), ('muP', COL_P)):
                if name in env and np.shape(env[name]) == (3,):
                    (x, y), = self.project(env[name])[0]
                    cross(scr, FG, (x, y), 10, 5)
                    cross(scr, col, (x, y), 9, 3)
                    lab = self.tex.get(L.TILES[name][2], 13, darker(col, 0.8))
                    scr.blit(lab, (x + 12, y - 22))
        self.draw_gizmo()
        scr.set_clip(clip)
        # legend
        y = VY + 8
        for col, text in ((COL_Q, 'Q target (fixed)'), (COL_P, 'P source (moves)')):
            pygame.draw.circle(scr, col, (VX + 14, y + 8), 5)
            scr.blit(self.small.render(text, True, FG), (VX + 24, y + 1))
            y += 18
        pairs = self.pairs_label()
        if pairs:
            scr.blit(self.small.render(pairs, True, MUTED), (VX + 10, y + 2))
        if self.ghost:
            scr.blit(self.small.render('V: faint = where P belongs', True, darker(COL_P)), (VX + 10, y + 20))
        hint = 'drag: orbit · Shift+drag: turn P · right-drag: move P · wheel: zoom'
        scr.blit(self.small.render(hint, True, FAINT), (VX + VW - 8 - self.small.size(hint)[0], VY + VH - 18))
        pygame.draw.rect(scr, LINE, (VX, VY, VW, VH), 1)

    def pairs_label(self):
        if self.tab == 'icp':
            if self.st.idx is None:
                return 'pairs: none yet (lines appear after a pair block runs)'
            return 'pairs: from your pair block'
        if self.scene.spec.known_pairs:
            return 'pairs: known (row k of P with row k of Q)'
        return 'pairs: guessed, each p with its closest q'

    def draw_gizmo(self):
        scr = self.screen
        o = np.array([VX + 40, VY + VH - 40])
        Rc = self.cam_matrix()
        for k, (col, name) in enumerate(((BAD, 'x'), (GOOD, 'y'), (COL_Q, 'z'))):
            v = Rc[:2, k] * np.array([22, -22])
            pygame.draw.line(scr, col, o, o + v, 2)
            scr.blit(self.small.render(name, True, col), o + v * 1.25 - 4)
        s = self.scale()
        x0, y0 = VX + VW - 30 - s, VY + VH - 30
        pygame.draw.line(scr, FG, (x0, y0), (x0 + s, y0), 2)
        lab = self.small.render('1 cm', True, FG)
        scr.blit(lab, (x0 + s / 2 - lab.get_width() / 2, y0 - 16))

    # ------------------------------------------------------------ title, plot, mode bar

    def draw_title(self):
        scr = self.screen
        s = self.scene.spec
        t = self.big.render(f'ICP · scene {s.key}: {s.name}', True, FG)
        scr.blit(t, (VX, 8))
        sub = self.small.render(f'seed {self.seed} · {len(self.scene.P0)} + {len(self.scene.Q)} points', True, MUTED)
        scr.blit(sub, (VX + t.get_width() + 14, 12))
        if self.run:
            st = {'run': 'running', 'paused': 'paused', 'step': 'one pass', 'block': 'line by line'}
            pc = self.run['pc']
            where = {'init': 'start of E', 'cond': 'condition'}.get(pc, f'line {pc + 1}' if isinstance(pc, int) else '')
            state = f'{st[self.run["mode"]]} · pass {self.st.passes + 1}, {where}'
            col = ACCENT
        elif self.run_done:
            state, col = f'done · {self.st.passes} passes', GOOD
        else:
            state, col = '', MUTED
        if state:
            r = self.font.render(state, True, col)
            scr.blit(r, (VX + VW - r.get_width(), 10))

    def draw_plot(self):
        scr = self.screen
        pygame.draw.rect(scr, PANE_BG, (GX, GY, GW, GH))
        pygame.draw.rect(scr, LINE, (GX, GY, GW, GH), 1)
        scr.blit(self.small.render('E per pass (mm², log scale)', True, MUTED), (GX + 8, GY + 4))
        E = np.array([e for e in self.st.errors if math.isfinite(e)]) * 100
        if len(E) == 0:
            scr.blit(self.small.render('run the ICP loop to see E fall', True, FAINT), (GX + 8, GY + 26))
            return
        x0, x1, y0, y1 = GX + 70, GX + GW - 14, GY + 22, GY + GH - 14
        lo, hi = np.log10(max(E.min(), 1e-6)), np.log10(max(E.max(), 1e-6))
        if hi - lo < 0.5:
            lo, hi = (lo + hi) / 2 - 0.25, (lo + hi) / 2 + 0.25
        nmax = max(10, len(E))
        xs = x0 + (np.arange(len(E)) + 1) / nmax * (x1 - x0)
        ys = y1 - (np.log10(np.maximum(E, 1e-6)) - lo) / (hi - lo) * (y1 - y0)
        for v in (lo, hi):
            yy = y1 - (v - lo) / (hi - lo) * (y1 - y0)
            pygame.draw.line(scr, TRACK, (x0, yy), (x1, yy))
            scr.blit(self.small.render(f'{10 ** v:.3g}', True, MUTED), (GX + 8, yy - 7))
        if len(E) > 1:
            pygame.draw.lines(scr, ACCENT, False, list(zip(xs, ys)), 2)
        for x, y in zip(xs, ys):
            pygame.draw.circle(scr, ACCENT, (int(x), int(y)), 3)
        last = self.small.render(f'pass {len(E)}: {E[-1]:.2f}', True, FG)
        scr.blit(last, (min(xs[-1] + 6, x1 - last.get_width()), ys[-1] - 16 if ys[-1] > y0 + 16 else ys[-1] + 4))

    def draw_modebar(self):
        scr = self.screen
        pygame.draw.rect(scr, PANE_BG, (MX, MY, MW, WIN_H - MY - 8))
        pygame.draw.rect(scr, LINE, (MX, MY, MW, WIN_H - MY - 8), 1)
        # E as ICP sees it, and the truth
        if self.tab == 'icp':
            E = self.st.E
            Etxt = 'E = ∞ (not measured yet)' if not math.isfinite(E) else f'E = {E * 100:.2f} mm²'
        else:
            idx = self.align_pairs()
            Etxt = f'E = {L.mean_sq(self.drawn_P(), self.scene.Q[idx]) * 100:.2f} mm²  with these pairs'
        rot, rms = L.pose_error(self.scene, self.A, self.b)
        rtxt = '—' if rot is None else f'{rot:.1f}°'
        x = MX + 10
        scr.blit(self.bold.render(Etxt, True, FG), (x, MY + 8))
        truth = f'true error: {rms * 10:.1f} mm, rotation {rtxt}'
        col = GOOD if rms < 0.2 else (AMBER if rms < 1.0 else BAD)
        r = self.font.render(truth, True, col)
        scr.blit(r, (MX + MW - r.get_width() - 10, MY + 8))
        if self.msg:
            m = self.font.render(self.msg, True, self.msg_col)
            if m.get_width() > MW - 20:
                m = self.small.render(self.msg, True, self.msg_col)
            scr.blit(m, (x, MY + 34))
        hint = HINT_ICP if self.tab == 'icp' else HINT_ALIGN
        scr.blit(self.small.render(hint, True, FAINT), (x, MY + 62))

    # ------------------------------------------------------------ sidebar

    def draw_sidebar(self):
        scr = self.screen
        pygame.draw.rect(scr, PANE_BG, (SBX, SBY, SBW, SBH), border_radius=6)
        pygame.draw.rect(scr, LINE, (SBX, SBY, SBW, SBH), 1, border_radius=6)
        self.hit = {'tabs': [], 'buttons': [], 'chips': [], 'steps': [], 'slots': [], 'palette': [],
                    'magnets': [], 'lines': []}
        x = SBX + 10
        for tab, label in (('align', '1  Align (known pairs)'), ('icp', '2  ICP loop')):
            w = self.bold.size(label)[0] + 24
            r = pygame.Rect(x, SBY + 8, w, 28)
            on = self.tab == tab
            pygame.draw.rect(scr, ACCENT if on else TRACK, r, border_radius=5)
            t = self.bold.render(label, True, WHITE if on else FG)
            scr.blit(t, (r.x + 12, r.y + 6))
            self.hit['tabs'].append((r, tab))
            x += w + 8
        tip = self.small.render('Tab switches', True, FAINT)
        scr.blit(tip, (SBX + SBW - tip.get_width() - 12, SBY + 16))
        if self.tab == 'align':
            self.draw_align_tab()
        else:
            self.draw_icp_tab()

    def chip(self, key, x, y, left=False):
        step = L.STEP[key]
        filled, empty = L.step_filled(step, self.fill), L.step_empty(step, self.fill)
        ok = self.checked[key]
        err = any((k in self.errors) for k in step.lines) or (key == 'I5' and (any(
            isinstance(k, tuple) for k in self.errors) or any(e for _, e in L.parse_loop(self.fill) + L.parse_stop(self.fill))))
        if ok is True:
            text, fg, bg = '✓ correct', GOOD, GOOD_BG
        elif ok is False:
            text, fg, bg = '✗ not yet', BAD, BAD_BG
        elif err:
            text, fg, bg = 'error', BAD, BAD_BG
        elif filled:
            text, fg, bg = 'runs', ACCENT, ACCENT_BG
        elif empty:
            text, fg, bg = 'TODO', AMBER, AMBER_BG
        else:
            text, fg, bg = 'filling', MUTED, TRACK
        t = self.small.render(text, True, fg)
        w = t.get_width() + 12
        r = pygame.Rect(x if left else x - w, y, w, 20)
        pygame.draw.rect(self.screen, bg, r, border_radius=10)
        self.screen.blit(t, (r.x + 6, r.y + 3))
        self.hit['chips'].append((r, key))
        if key in self.pops and key not in self.sparked and ok is True:
            self.sparked.add(key)
            self.sparks(r.centerx, r.centery)
        pop = self.pop_scale(key, 0) if ok is True else 1.0
        if pop != 1.0:
            pygame.draw.rect(self.screen, GOOD, r.inflate(int(8 * (pop - 1) / 0.32), int(8 * (pop - 1) / 0.32)),
                             2, border_radius=12)
        if key in self.released and ok is not True and not left:
            a = self.small.render('A answer', True, ACCENT)
            self.screen.blit(a, (r.x - a.get_width() - 8, y + 3))

    def draw_tile(self, name, rect, alpha=255, scale=1.0):
        kind, _, tex = L.TILES[name]
        col = KIND_COL[kind]
        if alpha < 255:
            col = lighter(col, 0.5)
        r = pygame.Rect(rect)
        lab = self.tex.get(tex, 13, WHITE)
        if scale != 1.0:
            c = r.center
            r = pygame.Rect(0, 0, int(r.w * scale), int(r.h * scale))
            r.center = c
            lab = pygame.transform.smoothscale(lab, (int(lab.get_width() * scale), int(lab.get_height() * scale)))
        tile_shape(self.screen, kind, r, col, darker(col))
        self.screen.blit(lab, (r.centerx - lab.get_width() // 2, r.centery - lab.get_height() // 2))
        if name in ('P', 'pc', 'Q', 'qc'):
            dot = COL_P if name in ('P', 'pc') else COL_Q
            pygame.draw.circle(self.screen, WHITE, (r.right - 9, r.centery), 5)
            pygame.draw.circle(self.screen, dot, (r.right - 9, r.centery), 4)

    def draw_slot(self, slot, kind, rect):
        val = self.get_slot(slot)
        dragging = self.drag is not None
        over = dragging and pygame.Rect(rect).inflate(8, 8).collidepoint(self.drag['pos'])
        if val is not None:
            step = L.LINE[slot[1]].step
            ordinal = L.STEP[step].lines.index(slot[1]) * 3 + slot[2]
            self.draw_tile(val, rect, scale=self.pop_scale(step, ordinal))
        else:
            fits = dragging and self.drag['what'][0] == 'tile' and L.tile_kind(self.drag['what'][1]) == kind
            fill = lighter(KIND_COL[kind], 0.85) if fits else (241, 245, 249)
            tile_shape(self.screen, kind, rect, fill, KIND_COL[kind] if fits else FAINT, 2 if over and fits else 1)
        self.hit['slots'].append((pygame.Rect(rect), slot))

    def draw_align_tab(self):
        scr = self.screen
        x0, y = SBX + 12, SBY + 46
        scr.blit(self.bold.render('VARIABLES', True, MUTED), (x0, y))
        scr.blit(self.small.render('drag a tile into a slot of the same shape', True, FAINT), (x0 + 96, y + 2))
        y += 22
        rows = (('cloud', 'clouds', ['P', 'Q', 'pc', 'qc']), ('point', 'points', ['muP', 'muQ', 't']),
                ('matrix', '3×3', ['W', 'U', 'D', 'Vt', 'R']))
        for kind, lab, names in rows:
            scr.blit(self.small.render(lab, True, MUTED), (x0, y + 6))
            for i, name in enumerate(names):
                r = pygame.Rect(x0 + 62 + i * (TILE_W + 10), y, TILE_W, TILE_H)
                self.draw_tile(name, r)
                self.hit['palette'].append((r, ('tile', name)))
            y += TILE_H + 6
        y += 8
        # the script
        hat = pygame.Rect(x0, y + 8, SBW - 24, 30)
        stack_block(scr, MYBLOCK, hat, hat=True)
        scr.blit(self.small.render('define', True, WHITE), (hat.x + 8, hat.y + 8))
        lab = self.tex.get(r'$R, t \leftarrow$ align $P$ to $Q$', 14, WHITE)
        scr.blit(lab, (hat.x + 62, hat.y + 15 - lab.get_height() // 2))
        note = self.small.render('row k of P goes with row k of Q', True, WHITE)
        scr.blit(note, (hat.right - note.get_width() - 8, hat.y + 8))
        y = hat.bottom + 8
        lines_by_step = {}
        for ln in L.ALIGN_LINES:
            lines_by_step.setdefault(ln.step, []).append(ln)
        row_y = {}
        for step in L.STEPS[:4]:
            lns = lines_by_step[step.key]
            rows_here = sorted({ln.row for ln in lns})
            top = y
            for row in rows_here:
                row_y[row] = y + (6 if len(rows_here) == 1 else 0)
                y += ROW_H
            y = max(y, top + 56)
            area = pygame.Rect(x0, top - 2, SBW - 24, y - top)
            glow = ('step', step.key) in self.glow
            bg = (255, 251, 235) if glow else self.step_bg(step.key, (248, 250, 252))
            pygame.draw.rect(scr, bg, area, border_radius=4)
            if glow:
                pygame.draw.rect(scr, GLOW, area, 3, border_radius=4)
            elif self.checked[step.key] is True:
                pygame.draw.rect(scr, lighter(GOOD, 0.45), area, 1, border_radius=4)
            self.hit['steps'].append((area, step.key))
            k = self.bold.render(step.key, True, FG)
            scr.blit(k, (x0 + 6, top + 2))
            if step.key in self.released and self.checked[step.key] is not True:
                scr.blit(self.small.render('A answer', True, ACCENT), (x0 + 12 + k.get_width(), top + 4))
            scr.blit(self.small.render(step.title, True, MUTED), (x0 + 6, top + 18))
            self.chip(step.key, x0 + 6, top + 33, left=True)
            y += 4
        for ln in L.ALIGN_LINES:
            lx = x0 + 88 + (ln.col * 196)
            self.draw_line(ln, lx, row_y[ln.row] + 4)
        # errors from the evaluator
        y += 2
        for key, text in self.errors.items():
            if not isinstance(key, tuple):
                scr.blit(self.small.render(f'{L.LINE[key].step}: {text}', True, BAD), (x0, y))
                y += 16
        self.draw_values(self.align_env(), x0, max(y + 4, SBY + 474))
        # buttons
        by = SBY + SBH - 74
        bx = x0
        for label, action, need in (('Translate  T', self.do_translate, ['I1']),
                                    ('Rotate  O', self.do_rotate, ['I1', 'I2', 'I3']),
                                    ('Align  L', self.do_align, ['I1', 'I2', 'I3', 'I4'])):
            ready = all(L.step_filled(L.STEP[k], self.fill) for k in need)
            bx = self.button(label, action, bx, by, ready)
        scr.blit(self.small.render('Translate needs I1 · Rotate I1–I3 · Align I1–I4', True, FAINT), (x0, by + 38))

    def draw_line(self, ln, x, y):
        """One line of the Align script: maths pieces with slots in between."""
        scr = self.screen
        err = ln.key in self.errors
        slot_i = 0
        cx = x
        for part in ln.parts:
            if part is None:
                kind = ln.kinds[slot_i]
                r = pygame.Rect(cx, y, TILE_W, TILE_H)
                self.draw_slot(('line', ln.key, slot_i), kind, r)
                slot_i += 1
                cx += TILE_W + 4
            else:
                s = self.tex.get(part, 13, BAD if err else FG)
                scr.blit(s, (cx, y + TILE_H // 2 - s.get_height() // 2))
                cx += s.get_width() + 5
        if err:
            pygame.draw.line(scr, BAD, (x, y + TILE_H + 3), (cx - 4, y + TILE_H + 3), 2)

    def draw_values(self, env, x0, y):
        scr = self.screen
        pygame.draw.line(scr, TRACK, (x0, y - 4), (SBX + SBW - 12, y - 4))
        if env is None or 'muP' not in env:
            scr.blit(self.small.render('values appear here as the script runs', True, FAINT), (x0, y + 4))
            return
        x = x0
        if 'W' in env:
            W = env['W']
            scr.blit(self.tex.get(r'$W$', 13), (x, y))
            vmax = np.abs(W).max()
            for i in range(3):
                for j in range(3):
                    pygame.draw.rect(scr, heat(W[i, j], vmax), (x + 22 + j * 18, y + i * 18, 17, 17))
            if 'sigma' in env:
                s = env['sigma']
                scr.blit(self.small.render('σ ' + ' '.join(f'{v:.2g}' for v in s), True, MUTED), (x, y + 58))
            x += 190
        if 'R' in env:
            R = env['R']
            scr.blit(self.tex.get(r'$R$', 13), (x, y))
            for i in range(3):
                row = ' '.join(f'{v:+.2f}' for v in R[i])
                scr.blit(self.small.render(row, True, FG), (x + 18, y + i * 16))
            ok = L.is_rotation(R)
            if ok:
                note, col = f'a rotation by {L.angle_of(R):.1f}°', GOOD
            elif abs(np.linalg.det(R) + 1) < 1e-3:
                note, col = 'a mirror: det R = −1', BAD
            else:
                note, col = 'not a rotation (RᵀR ≠ I)', BAD
            scr.blit(self.small.render(note, True, col), (x, y + 58))
            x += 175
        if 't' in env:
            t = env['t']
            scr.blit(self.tex.get(r'$t$', 13), (x, y))
            for i in range(3):
                scr.blit(self.small.render(f'{t[i] * 10:+.1f} mm', True, FG), (x + 14, y + i * 16))
        elif 'muP' in env and 'muQ' in env and 'W' not in env:
            d = np.linalg.norm(env['muQ'] - env['muP']) * 10
            scr.blit(self.small.render(f'|μ_Q − μ_P| = {d:.1f} mm', True, FG), (x, y + 4))

    def button(self, label, action, x, y, ready=True):
        t = self.bold.render(label, True, WHITE if ready else MUTED)
        r = pygame.Rect(x, y, t.get_width() + 22, 30)
        pygame.draw.rect(self.screen, ACCENT if ready else TRACK, r, border_radius=6)
        self.screen.blit(t, (r.x + 11, r.y + 7))
        self.hit['buttons'].append((r, action))
        return r.right + 8

    def magnet_surface(self, m):
        return self.tex.get(L.magnet_tex(m), 13, FG)

    def magnet_width(self, m):
        return max(20, self.magnet_surface(m).get_width() + 10)

    def draw_magnet(self, m, x, y, lifted=False, scale=1.0, good=False):
        """A fridge magnet: words on white, symbols on pale blue (green once the step is right)."""
        w = self.magnet_width(m)
        r = pygame.Rect(x, y, w, MAG_H)
        d = r
        s = self.magnet_surface(m)
        if scale != 1.0:
            d = pygame.Rect(0, 0, int(w * scale), int(MAG_H * scale))
            d.center = r.center
            s = pygame.transform.smoothscale(s, (int(s.get_width() * scale), int(s.get_height() * scale)))
        if lifted:
            pygame.draw.rect(self.screen, (203, 213, 225), d.move(3, 3), border_radius=3)
        bg = (220, 252, 231) if good else (233, 241, 255) if m in L.SYMBOLS else WHITE
        pygame.draw.rect(self.screen, bg, d, border_radius=3)
        pygame.draw.rect(self.screen, lighter(GOOD, 0.2) if good else (148, 163, 184), d, 1, border_radius=3)
        self.screen.blit(s, (d.centerx - s.get_width() // 2, d.centery - s.get_height() // 2))
        return r

    def wrap(self, text, width):
        lines, cur = [], ''
        for w in text.split(' '):
            nxt = f'{cur} {w}'.strip()
            if cur and self.small.size(nxt)[0] > width:
                lines.append(cur)
                cur = w
            else:
                cur = nxt
        return lines + [cur]

    def draw_row(self, li, row, parsed, note_xy, right, placeholder):
        """A line of the ICP script that magnets go into, and the note on what it does."""
        scr = self.screen
        toks = self.toks(li)
        drag_mag = self.drag is not None and self.drag['what'][0] == 'mag'
        over = drag_mag and row.inflate(6, 10).collidepoint(self.drag['pos'])
        pygame.draw.rect(scr, (255, 247, 230) if over else self.step_bg('I5', (248, 250, 252)), row, border_radius=4)
        if ('loop', li) in self.glow:
            pygame.draw.rect(scr, FG, row.inflate(8, 8), 1, border_radius=6)
            pygame.draw.rect(scr, GLOW, row.inflate(6, 6), 3, border_radius=5)
        elif not toks:
            dashed_rect(scr, CONTROL if drag_mag else FAINT, row)
        else:
            pygame.draw.rect(scr, LINE, row, 1, border_radius=4)
        self.hit['lines'].append((row, li))
        if not toks:
            t = self.small.render(placeholder, True, FAINT)
            scr.blit(t, (row.x + 8, row.centery - t.get_height() // 2))
        x = row.x + 3
        base = {'init': 0, 'cond': 1}.get(li, 2 + li if isinstance(li, int) else 0) * 8
        good = self.checked['I5'] is True
        for j, m in enumerate(toks):
            r = self.draw_magnet(m, x, row.centery - MAG_H // 2, scale=self.pop_scale('I5', base + j), good=good)
            self.hit['magnets'].append((r, (li, j)))
            x = r.right + 4
        if over:
            k = self.insert_index(li, self.drag['pos'][0])
            rs = [rr for rr, (lj, _) in self.hit['magnets'] if lj == li]
            cx = rs[k - 1].right + 2 if k > 0 else row.x + 2
            pygame.draw.line(scr, ACCENT, (cx, row.y + 2), (cx, row.bottom - 2), 3)
        act, err = parsed
        if ('loop', li) in self.errors:
            note, col = f'✗ {self.errors[("loop", li)]}', BAD
        elif act is not None:
            note, col = f'→ {act.says}', GOOD
        elif err:
            note, col = f'? {err}', AMBER
        else:
            return
        while self.small.size(note)[0] > right - note_xy[0] and len(note) > 4:
            note = note[:-2].rstrip() + '…'
        scr.blit(self.small.render(note, True, col), note_xy)

    def draw_icp_tab(self):
        scr = self.screen
        x0, y = SBX + 12, SBY + 46
        right = SBX + SBW - 12
        scr.blit(self.bold.render('MAGNETS', True, MUTED), (x0, y))
        scr.blit(self.small.render('build every line of the script below; reuse any magnet', True, FAINT),
                 (x0 + 70, y + 2))
        y += 22
        for group in (L.WORDS, L.SYMBOL_ORDER):
            x = x0
            for m in group:
                w = self.magnet_width(m)
                if x + w > right:
                    x, y = x0, y + MAG_H + 4
                r = self.draw_magnet(m, x, y)
                self.hit['palette'].append((r, ('mag', m)))
                x += w + 5
            y += MAG_H + 8
        align_ok = L.align_ready(self.fill)
        note = 'align means your Align script (tab 1)' + ('' if align_ok else ' — not complete yet')
        scr.blit(self.small.render(note, True, MUTED if align_ok else AMBER), (x0, y))
        y += 18
        # the script
        top_y = y
        hat = pygame.Rect(x0, y + 4, 230, 28)
        stack_block(scr, EVENTS, hat, hat=True)
        scr.blit(self.tex.get(r'when $\blacktriangleright$ Run clicked', 13, WHITE), (hat.x + 10, hat.y + 6))
        self.chip('I5', right - 2, hat.y + 4)
        (init, e_init), (cond, e_cond) = L.parse_stop(self.fill)
        y = hat.bottom + 2
        blk = pygame.Rect(x0, y, 160, MAG_H + 10)
        stack_block(scr, VARIABLE, blk)
        row = pygame.Rect(blk.x + 6, blk.y + 4, blk.w - 12, MAG_H + 2)
        self.draw_row('init', row, (init, e_init), (blk.right + 8, blk.y + 10), right, 'start of E')
        y = blk.bottom + 2
        label = self.tex.get('repeat while', 13, WHITE)
        cond_w = 286
        top = pygame.Rect(x0, y, label.get_width() + cond_w + 28, MAG_H + 10)
        inner_h = 18 + L.LOOP_SLOTS * LINE_PITCH
        area = pygame.Rect(x0, top_y, SBW - 24, top.bottom + inner_h + 16 - top_y)
        self.hit['steps'].append((area, 'I5'))
        pygame.draw.rect(scr, CONTROL, top, border_top_left_radius=4, border_top_right_radius=4,
                         border_bottom_right_radius=4)
        pygame.draw.rect(scr, CONTROL, (x0, y, 18, top.height + inner_h + 16), border_radius=4)
        pygame.draw.rect(scr, CONTROL, (x0, top.bottom + inner_h, 160, 16), border_radius=4)
        scr.blit(label, (x0 + 10, top.centery - label.get_height() // 2))
        row = pygame.Rect(x0 + 18 + label.get_width(), top.y + 4, cond_w, MAG_H + 2)
        self.draw_row('cond', row, (cond, e_cond), (x0 + 24, top.bottom + 1), right, 'condition')
        parsed = L.parse_loop(self.fill)
        sy = top.bottom + 20
        for i in range(L.LOOP_SLOTS):
            row = pygame.Rect(x0 + 24, sy, right - x0 - 24, MAG_H + 6)
            self.draw_row(i, row, parsed[i], (row.x + 2, row.bottom + 2), right, f'line {i + 1}')
            sy += LINE_PITCH
        y = area.bottom + 6
        # status
        st = self.st
        E = '∞' if not math.isfinite(st.E) else f'{st.E * 100:.2f} mm²'
        line = f'passes {st.passes}   E = {E}'
        scr.blit(self.font.render(line, True, FG), (x0, y + 2))
        if self.run_done:
            done = self.run_done
            col = BAD if done.startswith('stopped:') else AMBER if ('safety' in done or 'never ran' in done) else GOOD
            for k, text in enumerate(self.wrap(done, right - x0)[:2]):
                scr.blit(self.small.render(text, True, col), (x0, y + 20 + 15 * k))
        elif st.env is not None and 'R' in st.env:
            R, t = st.env['R'], st.env['t']
            turn = f'R turns {L.angle_of(R):.2f}°' if L.is_rotation(R) else self.r_desc(R).strip(' —')
            scr.blit(self.small.render(f'last align: {turn},  |t| = {np.linalg.norm(t) * 10:.2f} mm', True,
                                       MUTED if L.is_rotation(R) else BAD), (x0, y + 20))
        by = SBY + SBH - 74
        ready = L.step_filled(L.STEP['I5'], self.fill)
        running = self.run is not None and self.run['mode'] == 'run'
        bx = self.button('Pause  ␣' if running else 'Run  ␣', self.toggle_run, x0, by, ready)
        bx = self.button('One pass  →', self.step_pass, bx, by, ready)
        bx = self.button('One line  .', self.step_block, bx, by, ready)
        self.button('Reset  R', self.key_reset, bx, by, True)
        scr.blit(self.small.render(f'τ = {L.TAU_DEFAULT * 100:g} mm²   ·   at most {L.MAX_PASSES} passes', True, FAINT),
                 (x0, by + 38))

    def key_reset(self):
        self.reset_pose()
        self.say('P is back at its start pose', MUTED)

    def draw_drag(self):
        if not self.drag:
            return
        kind, name = self.drag['what']
        x, y = self.drag['pos']
        if kind == 'tile':
            self.draw_tile(name, (x - TILE_W // 2, y - TILE_H // 2, TILE_W, TILE_H))
        else:
            self.draw_magnet(name, x - self.magnet_width(name) // 2, y - MAG_H // 2, lifted=True)

    # ------------------------------------------------------------ main loop

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
    ap.add_argument('--scene', type=int, default=1, choices=sorted(L.SCENES))
    ap.add_argument('--seed', type=int, default=478)
    ap.add_argument('--answers', action='store_true', help='fill every step that has a released answer')
    args = ap.parse_args()
    App(args).loop()


if __name__ == '__main__':
    main()
