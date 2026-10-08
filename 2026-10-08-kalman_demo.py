"""Kalman filter demo for Tutorial 5 (2026-10-08, Object Tracking).

A car drives on a road. You cannot see it: ten times a second a sensor reports
its position with Gaussian noise (the crosses). Fill in the blanks of the
Kalman filter on the right, step by step, and watch what each step adds.

World pane (left): the road and the measurements. Red = prediction, green =
posterior after the correction (slide 109). V reveals the true car.
Time strip (below): p_x over the last 10 s, like slide 103.
Filter pane (right): K1-K6 with blanks. Click a blank to choose a token
(matrix cells: click to cycle, right-click to cycle back). A step runs once
all its blanks are filled; C or a click on its chip checks it.

Keys
    Space           play / pause
    Right / Left    when paused: half a step forward / back (predict, then correct)
    V               reveal / hide the true car
    1-4             scenes: highway, ring road, tunnel, busy street
    R / N           restart / new random seed
    C               check the step under the mouse (or click its chip)
    A               fill the step under the mouse with its released answer
    Backspace       empty the blank under the mouse (Delete: the whole step)
    [ ]             filter's sensor noise sigma_R        ; '   filter's process noise q
    - =             true sensor noise sigma_true         L     lock sigma_R to sigma_true
    , .             playback speed                       X     time strip: p_x / p_y
    Z               zoom 3x around the estimate (it never follows the hidden car)
    P               print mu, Sigma, K                   Esc   close a menu / quit

Run:  python 2026-10-08-kalman_demo.py          (--answers fills every released answer)
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

import kf_lib as L  # noqa: E402

PAD = 12
TITLE_H = 34
SCALE = 6.0                                   # px per metre
WX, WY = PAD, TITLE_H
WW, WH = int(L.WORLD_W * SCALE), int(round(L.WORLD_H * SCALE))
SX, SY, SW, SH = PAD, WY + WH + 8, WW, 132
BX, BY, BW, BH = PAD, SY + SH + 6, WW, 46
PX = WX + WW + PAD
PW = 500
WIN_W = PX + PW + PAD
WIN_H = BY + BH + 6
PY, PH = 8, WIN_H - 16

BG = (248, 250, 252)          # #f8fafc
FG = (15, 23, 42)             # #0f172a
MUTED = (100, 116, 139)       # #64748b
FAINT = (148, 163, 184)       # #94a3b8
TRACK = (226, 232, 240)       # #e2e8f0
LINE = (203, 213, 225)        # #cbd5e1
ACCENT = (37, 99, 235)        # #2563eb
ACCENT_BG = (219, 234, 254)   # #dbeafe
PANE_BG = (255, 255, 255)
WORLD_BG = (241, 245, 249)    # #f1f5f9
ROAD = (203, 213, 225)
PRED = (220, 38, 38)          # #dc2626 prediction (slide 109 red)
PRED_BG = (254, 226, 226)
POST = (22, 163, 74)          # #16a34a posterior (slide 109 green)
POST_BG = (220, 252, 231)
AMBER = (217, 119, 6)
AMBER_BG = (254, 243, 199)
TUNNEL = (51, 65, 85)
MONO = 'menlo,consolas,dejavusansmono,couriernew,monospace'

TOKEN_W, TOKEN_H = 34, 22
CELL_W, CELL_H = 30, 19
EQ_SIZE, CELL_SIZE, NOTE_SIZE = 13, 11, 10
ROW_H = 26                                    # one equation line in the pane
STRIP_SECONDS = 10.0
FADE_FRAMES = 20
LOOKAHEAD = 10                                # frames (1 s) of A^k mu
SLIDERS = {   # name: (label, lo, hi, log, unit)
    'sigma_true': ('σ_true  real sensor noise', 0.5, 8.0, False, 'm'),
    'sigma_R': ('σ_R     filter: R = σ_R² I', 0.5, 8.0, False, 'm'),
    'q': ('q       filter: Q (white accel.)', 0.05, 30.0, True, 'm/s²'),
    'speed': ('speed   playback', 0.25, 4.0, True, '×'),
}
HINT = 'Space play/pause  ←/→ half-step  V reveal  Z zoom  1-4 scene  R restart  N seed  C check  A answer'


# ---------------------------------------------------------------- math labels

class TexCache:
    """Math labels rendered once with matplotlib's mathtext (pygame fonts lack ᵀ, ⁻¹)."""

    def __init__(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        self._figure = Figure
        self.cache = {}

    def get(self, tex, size=15, color=FG):
        key = (tex, size, color)
        if key not in self.cache:
            fig = self._figure(figsize=(1, 1), dpi=100)
            fig.text(0, 0, tex, fontsize=size, color='#%02x%02x%02x' % color, math_fontfamily='dejavusans')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, transparent=True, bbox_inches='tight', pad_inches=0.02)
            buf.seek(0)
            self.cache[key] = pygame.image.load(buf, 'tex.png').convert_alpha()
        return self.cache[key]


def tex_of(name):
    return L.SYMBOLS[name][1]


# ---------------------------------------------------------------- drawing helpers

VIEW = {'cx': L.WORLD_W / 2, 'cy': L.WORLD_H / 2, 'z': 1}   # world pane camera (Z zooms)
ZOOM = 3


def w2s(p):
    s = SCALE * VIEW['z']
    return WX + WW / 2 + (p[0] - VIEW['cx']) * s, WY + WH / 2 - (p[1] - VIEW['cy']) * s


def ellipse_points(mean, C, chi2=L.CHI2_95, n=64):
    """Screen polygon of the chi2 region of N(mean, C), or None."""
    if mean is None or C is None or not np.all(np.isfinite(C)) or not np.all(np.isfinite(mean)):
        return None
    lam, V = np.linalg.eigh(0.5 * (C + C.T))
    lam = np.clip(lam, 0, None)
    if np.sqrt(lam.max() * chi2) > 2000:       # far bigger than the world: skip
        return None
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = mean[None, :] + (V @ (np.sqrt(chi2 * lam)[:, None] * np.stack([np.cos(t), np.sin(t)]))).T
    return [w2s(p) for p in pts]


def dashed_poly(surf, color, pts, width=1, dash=6, closed=True):
    if pts is None:
        return
    seq = list(pts) + ([pts[0]] if closed else [])
    on, left = True, dash
    for (x0, y0), (x1, y1) in zip(seq, seq[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        pos = 0.0
        while pos < seg:
            step = min(left, seg - pos)
            if on:
                a, b = pos / seg, (pos + step) / seg
                pygame.draw.line(surf, color, (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a),
                                 (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b), width)
            pos += step
            left -= step
            if left <= 1e-9:
                on, left = not on, dash


def cross(surf, color, c, r=4, width=2):
    x, y = c
    pygame.draw.line(surf, color, (x - r, y - r), (x + r, y + r), width)
    pygame.draw.line(surf, color, (x - r, y + r), (x + r, y - r), width)


def arrow(surf, color, a, b, width=2, head=8):
    pygame.draw.line(surf, color, a, b, width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    if math.hypot(b[0] - a[0], b[1] - a[1]) < head:
        return
    for s in (-1, 1):
        pygame.draw.line(surf, color, b, (b[0] - head * math.cos(ang + s * 0.45),
                                          b[1] - head * math.sin(ang + s * 0.45)), width)


def make_car():
    """Top-down car facing +x, ~1.4x real size so it reads at this scale."""
    w, h = int(4.6 * SCALE * 1.4), int(2.0 * SCALE * 1.4)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(s, FG, s.get_rect(), border_radius=5)
    pygame.draw.rect(s, ACCENT, s.get_rect().inflate(-2, -2), border_radius=5)
    pygame.draw.rect(s, (191, 219, 254), (int(w * 0.58), 3, int(w * 0.16), h - 6), border_radius=2)
    pygame.draw.rect(s, (191, 219, 254), (int(w * 0.12), 4, int(w * 0.1), h - 8), border_radius=2)
    for x in (int(w * 0.92),):
        pygame.draw.rect(s, (254, 240, 138), (x, 2, 3, 3))
        pygame.draw.rect(s, (254, 240, 138), (x, h - 5, 3, 3))
    return s


# ---------------------------------------------------------------- app

class App:
    def __init__(self, args):
        pygame.init()
        pygame.display.set_caption('Tutorial 5 - Kalman filter')
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.font = pygame.font.SysFont(MONO, 14)
        self.bold = pygame.font.SysFont(MONO, 14, bold=True)
        self.small = pygame.font.SysFont(MONO, 12)
        self.big = pygame.font.SysFont(MONO, 16, bold=True)
        self.tex = TexCache()
        self.car = make_car()
        self.clock = pygame.time.Clock()

        self.blanks = L.empty_blanks()
        self.released = L.released_answers()
        self.released_t = time.time()
        if args.answers:
            for d in self.released.values():
                self.blanks.update({k: list(v) for k, v in d.items()})
        self.checked = {}                 # step -> (signature, correct)
        self.attempts = {s.key: 0 for s in L.STEPS}
        self.params = L.Params()
        self.lock = False
        self.speed = 1.0
        self.strip_coord = 0
        self.popup = None                 # (slot, idx, anchor rect)
        self.popup_last = None
        self.drag = None
        self.hits = []
        self.hover_step = None
        self.toast, self.toast_t = '', 0.0
        self.load_scene(args.scene, args.seed)

    # -------------------------------------------------------- state

    def load_scene(self, key, seed):
        self.scene = L.make_scene(key, seed)
        self.params.q = self.scene.q_default
        self.revealed = False
        self.road_layers = {}
        self.restart()

    def restart(self):
        self.cur, self.phase = 0, 'correct'
        self.playing, self.acc, self.ended = True, 0.0, False
        self.model = L.build_model(self.blanks)
        self.recompute()

    def recompute(self):
        self.run = L.KalmanRun(self.scene, self.model, self.params)
        self.run.compute(self.cur)
        self._stats_key = None
        self.dirty = False

    def blanks_changed(self):
        self.model = L.build_model(self.blanks)
        self.dirty = True

    def advance(self):
        if self.cur + 1 >= self.scene.n:
            self.playing, self.ended = False, True
            return False
        self.cur += 1
        self.run.compute(self.cur)
        return True

    def half_step(self, direction):
        self.playing, self.acc = False, 0.0
        if direction > 0:
            if self.phase == 'correct':
                if self.advance():
                    self.phase = 'predict'
            else:
                self.phase = 'correct'
        else:
            if self.phase == 'correct' and self.cur > 0:
                self.phase = 'predict'
            elif self.cur > 0:
                self.cur -= 1
                self.phase = 'correct'
            self.ended = False

    def stats(self):
        upto = self.cur - (self.phase == 'predict')
        key = (id(self.run), upto)
        if key != self._stats_key:
            self._stats, self._stats_key = L.stats(self.run, upto), key
        return self._stats

    def say(self, msg):
        self.toast, self.toast_t = msg, time.time()

    def set_param(self, name, value):
        lo, hi = SLIDERS[name][1:3]
        value = float(np.clip(value, lo, hi))
        if name == 'speed':
            self.speed = value
            return
        setattr(self.params, name, value)
        if self.lock and name in ('sigma_true', 'sigma_R'):
            self.params.sigma_true = self.params.sigma_R = value
        self.dirty = True

    def get_param(self, name):
        return self.speed if name == 'speed' else getattr(self.params, name)

    def nudge(self, name, up):
        v = self.get_param(name)
        if SLIDERS[name][3]:
            self.set_param(name, v * (1.25 if up else 0.8))
        else:
            self.set_param(name, round(v + (0.25 if up else -0.25), 2))

    def check(self, key):
        step = L.STEP[key]
        state = self.model.status[key][0]
        if state in ('todo', 'partial'):
            self.say(f'{key}: fill every blank first')
            return
        self.attempts[key] += 1
        ok = L.check_step(step, self.blanks)
        self.checked[key] = (L.step_signature(step, self.blanks), ok)
        self.say(f'{key} {step.title}: ' + ('correct ✓' if ok else f'not yet ✗ (checked {self.attempts[key]}×)'))

    def fill_answer(self, key):
        if key not in self.released:
            self.say(f'no released answer for {key} yet')
            return
        self.blanks.update({k: list(v) for k, v in self.released[key].items()})
        self.blanks_changed()
        self.say(f'{key} filled with the released answer')

    def clear_step(self, key):
        for slot, n in L.blank_slots(L.STEP[key]):
            self.blanks[slot] = [None] * n
        self.blanks_changed()

    # -------------------------------------------------------- events

    def hit(self, pos):
        if self.popup is not None:
            for rect, kind, data in reversed(self.hits):
                if kind == 'token' and rect.collidepoint(pos):
                    return kind, data
        for rect, kind, data in reversed(self.hits):
            if rect.collidepoint(pos):
                return kind, data
        return None, None

    def on_click(self, pos, button):
        kind, data = self.hit(pos)
        if self.popup is not None:
            if kind == 'token':
                slot, idx = self.popup[:2]
                self.blanks[slot][idx] = data
                self.blanks_changed()
            self.popup = None
            if kind == 'token':
                return
            if kind == 'blank' and data[:2] == self.popup_last:
                return
        if kind == 'blank':
            slot, idx, cell = data
            if cell:
                opts = L.CELLS[slot]
                cur = self.blanks[slot][idx]
                i = -1 if cur is None else opts.index(cur)
                self.blanks[slot][idx] = opts[(i + (1 if button == 1 else -1)) % len(opts)]
                self.blanks_changed()
            elif button == 3:
                self.blanks[slot][idx] = None
                self.blanks_changed()
            else:
                rect = next(r for r, k, d in self.hits if k == 'blank' and d == data)
                self.popup = (slot, idx, rect)
            self.popup_last = (slot, idx)
        elif kind == 'chip':
            self.check(data)
        elif kind == 'answer':
            self.fill_answer(data)
        elif kind == 'slider':
            self.drag = data
            self.drag_to(pos)
        elif kind == 'lock':
            self.toggle_lock()

    def drag_to(self, pos):
        rect = next(r for r, k, d in self.hits if k == 'slider' and d == self.drag)
        t = np.clip((pos[0] - rect.x) / rect.w, 0, 1)
        _, lo, hi, log, _ = SLIDERS[self.drag]
        self.set_param(self.drag, lo * (hi / lo) ** t if log else lo + t * (hi - lo))

    def on_wheel(self, pos, dy):
        kind, data = self.hit(pos)
        if kind == 'blank':
            slot, idx, cell = data
            opts = L.CELLS[slot] if cell else L.TOKENS
            cur = self.blanks[slot][idx]
            i = -1 if cur is None else opts.index(cur)
            self.blanks[slot][idx] = opts[(i - dy) % len(opts)]
            self.blanks_changed()
        elif kind == 'slider':
            self.nudge(data, dy > 0)

    def toggle_lock(self):
        self.lock = not self.lock
        if self.lock:
            self.set_param('sigma_R', self.params.sigma_true)
        self.say('σ_R locked to σ_true' if self.lock else 'σ_R unlocked')

    def on_key(self, e):
        k = e.key
        if k == pygame.K_ESCAPE:
            if self.popup is not None:
                self.popup = None
                return True
            return False
        if k == pygame.K_SPACE:
            if self.ended:
                self.restart()
            else:
                self.playing = not self.playing
                self.phase, self.acc = 'correct', 0.0
        elif k == pygame.K_RIGHT:
            self.half_step(+1)
        elif k == pygame.K_LEFT:
            self.half_step(-1)
        elif k == pygame.K_v:
            self.revealed = not self.revealed
        elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            self.load_scene(k - pygame.K_0, self.scene.seed)
        elif k == pygame.K_r:
            self.restart()
        elif k == pygame.K_n:
            self.load_scene(self.scene.key, int(np.random.default_rng().integers(1, 10000)))
        elif k == pygame.K_c and self.hover_step:
            self.check(self.hover_step)
        elif k == pygame.K_a and self.hover_step:
            self.fill_answer(self.hover_step)
        elif k == pygame.K_BACKSPACE:
            kind, data = self.hit(pygame.mouse.get_pos())
            if kind == 'blank':
                self.blanks[data[0]][data[1]] = None
                self.blanks_changed()
        elif k == pygame.K_DELETE and self.hover_step:
            self.clear_step(self.hover_step)
        elif k == pygame.K_LEFTBRACKET:
            self.nudge('sigma_R', False)
        elif k == pygame.K_RIGHTBRACKET:
            self.nudge('sigma_R', True)
        elif k == pygame.K_SEMICOLON:
            self.nudge('q', False)
        elif k == pygame.K_QUOTE:
            self.nudge('q', True)
        elif k == pygame.K_MINUS:
            self.nudge('sigma_true', False)
        elif k == pygame.K_EQUALS:
            self.nudge('sigma_true', True)
        elif k == pygame.K_COMMA:
            self.nudge('speed', False)
        elif k == pygame.K_PERIOD:
            self.nudge('speed', True)
        elif k == pygame.K_l:
            self.toggle_lock()
        elif k == pygame.K_z:
            VIEW['z'] = 1 if VIEW['z'] > 1 else ZOOM
            if VIEW['z'] > 1 and self.run.frames[self.cur].est is not None:
                VIEW['cx'], VIEW['cy'] = self.run.frames[self.cur].est
        elif k == pygame.K_x:
            self.strip_coord = 1 - self.strip_coord
        elif k == pygame.K_p:
            self.print_state()
        return True

    def print_state(self):
        f = self.run.frames[self.cur]
        np.set_printoptions(precision=3, suppress=True)
        print(f'--- frame {self.cur} (t = {self.cur * L.DT:.1f} s), {self.scene.name}')
        for name in ('y', 'mu_pred', 'S_pred', 'r', 'S', 'K', 'mu', 'Sigma'):
            v = getattr(f, name)
            if v is not None:
                print(f'{name} =\n{np.asarray(v)}')
        print('truth =', self.scene.pos[self.cur], self.scene.vel[self.cur])

    # -------------------------------------------------------- world pane

    def road_layer(self, z):
        """The static background (grid, road, tunnel) at zoom z, drawn once per scene."""
        if z in self.road_layers:
            return self.road_layers[z]
        sc = SCALE * z
        w, h = int(L.WORLD_W * sc), int(L.WORLD_H * sc)
        s = pygame.Surface((w, h))
        s.fill(WORLD_BG)

        def tr(p):
            return p[0] * sc, (L.WORLD_H - p[1]) * sc
        for x in range(0, int(L.WORLD_W) + 1, 10 if z == 1 else 5):
            pygame.draw.line(s, TRACK, (x * sc, 0), (x * sc, h))
        for y in range(0, int(L.WORLD_H) + 1, 10 if z == 1 else 5):
            pygame.draw.line(s, TRACK, (0, h - y * sc), (w, h - y * sc))
        road = self.scene.road
        pts = [tr(p) for p in road.xy[::2]]
        if road.closed:
            pts.append(pts[0])
        rw = int(L.ROAD_WIDTH * sc)
        pygame.draw.lines(s, ROAD, False, pts, rw)
        for p in pts:
            pygame.draw.circle(s, ROAD, p, rw // 2)
        dash = [tr(p) for p in road.xy[::2]]
        for i in range(0, len(dash) - 4, 12):
            pygame.draw.line(s, BG, dash[i], dash[i + 4], 2 * z)
        tp = road.tunnel_polyline()
        if tp is not None:
            tpts = [tr(p) for p in tp]
            pygame.draw.lines(s, TUNNEL, False, tpts, rw + 8)
            for p in (tpts[0], tpts[-1]):
                pygame.draw.circle(s, TUNNEL, p, (rw + 8) // 2)
            for p in (tpts[len(tpts) // 4], tpts[3 * len(tpts) // 4]) if z > 1 else (tpts[len(tpts) // 2],):
                lab = self.small.render('TUNNEL: no sensor', True, BG)
                s.blit(lab, lab.get_rect(center=p))
        self.road_layers[z] = s
        return s

    def update_view(self):
        if VIEW['z'] == 1:
            VIEW['cx'], VIEW['cy'] = L.WORLD_W / 2, L.WORLD_H / 2
            return
        f = self.run.frames[self.cur]
        target = None
        if self.phase == 'predict' and f.mu_pred is not None:
            target = f.mu_pred[:2, 0]
        elif f.est is not None:
            target = f.est
        if target is None or not np.all(np.isfinite(target)) or np.abs(target).max() > 1e4:
            return
        VIEW['cx'] += 0.15 * (target[0] - VIEW['cx'])
        VIEW['cy'] += 0.15 * (target[1] - VIEW['cy'])

    def draw_world(self):
        scr = self.screen
        clip = scr.get_clip()
        scr.set_clip(pygame.Rect(WX, WY, WW, WH))
        scr.fill(WORLD_BG)
        self.update_view()
        scr.blit(self.road_layer(VIEW['z']), w2s((0.0, L.WORLD_H)))
        over = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        sc, run, m, k = self.scene, self.run, self.model, self.cur
        f = run.frames[k]
        prev = run.frames[k - 1] if k > 0 else None
        predicting = self.phase == 'predict'
        shown_k = k - 1 if predicting else k

        # measurements, fading
        for j in range(max(0, shown_k - FADE_FRAMES + 1), shown_k + 1):
            y = run.frames[j].y
            if y is None:
                continue
            age = shown_k - j
            if age == 0:
                cross(scr, FG, w2s(y), 5, 2)
            else:
                a = int(200 * (1 - age / FADE_FRAMES))
                cross(over, (*MUTED, a), w2s(y), 4, 2)

        running = f.mu is not None or (prev is not None and prev.mu is not None)
        # estimate trail
        trail = [w2s(g.mu[:2, 0]) for g in run.frames[max(0, shown_k - 50):shown_k + 1] if g.mu is not None]
        if len(trail) > 1:
            pygame.draw.lines(over, (*POST, 150), False, trail, 2)

        # stage 1: look-ahead from differenced measurements (no filter yet)
        if m.A is not None and not running and run.failed is None and k >= 1:
            ys = [g.y for g in run.frames[max(0, shown_k - 1):shown_k + 1]]
            if len(ys) == 2 and all(y is not None for y in ys):
                x = np.r_[ys[1], (ys[1] - ys[0]) / L.DT]
                self.draw_lookahead(over, x, (*MUTED, 220))
                lab = self.small.render('velocity from the last 2 crosses', True, MUTED)
                over.blit(lab, (w2s(ys[1])[0] + 8, w2s(ys[1])[1] + 8))

        if f.error is None and running:
            if predicting:
                if prev is not None and prev.mu is not None:          # previous posterior, faint
                    pts = ellipse_points(prev.mu[:2, 0], prev.Sigma[:2, :2])
                    if pts:
                        pygame.draw.polygon(over, (*POST, 40), pts)
                        pygame.draw.polygon(over, (*POST, 120), pts, 1)
                self.draw_prior(over, f, strong=True)
            else:
                self.draw_prior(over, f, strong=not f.corrected)
                if m.ok('K3') and f.y is not None:
                    R = self.params.sigma_R ** 2 * np.eye(2)
                    dashed_poly(over, (*MUTED, 200), ellipse_points(f.y, R), 1, 4)
                if f.r is not None and f.S is not None and f.mu_pred is not None and m.H is not None:
                    hm = (m.H @ f.mu_pred)[:, 0]
                    dashed_poly(over, (*AMBER, 230), ellipse_points(hm, f.S), 2, 6)
                    arrow(over, (*AMBER, 255), w2s(hm), w2s(f.y), 2)
                    if f.d2 is not None:
                        txt = f'd = {math.sqrt(max(f.d2, 0)):.1f}' + ('  gated' if f.gated else '')
                        col = PRED if f.gated else AMBER
                        lab = self.small.render(txt, True, col)
                        p = w2s(f.y)
                        over.blit(lab, (p[0] + 9, p[1] - 18))
                    if f.gated:
                        pygame.draw.circle(over, PRED, w2s(f.y), 9, 2)
                if f.corrected:
                    pts = ellipse_points(f.mu[:2, 0], f.Sigma[:2, :2])
                    if pts:
                        pygame.draw.polygon(over, (*POST, 70), pts)
                        pygame.draw.polygon(over, (*POST, 255), pts, 2)
                    pygame.draw.circle(over, POST, w2s(f.mu[:2, 0]), 4)
                if f.mu is not None and m.A is not None:
                    self.draw_lookahead(over, f.mu[:, 0], (*PRED, 160))

        if self.revealed:
            j0 = max(0, k - 50)
            tr = [w2s(p) for p in sc.pos[j0:k + 1]]
            if len(tr) > 1:
                pygame.draw.lines(over, (*FG, 170), False, tr, 2)
            if self.playing and k + 1 < sc.n:
                a = min(self.acc / L.DT, 1.0)
                p = (1 - a) * sc.pos[k] + a * sc.pos[k + 1]
                v = (1 - a) * sc.vel[k] + a * sc.vel[k + 1]
            else:
                p, v = sc.pos[k], sc.vel[k]
            car = pygame.transform.rotozoom(self.car, math.degrees(math.atan2(v[1], v[0])),
                                            1.0 if VIEW['z'] == 1 else VIEW['z'] / 1.4)
            if sc.blind[k]:
                car.set_alpha(110)
            est = f.est if not predicting else (f.mu_pred[:2, 0] if f.mu_pred is not None else None)
            if est is not None:
                pygame.draw.line(over, (*FG, 120), w2s(p), w2s(est), 1)
            over.blit(car, car.get_rect(center=w2s(p)))
        scr.blit(over, (0, 0))
        bar_m = 10 if VIEW['z'] == 1 else 5
        x0, y0 = WX + 12, WY + WH - 12
        pygame.draw.line(scr, MUTED, (x0, y0), (x0 + bar_m * SCALE * VIEW['z'], y0), 2)
        scr.blit(self.small.render(f'{bar_m} m' + ('   zoom 3× follows the estimate (Z)' if VIEW['z'] > 1 else ''),
                                   True, MUTED), (x0 + 6 + bar_m * SCALE * VIEW['z'], y0 - 7))

        if f.error:
            self.banner(f'filter stopped at t = {k * L.DT:.1f} s: {f.error}', PRED)
        elif self.run.failed:
            self.banner(f'filter stopped at t = {self.run.failed[0] * L.DT:.1f} s: {self.run.failed[1]}', PRED)
        elif self.ended:
            self.banner('end of the run: Space or R to restart, N for a new seed', MUTED)
        scr.set_clip(clip)
        pygame.draw.rect(scr, LINE, (WX, WY, WW, WH), 1)
        badge = 'REVEALED (V)' if self.revealed else 'car hidden (V reveals)'
        lab = self.bold.render(badge, True, AMBER if self.revealed else MUTED)
        r = lab.get_rect(topright=(WX + WW - 8, WY + 6))
        pygame.draw.rect(scr, AMBER_BG if self.revealed else WORLD_BG, r.inflate(10, 4), border_radius=4)
        scr.blit(lab, r)

    def draw_prior(self, over, f, strong):
        if f.mu_pred is None:
            return
        pts = ellipse_points(f.mu_pred[:2, 0], f.S_pred[:2, :2])
        if pts:
            if strong:
                pygame.draw.polygon(over, (*PRED, 40), pts)
            dashed_poly(over, (*PRED, 255 if strong else 170), pts, 2, 7)
        c = w2s(f.mu_pred[:2, 0])
        pygame.draw.circle(over, (*PRED, 255), c, 3)
        if self.model.H is not None and self.model.ok('K3'):
            hm = w2s((self.model.H @ f.mu_pred)[:, 0])
            pygame.draw.polygon(over, PRED, [(hm[0], hm[1] - 5), (hm[0] + 5, hm[1]),
                                             (hm[0], hm[1] + 5), (hm[0] - 5, hm[1])], 1)

    def draw_lookahead(self, over, x, color):
        A = self.model.A
        pts, x = [], np.asarray(x, float)
        for _ in range(LOOKAHEAD):
            x = A @ x
            if not np.all(np.isfinite(x)) or np.abs(x).max() > 1e4:
                break
            pts.append(w2s(x[:2]))
        for p in pts[1::2]:
            pygame.draw.circle(over, color, p, 2)
        if pts:
            pygame.draw.circle(over, color, pts[-1], 3, 1)

    def banner(self, msg, color):
        lab = self.font.render(msg, True, color)
        r = lab.get_rect(midbottom=(WX + WW // 2, WY + WH - 8))
        pygame.draw.rect(self.screen, PANE_BG, r.inflate(16, 8), border_radius=6)
        pygame.draw.rect(self.screen, color, r.inflate(16, 8), 1, border_radius=6)
        self.screen.blit(lab, r)

    # -------------------------------------------------------- time strip (slide 103)

    def draw_strip(self):
        scr, run, c = self.screen, self.run, self.strip_coord
        pygame.draw.rect(scr, PANE_BG, (SX, SY, SW, SH))
        pygame.draw.rect(scr, LINE, (SX, SY, SW, SH), 1)
        n = int(STRIP_SECONDS / L.DT)
        k_end = self.cur
        k0 = max(0, k_end - n + 1)
        frames = run.frames[k0:k_end + 1]
        predicting = self.phase == 'predict'
        vals = []
        for g in frames:
            if g.y is not None:
                vals.append(g.y[c])
            if g.mu is not None:
                vals.append(g.mu[c, 0])
        if self.revealed:
            vals += list(self.scene.pos[k0:k_end + 1, c])
        if not vals:
            return
        lo, hi = min(vals), max(vals)
        mid, half = (lo + hi) / 2, max((hi - lo) / 2 * 1.15, 4.0)
        lo, hi = mid - half, mid + half
        x0, x1, y0, y1 = SX + 44, SX + SW - 10, SY + 20, SY + SH - 8

        def X(kk):
            return x0 + (kk - (k_end - n + 1)) / (n - 1) * (x1 - x0)

        def Y(v):
            return y1 - (v - lo) / (hi - lo) * (y1 - y0)

        for v in np.linspace(lo, hi, 3):
            scr.blit(self.small.render(f'{v:5.0f}', True, FAINT), (SX + 4, Y(v) - 7))
            pygame.draw.line(scr, TRACK, (x0, Y(v)), (x1, Y(v)))
        scr.blit(self.small.render(('p_x' if c == 0 else 'p_y') + ' (m) over the last 10 s   [X]',
                                   True, MUTED), (SX + 6, SY + 3))
        legend = [('×', MUTED, 'measurement'), ('*', PRED, 'predicted ±2σ'), ('+', POST, 'corrected ±2σ')]
        if self.revealed:
            legend.append(('—', FG, 'truth'))
        lx = SX + SW - 10
        for sym, col, txt in reversed(legend):
            lab = self.small.render(f'{sym} {txt}', True, col)
            lx -= lab.get_width() + 14
            scr.blit(lab, (lx, SY + 3))
        old = scr.get_clip()
        scr.set_clip(pygame.Rect(x0 - 6, y0 - 4, x1 - x0 + 12, y1 - y0 + 8))
        if self.revealed:
            pts = [(X(kk), Y(self.scene.pos[kk, c])) for kk in range(k0, k_end + 1)]
            if len(pts) > 1:
                pygame.draw.lines(scr, FG, False, pts, 1)
        for g in frames:
            kk = g.k
            last = kk == k_end
            if g.y is not None and not (last and predicting):
                cross(scr, MUTED, (X(kk), Y(g.y[c])), 3, 1)
            if g.mu_pred is not None and g.S_pred is not None:
                xx, sd = X(kk) - 2, 2 * math.sqrt(max(g.S_pred[c, c], 0))
                ym = Y(g.mu_pred[c, 0])
                pygame.draw.line(scr, PRED_BG if not last else PRED, (xx, Y(g.mu_pred[c, 0] - sd)),
                                 (xx, Y(g.mu_pred[c, 0] + sd)), 1)
                for a in (0, 60, 120):
                    dx, dy = 3 * math.cos(math.radians(a)), 3 * math.sin(math.radians(a))
                    pygame.draw.line(scr, PRED, (xx - dx, ym - dy), (xx + dx, ym + dy), 1)
            if g.corrected and g.Sigma is not None and not (last and predicting):
                xx, sd = X(kk) + 2, 2 * math.sqrt(max(g.Sigma[c, c], 0))
                ym = Y(g.mu[c, 0])
                pygame.draw.line(scr, (134, 239, 172), (xx, Y(g.mu[c, 0] - sd)), (xx, Y(g.mu[c, 0] + sd)), 1)
                pygame.draw.line(scr, POST, (xx - 3, ym), (xx + 3, ym), 2)
                pygame.draw.line(scr, POST, (xx, ym - 3), (xx, ym + 3), 2)
        scr.set_clip(old)

    # -------------------------------------------------------- mode bar

    def draw_bar(self):
        scr = self.screen
        st = self.stats()

        def fmt(v, f):
            return '—' if v is None else f.format(v)
        items = [('measurements RMSE', fmt(st['meas_rmse'], '{:.1f} m'), FG),
                 ('estimate RMSE', fmt(st['est_rmse'], '{:.1f} m'), POST if (st['est_rmse'] or 1e9) <
                  (st['meas_rmse'] or 0) else FG),
                 ('truth in 95% ellipse', fmt(st['coverage'] and 100 * st['coverage'], '{:.0f} %'), FG),
                 ('NIS', fmt(st['nis'], '{:.1f}') + ' (≈2 if R, Q right)', FG)]
        x = BX
        for name, val, col in items:
            lab = self.small.render(name, True, MUTED)
            scr.blit(lab, (x, BY + 2))
            v = self.bold.render(val, True, col)
            scr.blit(v, (x, BY + 15))
            x += max(lab.get_width(), v.get_width()) + 22
        scr.blit(self.small.render(HINT, True, FAINT), (BX, BY + 33))

    def draw_title(self):
        scr = self.screen
        sc = self.scene
        t = self.big.render(f'{sc.key}  {sc.name}', True, FG)
        scr.blit(t, (WX, 8))
        if self.playing:
            state = f'playing {self.speed:.2g}×'
        else:
            state = 'paused, after ' + self.phase
        lab = self.font.render(f'seed {sc.seed}   {state}   t {self.cur * L.DT:5.1f} s', True, FG)
        scr.blit(lab, lab.get_rect(topright=(WX + WW, 10)))
        about = sc.about
        room = WW - t.get_width() - lab.get_width() - 40
        while self.small.size(about)[0] > room and len(about) > 4:
            about = about[:-2] + '…'
        scr.blit(self.small.render(about, True, MUTED), (WX + t.get_width() + 14, 12))
        if self.toast and time.time() - self.toast_t < 3.0:
            lab = self.bold.render(self.toast, True, ACCENT)
            r = lab.get_rect(midtop=(WX + WW // 2, WY + 8))
            pygame.draw.rect(scr, ACCENT_BG, r.inflate(14, 6), border_radius=5)
            scr.blit(lab, r)

    # -------------------------------------------------------- filter pane

    def chip(self, key):
        state, msg = self.model.status[key]
        chk = self.checked.get(key)
        sig = L.step_signature(L.STEP[key], self.blanks) if state not in ('todo', 'partial') else None
        if chk and chk[0] == sig:
            return ('✓ correct', POST, POST_BG) if chk[1] else ('✗ not yet', PRED, PRED_BG)
        if state == 'partial':
            return ('filling', AMBER, AMBER_BG)
        return {'todo': ('TODO', MUTED, TRACK), 'error': ('shape error', PRED, PRED_BG),
                'ok': ('runs', ACCENT, ACCENT_BG)}[state]

    def blank(self, x, y, slot, idx, cell, mouse):
        tok = self.blanks[slot][idx]
        w, h = (CELL_W, CELL_H) if cell else (TOKEN_W, TOKEN_H)
        surf = self.tex.get(tex_of(tok), CELL_SIZE if cell else EQ_SIZE) if tok is not None else None
        if surf is not None:
            w = max(w, surf.get_width() + 8)
        r = pygame.Rect(x, y - h // 2, w, h)
        hov = r.collidepoint(mouse) or (self.popup is not None and self.popup[:2] == (slot, idx))
        pygame.draw.rect(self.screen, ACCENT_BG if tok is not None else (241, 245, 249), r, border_radius=4)
        pygame.draw.rect(self.screen, ACCENT if hov else (LINE if tok is not None else FAINT), r,
                         2 if hov else 1, border_radius=4)
        if surf is not None:
            self.screen.blit(surf, surf.get_rect(center=r.center))
        else:
            lab = self.small.render('?', True, FAINT)
            self.screen.blit(lab, lab.get_rect(center=r.center))
        self.hits.append((r, 'blank', (slot, idx, cell)))
        return r.right

    def put_tex(self, tex, x, y, size=EQ_SIZE, color=FG):
        s = self.tex.get(tex, size, color)
        self.screen.blit(s, s.get_rect(midleft=(x, y)))
        return x + s.get_width()

    def equation(self, eq_key, x, y, mouse):
        eq = L.EQUATIONS[eq_key]
        x = self.put_tex(tex_of(eq.lhs) + r'$\;=\;$', x, y)

        def terms(ts, x):
            for i, (sign, factors) in enumerate(ts):
                if sign < 0:
                    x = self.put_tex(r'$\;-\;$' if i else r'$-$', x, y)
                elif i:
                    x = self.put_tex(r'$\;+\;$', x, y)
                for fct in factors:
                    if isinstance(fct, tuple):
                        x = self.put_tex(r'$($', x, y)
                        x = terms(fct[1], x)
                        x = self.put_tex(r'$)$', x, y) + 2
                    elif isinstance(fct, int):
                        x = self.blank(x + 2, y, eq_key, fct, False, mouse) + 2
                    else:
                        x = self.put_tex(tex_of(fct), x, y) + 1
            return x
        return terms(eq.terms, x)

    def matrix(self, key, x, top, mouse, rows, cols, fixed, row_labels, col_labels):
        """Matrix with cell blanks; fixed maps (i, j) -> tex of a given entry. Returns its right edge."""
        pitch = CELL_W + 3
        h = rows * (CELL_H + 2)
        for j, cl in enumerate(col_labels):
            lab = self.small.render(cl, True, FAINT)
            self.screen.blit(lab, lab.get_rect(midbottom=(x + 6 + j * pitch + CELL_W // 2, top + 1)))
        right = x + 6 + cols * pitch + 2
        for bx, d in ((x, 4), (right, -4)):
            pygame.draw.line(self.screen, FG, (bx, top + 1), (bx, top + h - 1), 2)
            pygame.draw.line(self.screen, FG, (bx, top + 1), (bx + d, top + 1), 2)
            pygame.draw.line(self.screen, FG, (bx, top + h - 1), (bx + d, top + h - 1), 2)
        b = 0
        for i in range(rows):
            cy = top + i * (CELL_H + 2) + CELL_H // 2 + 1
            for j in range(cols):
                cx = x + 6 + j * pitch
                if (i, j) in fixed:
                    s = self.tex.get(fixed[(i, j)], CELL_SIZE, MUTED)
                    self.screen.blit(s, s.get_rect(center=(cx + CELL_W // 2, cy)))
                else:
                    self.blank(cx, cy, key, b, True, mouse)
                    b += 1
            lab = self.small.render(row_labels[i], True, FAINT)
            self.screen.blit(lab, (right + 6, cy - 7))
        return right + 34

    STEP_H = {'K1': 4 * (CELL_H + 2) + 14, 'K2': 2 * ROW_H, 'K3': 2 * (CELL_H + 2) + 14,
              'K4': 2 * ROW_H, 'K6': 40, 'K5': 3 * ROW_H}

    def step_height(self, key):
        h = max(self.STEP_H[key], 40)
        return h + (14 if self.model.status[key][0] == 'error' else 0)

    def draw_pane(self):
        scr = self.screen
        mouse = pygame.mouse.get_pos()
        self.hits = []
        pygame.draw.rect(scr, PANE_BG, (PX, PY, PW, PH), border_radius=8)
        pygame.draw.rect(scr, LINE, (PX, PY, PW, PH), 1, border_radius=8)
        x0, y = PX + 12, PY + 8
        scr.blit(self.big.render('KALMAN FILTER', True, FG), (x0, y))
        lab = self.small.render('state x = (p_x, p_y, v_x, v_y)   slides 105-112', True, MUTED)
        scr.blit(lab, lab.get_rect(topright=(PX + PW - 12, y + 3)))
        y += 26
        self.put_tex(r'K0 prior (given):  $\mu_0=(y_1,\ \mathrm{10\,m/s\ along\ road})$,  '
                     r'$\Sigma_0=\mathrm{diag}(\sigma_R^2,\sigma_R^2,3^2,3^2)$', x0, y, 9, MUTED)
        y += 14
        self.hover_step = None
        for phase, title in (('predict', '1  PREDICT'), ('correct', '2  CORRECT')):
            col = PRED if phase == 'predict' else POST
            steps = [s for s in L.STEPS if s.phase == phase]
            block_h = 20 + sum(self.step_height(s.key) + 6 for s in steps)
            if not self.playing and self.phase == phase:
                pygame.draw.rect(scr, PRED_BG if phase == 'predict' else POST_BG,
                                 (PX + 4, y - 2, PW - 8, block_h + 2), border_radius=6)
            scr.blit(self.bold.render(title, True, col), (x0, y))
            pygame.draw.line(scr, col, (x0 + 104, y + 8), (PX + PW - 12, y + 8), 1)
            y += 20
            for s in steps:
                y = self.draw_step(s, x0, y, mouse) + 6
        self.draw_live(x0, y + 2)
        self.draw_sliders(x0, PY + PH - 4 * 20 - 6)
        if self.popup is not None:
            self.draw_popup()

    def draw_step(self, s, x0, y, mouse):
        scr = self.screen
        h = self.step_height(s.key)
        area = pygame.Rect(PX + 4, y - 3, PW - 8, h + 4)
        if area.collidepoint(mouse):
            self.hover_step = s.key
            pygame.draw.rect(scr, LINE, area, 1, border_radius=5)
        key = self.bold.render(s.key, True, FG)
        scr.blit(key, (x0, y))
        scr.blit(self.small.render(s.title, True, MUTED), (x0 + 26, y + 2))
        text, fg, bg = self.chip(s.key)
        lab = self.small.render(text, True, fg)
        r = lab.get_rect(topleft=(x0 + 4, y + 21)).inflate(8, 4)
        pygame.draw.rect(scr, bg, r, border_radius=4)
        scr.blit(lab, lab.get_rect(center=r.center))
        self.hits.append((r, 'chip', s.key))
        if s.key in self.released:
            al = self.small.render('A answer', True, ACCENT)
            ar = al.get_rect(topleft=(r.right + 10, r.y + 2)).inflate(8, 4)
            pygame.draw.rect(scr, ACCENT, ar, 1, border_radius=4)
            scr.blit(al, al.get_rect(center=ar.center))
            self.hits.append((ar, 'answer', s.key))
        ex = x0 + 156
        if s.key == 'K1':
            fixed = {(i, j): ('$1$' if i == j else '$0$') for i in range(4) for j in range(2)}
            top = y + 12
            self.put_tex(r'$A\,=$', ex, top + 2 * (CELL_H + 2))
            right = self.matrix('K1', ex + 36, top, mouse, 4, 4, fixed, ['p_x', 'p_y', 'v_x', 'v_y'],
                                ['p_x', 'p_y', 'v_x', 'v_y'])
            self.put_tex(r'$x_t = A\,x_{t-1} + v_t$', right, top + 14, NOTE_SIZE, MUTED)
            self.put_tex(r'$v_t \sim \mathcal{N}(0, Q)$', right, top + 34, NOTE_SIZE, MUTED)
            self.put_tex(r'$Q$ given (knob $q$)', right, top + 54, NOTE_SIZE, MUTED)
            self.put_tex(r'$\Delta t = 0.1$ s', right, top + 74, NOTE_SIZE, MUTED)
        elif s.key == 'K3':
            top = y + 12
            self.put_tex(r'$H\,=$', ex, top + (CELL_H + 2))
            right = self.matrix('K3', ex + 36, top, mouse, 2, 4, {}, ['y_x', 'y_y'],
                                ['p_x', 'p_y', 'v_x', 'v_y'])
            self.put_tex(r'$y_t = H\,x_t + w_t$', right, top + 10, NOTE_SIZE, MUTED)
            self.put_tex(r'$w_t \sim \mathcal{N}(0, R)$', right, top + 30, NOTE_SIZE, MUTED)
        else:
            for i, e in enumerate(s.eqs):
                self.equation(e, ex, y + 11 + i * ROW_H, mouse)
            if s.key == 'K6':
                self.put_tex(r'skip the correction if $d^2 > 9.21$', ex, y + 11 + ROW_H, NOTE_SIZE, MUTED)
        if self.model.status[s.key][0] == 'error':
            msg = self.model.status[s.key][1]
            lab = self.small.render(msg, True, PRED)
            while lab.get_width() > PX + PW - 12 - x0 and len(msg) > 10:
                msg = msg[:-2] + '…'
                lab = self.small.render(msg, True, PRED)
            scr.blit(lab, (x0, y + h - 13))
        return y + h

    def draw_live(self, x0, y):
        scr = self.screen
        pygame.draw.line(scr, LINE, (x0, y), (PX + PW - 12, y), 1)
        y += 5
        f = self.run.frames[self.cur]
        pred = self.phase == 'predict'
        mu = f.mu_pred if pred else f.mu
        Sg = f.S_pred if pred else f.Sigma

        def vec(v):
            return ('(' + ', '.join(f'{a:5.1f}' for a in v[:2]) + ' |'
                    + ', '.join(f'{a:5.1f}' for a in v[2:]) + ')')
        lines = []
        if mu is not None:
            lines.append((('μ⁻' if pred else 'μ ') + f' = {vec(mu[:, 0])}', PRED if pred else POST))
            lines.append((f'σ  = {vec(np.sqrt(np.clip(np.diag(Sg), 0, None)))}', MUTED))
        else:
            lines.append(('no filter yet:', MUTED))
            lines.append(('estimate = latest cross', MUTED))
        if pred:
            lines.append(('y not seen yet: → corrects', MUTED))
        elif f.d2 is not None:
            lines.append((f'r = ({f.r[0, 0]:.1f}, {f.r[1, 0]:.1f})  d = {math.sqrt(max(f.d2, 0)):.2f}'
                          + ('  GATED' if f.gated else ''), PRED if f.gated else AMBER))
        elif f.y is None and mu is not None:
            lines.append(('tunnel: no measurement', MUTED))
        for i, (t, col) in enumerate(lines):
            scr.blit(self.small.render(t, True, col), (x0, y + i * 16))
        # gain K, 4 x 2
        gx, gy = PX + PW - 12 - 30 - 2 * 44, y
        scr.blit(self.tex.get(r'$K$', EQ_SIZE), (gx - 20, gy + 22))
        K = None if pred else f.K
        kmax = np.abs(K).max() if K is not None else 1.0
        for i in range(4):
            for j in range(2):
                r = pygame.Rect(gx + j * 44, gy + i * 15, 42, 14)
                if K is not None:
                    v = K[i, j]
                    t = min(abs(v) / max(kmax, 1e-9), 1.0)
                    col = tuple(int(255 - t * (255 - c) * 0.6) for c in (POST if v >= 0 else PRED))
                    pygame.draw.rect(scr, col, r)
                    lab = self.small.render(f'{v:5.2f}', True, FG)
                    scr.blit(lab, lab.get_rect(center=r.center))
                else:
                    pygame.draw.rect(scr, TRACK, r)
            scr.blit(self.small.render(['p_x', 'p_y', 'v_x', 'v_y'][i], True, FAINT), (gx + 92, gy + i * 15))

    def draw_sliders(self, x0, y):
        scr = self.screen
        pygame.draw.line(scr, LINE, (x0, y - 5), (PX + PW - 12, y - 5), 1)
        for i, (name, (lab, lo, hi, log, unit)) in enumerate(SLIDERS.items()):
            yy = y + i * 20
            scr.blit(self.small.render(lab, True, FG), (x0, yy + 2))
            r = pygame.Rect(x0 + 250, yy + 7, 96, 5)
            v = self.get_param(name)
            t = math.log(v / lo) / math.log(hi / lo) if log else (v - lo) / (hi - lo)
            pygame.draw.rect(scr, TRACK, r, border_radius=3)
            pygame.draw.rect(scr, ACCENT, (r.x, r.y, int(t * r.w), r.h), border_radius=3)
            pygame.draw.circle(scr, ACCENT, (r.x + int(t * r.w), r.centery), 6)
            self.hits.append((r.inflate(12, 14), 'slider', name))
            val = f'{v:.2g}×' if unit == '×' else f'{v:.2g} {unit}'
            scr.blit(self.small.render(val, True, FG), (r.right + 10, yy + 2))
            if name == 'sigma_R':
                lr = self.small.render('L locked' if self.lock else 'L lock', True,
                                       ACCENT if self.lock else FAINT)
                rr = lr.get_rect(topright=(PX + PW - 12, yy + 2))
                scr.blit(lr, rr)
                self.hits.append((rr.inflate(6, 4), 'lock', None))

    def draw_popup(self):
        slot, idx, anchor = self.popup
        cols, w, h = 4, 52, 32
        rows = math.ceil(len(L.TOKENS) / cols)
        pw, ph = cols * w + 12, rows * h + 30
        x = min(anchor.x, PX + PW - pw - 6)
        y = anchor.bottom + 4
        if y + ph > WIN_H - 4:
            y = anchor.top - ph - 4
        box = pygame.Rect(x, y, pw, ph)
        shadow = box.move(2, 3)
        pygame.draw.rect(self.screen, LINE, shadow, border_radius=6)
        pygame.draw.rect(self.screen, PANE_BG, box, border_radius=6)
        pygame.draw.rect(self.screen, ACCENT, box, 1, border_radius=6)
        self.screen.blit(self.small.render('choose a token (∅ = nothing)', True, MUTED), (x + 6, y + 5))
        mouse = pygame.mouse.get_pos()
        cur = self.blanks[slot][idx]
        for i, tok in enumerate(L.TOKENS):
            r = pygame.Rect(x + 6 + (i % cols) * w, y + 24 + (i // cols) * h, w - 4, h - 4)
            sel = tok == cur
            hov = r.collidepoint(mouse)
            pygame.draw.rect(self.screen, ACCENT_BG if (hov or sel) else (248, 250, 252), r, border_radius=4)
            pygame.draw.rect(self.screen, ACCENT if sel else LINE, r, 1, border_radius=4)
            s = self.tex.get(tex_of(tok), EQ_SIZE + 1)
            self.screen.blit(s, s.get_rect(center=r.center))
            self.hits.append((r, 'token', tok))
        self.hits.append((box, 'popup', None))

    # -------------------------------------------------------- loop

    def poll_answers(self):
        if time.time() - self.released_t > 1.5:
            self.released_t = time.time()
            new = L.released_answers()
            for key in new.keys() - self.released.keys():
                self.say(f'answer for {key} released: hover the step and press A')
            self.released = new

    def tick(self, dt):
        """Handle events, advance the clock, draw. Returns False to quit."""
        running = True
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                running = self.on_key(e) and running
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3):
                self.on_click(e.pos, e.button)
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                self.drag = None
            elif e.type == pygame.MOUSEMOTION and self.drag:
                self.drag_to(e.pos)
            elif e.type == pygame.MOUSEWHEEL:
                self.on_wheel(pygame.mouse.get_pos(), e.y)
        if self.dirty:
            self.recompute()
        if self.playing:
            self.acc += dt * self.speed
            while self.acc >= L.DT and self.playing:
                self.acc -= L.DT
                self.advance()
        self.poll_answers()
        self.screen.fill(BG)
        self.draw_title()
        self.draw_world()
        self.draw_strip()
        self.draw_bar()
        self.draw_pane()
        pygame.display.flip()
        return running

    def loop(self):
        while self.tick(self.clock.tick(60) / 1000.0):
            pass
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
