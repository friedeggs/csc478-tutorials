"""IMU demo for Tutorial 8 (2026-11-05, Inertial Measurement Units).

Track a head with a simulated 1000 Hz IMU, the way the Oculus Rift did
(LaValle et al., "Head Tracking for the Oculus Rift", ICRA 2014). Build each
filter line from magnets on the right; every stage has a Goal / Do / See /
Think card and unlocks the next one when its line is right.

Tab 1 Flatland (slides 27-33): a head tilting side to side. M1 integrates the
gyro, M2 reads the angle off gravity, M3 blends them (complementary filter).
Tab 2 Rift: a headset in 3D and what the eye sees. M4 integrates the gyro on
rotations, M5 levels the horizon with gravity, M6 predicts over the latency.

Mouse
    drag a magnet       from the palette into a slot; out of a slot to remove it
    right-click         remove a magnet
    tab 1 view          drag: tilt the head yourself · right-drag: shake it sideways
    tab 2 eye view      drag: turn the head yourself · wheel: tilt it · right-drag: shake it
    tab 2 3D view       drag: orbit · wheel: zoom

Keys
    1 / 2           Flatland / Rift
    Space           pause                       .  fast-forward x10
    R               restart the recording       H  recorded head / your head
    B / N           gyro bias / noise on or off K  calibrate the gyro (hold still 2 s) / undo
    [ ]             alpha                       S  reject shakes
    M / G           magnetometer / a magnet nearby (tab 2)
    L / Shift+L     latency up / down (M6)      O  prediction on / off (M6)
    P               position by double integration (tab 2)
    C               reset the 3D camera         A  fill the released answer for this stage
    Backspace       remove the last magnet of the slot under the mouse (Delete: the whole line)
    Esc             quit

Run:  python 2026-11-05-imu_demo.py     (--answers fills every released answer; --stage M5 jumps)
"""
import argparse
import io
import math
import time

import numpy as np
import pygame

import imu_lib as L

# ---------------------------------------------------------------- layout and colours

WIN_W, WIN_H = 1260, 720
PAD = 12
VIEW = pygame.Rect(PAD, 40, 736, 400)
V3D = pygame.Rect(PAD, 40, 380, 400)
VVR = pygame.Rect(PAD + 388, 40, 348, 400)
STRIP = pygame.Rect(PAD, 448, 736, 196)
MODE = pygame.Rect(PAD, 650, 736, 58)
SB = pygame.Rect(760, 40, 488, 668)

BG = (248, 250, 252)          # #f8fafc
FG = (15, 23, 42)             # #0f172a
MUTED = (100, 116, 139)
FAINT = (148, 163, 184)
TRACK = (226, 232, 240)
LINE = (203, 213, 225)
ACCENT = (37, 99, 235)        # #2563eb
PANE_BG = (255, 255, 255)
VIEW_BG = (241, 245, 249)
GOOD = (22, 163, 74)
GOOD_BG = (220, 252, 231)
BAD = (220, 38, 38)
AMBER = (217, 119, 6)
AMBER_BG = (254, 243, 199)
WHITE = (255, 255, 255)
GLOW = (250, 204, 21)
SOLID = (139, 92, 246)        # success violet, as in Tutorial 6

COL_TRUE = (148, 163, 184)    # the truth: grey ghost
COL_GYRO = (37, 99, 235)      # gyro only, blue (slide 33)
COL_ACC = (245, 158, 11)      # accelerometer, orange
COL_FUSED = (220, 38, 38)     # fused, red
COL_UP = (22, 163, 74)        # gravity / up, green
COL_AXIS = (124, 58, 237)     # tilt axis, violet
SKY = (224, 236, 252)
GROUND = (226, 232, 222)
CONFETTI = [COL_GYRO, COL_ACC, COL_FUSED, COL_UP, COL_AXIS, GLOW]

MONO = 'menlo,consolas,dejavusansmono,couriernew,monospace'
MAG_H = 26
POP_SECS, POP_STAGGER = 0.32, 0.05
STRIP_SECS = 20
DECIM = 10                    # samples per strip point (100 Hz)
RING = STRIP_SECS * L.RATE // DECIM
PX_PER_M = 600                # Flatland drawing scale
SHAKE_GAIN = 4                # the shakes are drawn 4x larger than they are

GOAL_M3 = 1.5                 # deg RMS of the fused angle over the last 20 s
GOAL_M5 = 1.0                 # deg RMS tilt error over the last 10 s
PAPER_20MS = (1.46302, 4.77900, 0.19395, 0.71637)   # [LaValle Fig. 9 table]: none avg/worst, constant rate avg/worst


# ---------------------------------------------------------------- helpers

class TexCache:
    """Math labels rendered once with matplotlib's mathtext (pygame fonts lack ω̃, ᵀ, Δ)."""

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


def lighter(c, f=0.55):
    return tuple(int(v + (255 - v) * f) for v in c)


def dashed_rect(surf, color, rect, dash=4):
    r = pygame.Rect(rect)
    for x in range(r.left, r.right, dash * 2):
        pygame.draw.line(surf, color, (x, r.top), (min(x + dash, r.right), r.top))
        pygame.draw.line(surf, color, (x, r.bottom - 1), (min(x + dash, r.right), r.bottom - 1))
    for y in range(r.top, r.bottom, dash * 2):
        pygame.draw.line(surf, color, (r.left, y), (r.left, min(y + dash, r.bottom)))
        pygame.draw.line(surf, color, (r.right - 1, y), (r.right - 1, min(y + dash, r.bottom)))


def dashed_line(surf, color, a, b, dash=6, width=1):
    ax, ay = a
    bx, by = b
    n = max(1, int(math.hypot(bx - ax, by - ay) / dash))
    for i in range(0, n, 2):
        p = (ax + (bx - ax) * i / n, ay + (by - ay) * i / n)
        q = (ax + (bx - ax) * min(n, i + 1) / n, ay + (by - ay) * min(n, i + 1) / n)
        pygame.draw.line(surf, color, p, q, width)


def arrow2d(surf, color, a, b, width=3, head=10):
    pygame.draw.line(surf, color, a, b, width)
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    if n < 2:
        return
    ux, uy = dx / n, dy / n
    p1 = (b[0] - ux * head - uy * head * 0.5, b[1] - uy * head + ux * head * 0.5)
    p2 = (b[0] - ux * head + uy * head * 0.5, b[1] - uy * head - ux * head * 0.5)
    pygame.draw.polygon(surf, color, [b, p1, p2])


class Ring:
    """The last STRIP_SECS of decimated samples, one row per strip point."""

    def __init__(self, cols):
        self.a = np.full((RING, cols), np.nan)
        self.i = 0

    def clear(self):
        self.a[:] = np.nan
        self.i = 0

    def add(self, row):
        self.a[self.i % RING] = row
        self.i += 1

    def rows(self, secs=STRIP_SECS):
        n = min(self.i, RING, int(secs * L.RATE / DECIM))
        if n == 0:
            return self.a[:0]
        idx = (np.arange(self.i - n, self.i)) % RING
        return self.a[idx]

    def full(self, secs):
        return self.i >= secs * L.RATE / DECIM


def rms(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if len(x) else math.nan


# ---------------------------------------------------------------- the app

class App:
    def __init__(self, args):
        pygame.init()
        pygame.display.set_caption('Tutorial 8 — Inertial Measurement Units')
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.font = pygame.font.SysFont(MONO, 14)
        self.bold = pygame.font.SysFont(MONO, 14, bold=True)
        self.small = pygame.font.SysFont(MONO, 12)
        self.big = pygame.font.SysFont(MONO, 17, bold=True)
        self.tex = TexCache()
        self.clock = pygame.time.Clock()

        # stages
        self.slots = {s.key: [[] for _ in range(s.n_slots)] for s in L.STAGES}
        self.expr = {s.key: None for s in L.STAGES}
        self.err = {s.key: None for s in L.STAGES}
        self.checked = {s.key: None for s in L.STAGES}
        self.hint = {s.key: None for s in L.STAGES}
        self.done = set()
        self.released = L.released_answers()
        self.released_t = time.time()
        self.sel = {1: 'M0', 2: 'M4'}
        self.tab = 1
        self.pops = {}
        self.particles = []
        self.banner = None
        self.celebrated = set()
        self.seen = set()                      # stages opened once (first-visit settings)

        # simulation, tab 1
        self.rec1 = L.flat_recording()
        self.imu1 = L.IMU(1, seed=args.seed)
        self.trk1 = L.FlatTracker()
        self.live1 = L.LiveFlat()
        self.ring1 = Ring(10)   # t, truth, gyro, acc, fused, w_meas, w_true, ax, ay, rejected
        self.alpha1 = 0.98
        self.reject1 = False
        # simulation, tab 2
        self.rec2 = L.rift_recording()
        self.imu2 = L.IMU(3, seed=args.seed + 1)
        self.trk2 = L.RiftTracker()
        self.live2 = L.LiveRift()
        self.ring2 = Ring(8)    # t, tilt_g, yaw_g, tilt_f, yaw_f, pred_none, pred_yes, rejected
        self.alpha2 = 0.995
        self.reject2 = True
        self.use_mag = False
        self.magnet = False
        self.latency_ms = 40
        self.predict_on = True
        self.pos_demo = None
        self.cam = [35.0, 20.0, 0.95]          # 3D view: yaw, pitch (deg), distance (m)
        self.st = {1: self.fresh_state(), 2: self.fresh_state()}
        self.live_mode = {1: False, 2: False}
        self.paused = False
        self.speed = 1

        self.msg, self.msg_col, self.msg_t = '', FG, 0.0
        self.drag = None
        self.slider_drag = None
        self.view_drag = None
        self.mouse = (0, 0)
        self.hit = {}
        self.last_frame = time.time()

        if args.answers or args.stage:
            for key, toks in self.released.items():
                if args.stage and L.STAGE_ORDER.index(key) >= L.STAGE_ORDER.index(args.stage):
                    continue
                self.slots[key] = [list(s) for s in toks]
            self.done.add('M0')
        for s in L.STAGES[1:]:
            self.recompile(s.key, quiet=True)
        if args.stage:
            need = [k for k in L.STAGE_ORDER[1:L.STAGE_ORDER.index(args.stage)] if self.checked[k] is not True]
            if need:
                print(f'--stage {args.stage}: no released answer for {", ".join(need)}')
            else:
                st = L.STAGE[args.stage]
                self.sel[st.tab] = args.stage
                self.tab = st.tab
                if st.tab == 2:
                    self.sel[1] = 'M3'
        elif args.answers:
            last = max((k for k in L.STAGE_ORDER if self.unlocked(k)), key=L.STAGE_ORDER.index)
            self.sel[L.STAGE[last].tab] = last
            self.tab = L.STAGE[last].tab
            if self.tab == 2:
                self.sel[1] = 'M3'
        self.enter_stage(self.sel[self.tab])
        self.restart(1)
        self.restart(2)
        self.say('Read the card on the right. The grey head is the truth; the sensors only see its motion.', MUTED)

    @staticmethod
    def fresh_state():
        return dict(k=0, t_abs=0.0, b_hat=None, calib=None, last=None)

    # ------------------------------------------------------------ stages

    def stage(self):
        return L.STAGE[self.sel[self.tab]]

    def unlocked(self, key):
        i = L.STAGE_ORDER.index(key)
        if i == 0:
            return True
        prev = L.STAGE_ORDER[i - 1]
        return prev in self.done

    def recompile(self, key, quiet=False):
        st = L.STAGE[key]
        e, err = L.compile_line(st, self.slots[key])
        self.expr[key], self.err[key] = e, err
        ok, hint = L.check(st, self.slots[key])
        was = self.checked[key]
        self.checked[key], self.hint[key] = ok, hint
        if ok:
            self.done.add(key)
        self.celebrated.discard(key)
        if quiet:
            return
        if ok and was is not True:
            self.pops[key] = time.time()
            nxt = L.STAGE_ORDER.index(key) + 1
            more = f' — {L.STAGE_ORDER[nxt]} is open' if nxt < len(L.STAGE_ORDER) else ''
            self.say(f'{key} ✓ correct{more}. Watch what it does before moving on.', GOOD)
            self.sparks_at_line()
        elif ok is False:
            self.say(f'{key} ✗ not yet — {hint}' if hint else f'{key} runs, but not right yet: watch the view', AMBER)
        # the line changed: start that tab's estimates again from the truth
        self.reset_trackers(st.tab)

    def enter_stage(self, key):
        st = L.STAGE[key]
        self.sel[st.tab] = key
        self.tab = st.tab
        if key in self.seen:
            return
        self.seen.add(key)
        if key == 'M4':                        # the order lesson is cleanest without bias and noise
            self.imu2.bias_on = self.imu2.noise_on = False
            self.say('M4 starts with the gyro bias and noise off (B, N): only the order matters here.', MUTED)
        elif key == 'M5':
            self.imu2.bias_on = self.imu2.noise_on = True
            self.restart(2)
            self.say('M5: bias and noise are back on. Watch the blue (gyro only) estimate drift first.', MUTED)
        elif key == 'M6':
            self.restart(2)

    def fill_answer(self):
        key = self.sel[self.tab]
        if key == 'M0':
            return
        if key not in self.released:
            self.say(f'no released answer for {key} yet', AMBER)
            return
        self.slots[key] = [list(s) for s in self.released[key]]
        self.recompile(key)
        self.say(f'{key} filled with the released answer', ACCENT)

    def poll_answers(self):
        self.released_t = time.time()
        new = L.released_answers()
        fresh = sorted(set(new) - set(self.released))
        self.released = new
        if fresh:
            self.say(f'answer for {", ".join(fresh)} released: open the stage and press A', ACCENT)

    # ------------------------------------------------------------ simulation

    def restart(self, tab):
        self.st[tab].update(k=0, t_abs=0.0, calib=None, last=None)
        if tab == 1:
            self.trk1.reset(0.0)
            self.ring1.clear()
            if self.live_mode[1]:
                self.trk1.reset(self.live1.ang[0])
        else:
            R0 = self.live2.R if self.live_mode[2] else self.rec2.R[0]
            self.trk2.reset(R0)
            self.ring2.clear()
            self.pos_demo = None
        self.celebrated = {k for k in self.celebrated if L.STAGE[k].tab != tab}

    def reset_trackers(self, tab):
        """Estimates start again from the current truth (after a line changed)."""
        last = self.st[tab]['last']
        if tab == 1:
            self.trk1.reset(last['theta'] if last else 0.0)
            self.ring1.clear()
        else:
            self.trk2.reset(last['R'] if last else np.eye(3))
            self.ring2.clear()
            self.pos_demo = None

    def truth_samples(self, tab, n):
        st = self.st[tab]
        out = []
        if tab == 1:
            if self.live_mode[1]:
                return self.live1.samples(n)
            r = self.rec1
            for j in range(n):
                i = (st['k'] + j) % len(r.t)
                out.append((r.theta[i], r.omega[i], r.acc[i], r.pivot[i]))
            return out
        if self.live_mode[2]:
            return self.live2.samples(n)
        r = self.rec2
        for j in range(n):
            i = (st['k'] + j) % len(r.t)
            out.append((r.R[i], r.omega[i], r.acc[i], r.pos[i]))
        return out

    def calib_step(self, tab, w_meas):
        st = self.st[tab]
        c = st['calib']
        if c is None:
            return
        t = st['t_abs'] if not self.live_mode[tab] else time.time()
        if t >= c['t0']:
            c['ws'].append(w_meas)
        if t >= c['t1']:
            ws = np.array(c['ws'])
            st['b_hat'] = ws.mean(0) if len(ws) else None
            st['calib'] = None
            if st['b_hat'] is not None:
                spread = np.degrees(ws.std(0).max())
                b = np.degrees(np.linalg.norm(st['b_hat']))
                if spread > 10:
                    self.say(f'calibrated, but the head moved (spread {spread:.0f}°/s): the bias estimate is off. '
                             'K twice to try again.', AMBER)
                else:
                    self.say(f'calibrated: bias {b:.2f}°/s measured while still, now subtracted from ω̃', GOOD)

    def start_calibration(self, tab):
        st = self.st[tab]
        if st['b_hat'] is not None or st['calib'] is not None:
            st['b_hat'] = None
            st['calib'] = None
            self.say('calibration off: the raw gyro readings again', MUTED)
            return
        if self.live_mode[tab]:
            now = time.time()
            st['calib'] = dict(t0=now, t1=now + 2.0, ws=[])
            self.say('calibrating: hold your head still for 2 s…', ACCENT)
        else:
            self.restart(tab)
            t1 = 2.5 if tab == 1 else 1.8
            st['calib'] = dict(t0=0.5, t1=t1, ws=[])
            self.say('calibrating: the recording restarts, and the gyro is averaged while the head holds still…',
                     ACCENT)

    def simulate(self, tab, n):
        if n <= 0:
            return
        if tab == 1:
            self.simulate1(n)
        else:
            self.simulate2(n)

    def simulate1(self, n):
        st, imu, trk = self.st[1], self.imu1, self.trk1
        m1, m2, m3 = self.expr['M1'], self.expr['M2'], self.expr['M3']
        for th, w, acc, piv in self.truth_samples(1, n):
            wm = imu.gyro(w, st['t_abs'])[0]
            am = imu.accel_flat(th, acc)
            self.calib_step(1, np.array([wm]))
            wc = wm - (st['b_hat'][0] if st['b_hat'] is not None else 0.0)
            trk.step(wc, am, m1, m2, m3, self.alpha1, self.reject1)
            st['k'] += 1
            st['t_abs'] += L.DT
            if st['k'] % DECIM == 0:
                d = math.degrees
                self.ring1.add([st['t_abs'], d(th), d(L.wrap(trk.th_g)), d(L.wrap(trk.th_a)), d(L.wrap(trk.th_f)),
                                d(wm), d(w), am[0], am[1], float(trk.rejected)])
        st['last'] = dict(theta=th, omega=w, acc=acc, pivot=piv, w_meas=wm, a_meas=am)

    def simulate2(self, n):
        st, imu, trk = self.st[2], self.imu2, self.trk2
        m4, m5, m6 = self.expr['M4'], self.expr['M5'], self.expr['M6']
        field = L.magnet_nearby_field() if self.magnet else L.M_WORLD
        T = self.latency_ms / 1000
        back = self.latency_ms
        for R, w, acc, pos in self.truth_samples(2, n):
            wm = imu.gyro(w, st['t_abs'])
            am = imu.accel(R, acc)
            mm = imu.magnet(R, field)
            self.calib_step(2, wm)
            wc = wm - (st['b_hat'] if st['b_hat'] is not None else 0.0)
            trk.step(wc, am, mm, m4, m5, self.alpha2, self.reject2, self.use_mag)
            st['k'] += 1
            st['t_abs'] += L.DT
            if self.pos_demo is not None:
                pd = self.pos_demo
                a_lin = trk.R_f @ am - L.G * L.UP
                pd['v'] = pd['v'] + a_lin * L.DT
                pd['p'] = pd['p'] + pd['v'] * L.DT
                pd['true'] = pos
                pd['t'] += L.DT
            if st['k'] % DECIM == 0:
                pn = py = math.nan
                past = trk.past(back)
                if past is not None:
                    pn = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(past.T @ trk.R_f) - 1) / 2))))
                    if m6 is not None:
                        Rp = trk.predicted(m6, T, back)
                        py = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rp.T @ trk.R_f) - 1) / 2))))
                if m4 is None:                 # nothing tracks yet: nothing to plot
                    self.ring2.add([st['t_abs']] + [math.nan] * 7)
                else:
                    self.ring2.add([st['t_abs'], L.tilt_error(trk.R_g, R), L.yaw_error(trk.R_g, R),
                                    L.tilt_error(trk.R_f, R), L.yaw_error(trk.R_f, R), pn, py,
                                    float(trk.rejected)])
        st['last'] = dict(R=R, omega=w, acc=acc, pos=pos, w_meas=wm, a_meas=am, m_meas=mm)

    def update(self):
        now = time.time()
        dt = min(0.1, now - self.last_frame)
        self.last_frame = now
        if now - self.released_t > 2.0:
            self.poll_answers()
        speed = 1 if self.live_mode[self.tab] else self.speed
        n = 0 if self.paused else int(round(dt * L.RATE * speed))
        self.simulate(self.tab, n)
        self.check_goals()
        self.update_particles(dt)

    # ------------------------------------------------------------ goals and celebrations

    def check_goals(self):
        key = self.sel[self.tab]
        if key in self.celebrated or self.checked.get(key) is not True:
            return
        if key == 'M3' and self.ring1.full(STRIP_SECS):
            e = rms(self.ring1.rows()[:, 4] - self.ring1.rows()[:, 1])
            if e < GOAL_M3:
                self.celebrate('M3', f'✓ fused angle within {e:.2f}° RMS over 20 s: no drift, little noise')
        elif key == 'M5' and self.ring2.full(10):
            e = rms(self.ring2.rows(10)[:, 3])
            if e < GOAL_M5:
                self.celebrate('M5', f'✓ horizon level: tilt error {e:.2f}° RMS over 10 s')
        elif key == 'M6' and self.predict_on and self.latency_ms >= 20 and self.ring2.full(STRIP_SECS):
            rows = self.ring2.rows()
            a0, a1 = np.nanmean(rows[:, 5]), np.nanmean(rows[:, 6])
            if a1 < a0 / 3:
                self.celebrate('M6', f'✓ prediction: {a1:.2f}° instead of {a0:.2f}° at {self.latency_ms} ms')

    def celebrate(self, key, text):
        self.celebrated.add(key)
        self.banner = dict(text=text, t0=time.time())
        rng = np.random.default_rng()
        area = VIEW
        for _ in range(110):
            self.particles.append(dict(
                kind='confetti', x=area.centerx + rng.normal(0, 90), y=area.y + area.h * 0.55,
                vx=rng.normal(0, 170), vy=-rng.uniform(280, 560), life=rng.uniform(1.6, 2.6),
                col=CONFETTI[rng.integers(len(CONFETTI))], w=rng.uniform(4, 7), h=rng.uniform(8, 12),
                spin=rng.uniform(-9, 9), a=rng.uniform(0, 6.3)))
        for p in self.particles:
            p.setdefault('max', p['life'])

    def sparks_at_line(self):
        r = self.hit.get('line_rect')
        if r is None:
            return
        rng = np.random.default_rng()
        for k in range(18):
            ang = 2 * math.pi * k / 18
            v = rng.uniform(110, 210)
            life = rng.uniform(0.45, 0.7)
            self.particles.append(dict(kind='spark', x=r.centerx, y=r.centery, vx=v * math.cos(ang),
                                       vy=v * math.sin(ang), life=life, max=life,
                                       col=GOOD if k % 2 else GLOW, r=rng.uniform(2.5, 4)))

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
        for p in self.particles:
            if p['kind'] == 'confetti':
                c, s_ = math.cos(p['a']), math.sin(p['a'])
                w, h = p['w'] / 2, p['h'] / 2 * abs(math.cos(p['a'] * 0.7))
                pts = [(p['x'] + dx * c - dy * s_, p['y'] + dx * s_ + dy * c)
                       for dx, dy in ((-w, -h), (w, -h), (w, h), (-w, h))]
                pygame.draw.polygon(scr, p['col'], pts)
            else:
                r = p['r'] * (p['life'] / p['max'])
                pygame.draw.circle(scr, p['col'], (int(p['x']), int(p['y'])), max(1, int(r)))

    def draw_banner(self):
        b = self.banner
        if not b:
            return
        age = time.time() - b['t0']
        if age > 5:
            self.banner = None
            return
        t = self.bold.render(b['text'], True, WHITE)
        r = pygame.Rect(0, 0, t.get_width() + 28, 34)
        r.midbottom = (VIEW.centerx, VIEW.bottom - 10)
        pygame.draw.rect(self.screen, SOLID, r, border_radius=8)
        self.screen.blit(t, (r.x + 14, r.y + 9))

    def pop_scale(self, key, ordinal):
        t0 = self.pops.get(key)
        if t0 is None:
            return 1.0
        t = time.time() - t0 - POP_STAGGER * ordinal
        if 0 <= t < POP_SECS:
            return 1 + 0.32 * math.sin(math.pi * t / POP_SECS)
        return 1.0

    # ------------------------------------------------------------ drawing: tab 1

    def head_pivot(self, pivot_x):
        return (VIEW.centerx + pivot_x * PX_PER_M * SHAKE_GAIN, VIEW.y + 330)

    def draw_flatland(self):
        scr = self.screen
        pygame.draw.rect(scr, VIEW_BG, VIEW)
        last = self.st[1]['last']
        if last is None:
            return
        key = self.sel[1]
        px, py = self.head_pivot(last['pivot'])
        th = last['theta']
        neck = L.NECK * PX_PER_M
        cx, cy = px - neck * math.sin(th), py - neck * math.cos(th)

        # protractor around the head's centre
        R0 = 150
        for deg in range(-90, 91, 10):
            a = math.radians(deg)
            r1 = R0 + (10 if deg % 30 == 0 else 5)
            p = (VIEW.centerx - R0 * math.sin(a), VIEW.y + 330 - neck - R0 * math.cos(a))
            q = (VIEW.centerx - r1 * math.sin(a), VIEW.y + 330 - neck - r1 * math.cos(a))
            pygame.draw.line(scr, LINE, p, q, 2 if deg % 30 == 0 else 1)
            if deg % 30 == 0:
                t = self.small.render(f'{deg}°', True, FAINT)
                o = (VIEW.centerx - (r1 + 16) * math.sin(a), VIEW.y + 330 - neck - (r1 + 16) * math.cos(a))
                scr.blit(t, (o[0] - t.get_width() / 2, o[1] - t.get_height() / 2))

        # body and head (the truth)
        pygame.draw.polygon(scr, (203, 213, 225), [(px - 120, VIEW.bottom), (px - 90, py + 20), (px + 90, py + 20),
                                                   (px + 120, VIEW.bottom)])
        pygame.draw.line(scr, (148, 163, 184), (px, py + 20), (cx, cy), 18)
        self.draw_head(cx, cy, th)

        # needles: estimates from the head's centre
        hc = (VIEW.centerx, VIEW.y + 330 - neck)
        needles = [('truth', th, COL_TRUE)]
        trk = self.trk1
        if key in ('M1', 'M2', 'M3') and self.expr['M1'] is not None:
            needles.append(('M1 gyro', trk.th_g, COL_GYRO))
        if key in ('M2', 'M3') and self.expr['M2'] is not None:
            needles.append(('M2 accel', trk.th_a, COL_ACC))
        if key == 'M3' and self.expr['M3'] is not None:
            needles.append(('M3 fused', trk.th_f, COL_FUSED))
        for i, (name, a, col) in enumerate(needles):
            a = L.wrap(a)
            tip = (hc[0] - (R0 - 4) * math.sin(a), hc[1] - (R0 - 4) * math.cos(a))
            w = 2 if name == 'truth' else 4
            pygame.draw.line(scr, col, (hc[0] - 100 * math.sin(a), hc[1] - 100 * math.cos(a)), tip, w)
            pygame.draw.circle(scr, col, tip, 5)
            # legend
            t = self.small.render(f'{name:9s} {math.degrees(a):+7.1f}°', True, col)
            scr.blit(t, (VIEW.right - 190, VIEW.y + 120 + 18 * i))

        # the accelerometer arrow, drawn on the head (head frame), scaled 9 px per m/s²
        am = last['a_meas']
        c, s = math.cos(th), math.sin(th)
        wx, wy = am[0] * c - am[1] * s, am[0] * s + am[1] * c   # head frame -> world
        k = 9
        arrow2d(scr, COL_UP, (cx, cy), (cx + wx * k, cy - wy * k), 4, 12)
        lab = self.tex.get(r'$\tilde{a}$', 15, COL_UP)
        scr.blit(lab, (cx + wx * k + 6, cy - wy * k - 18))
        # head axes
        for (bx, by), name in (((c, s), 'x'), ((-s, c), 'y')):
            pygame.draw.line(scr, FAINT, (cx, cy), (cx + bx * 36, cy - by * 36), 1)
            scr.blit(self.small.render(name, True, FAINT), (cx + bx * 42 - 4, cy - by * 42 - 7))

        self.draw_gyro_dial(last)
        r = self.small.render(f'shakes drawn {SHAKE_GAIN}× larger', True, FAINT)
        scr.blit(r, (VIEW.right - r.get_width() - 8, VIEW.bottom - 20))
        if self.trk1.rejected and key == 'M3' and self.reject1:
            t = self.bold.render('accelerometer rejected (shake)', True, COL_ACC)
            scr.blit(t, (VIEW.right - t.get_width() - 10, VIEW.y + 96))

    def draw_head(self, cx, cy, th):
        scr = self.screen
        c, s = math.cos(th), math.sin(th)

        def P(x, y):                           # head frame (px, y up) -> screen
            return (cx + x * c - y * s, cy - (x * s + y * c))

        pygame.draw.circle(scr, (254, 243, 230), (int(cx), int(cy)), 62)
        pygame.draw.circle(scr, (180, 150, 130), (int(cx), int(cy)), 62, 2)
        for ex in (-64, 64):                   # ears
            pygame.draw.circle(scr, (254, 243, 230), P(ex, 0), 12)
            pygame.draw.circle(scr, (180, 150, 130), P(ex, 0), 12, 2)
        pygame.draw.circle(scr, (254, 243, 230), (int(cx), int(cy)), 60)
        visor = [P(-58, 22), P(58, 22), P(58, -10), P(-58, -10)]
        pygame.draw.polygon(scr, (51, 65, 85), visor)
        pygame.draw.line(scr, (30, 41, 59), P(-62, 6), P(62, 6), 4)        # strap
        pygame.draw.circle(scr, (220, 38, 38), P(0, 26), 5)                # the IMU on the visor
        mouth = [P(-18, -34), P(0, -40), P(18, -34)]
        pygame.draw.lines(scr, (180, 150, 130), False, mouth, 3)

    def draw_gyro_dial(self, last):
        scr = self.screen
        cx, cy, r = VIEW.x + 92, VIEW.y + 96, 64
        pygame.draw.circle(scr, PANE_BG, (cx, cy), r + 6)
        pygame.draw.circle(scr, LINE, (cx, cy), r + 6, 1)
        full = 300.0                            # deg/s at ±135°
        for v in range(-300, 301, 100):
            a = math.radians(v / full * 135)
            p = (cx - r * math.sin(a), cy - r * math.cos(a))
            q = (cx - (r - 8) * math.sin(a), cy - (r - 8) * math.cos(a))
            pygame.draw.line(scr, FAINT, p, q, 2)
        for val, col, w in ((math.degrees(last['omega']), COL_TRUE, 2), (math.degrees(last['w_meas']), FG, 3)):
            a = math.radians(max(-full, min(full, val)) / full * 135)
            pygame.draw.line(scr, col, (cx, cy), (cx - (r - 10) * math.sin(a), cy - (r - 10) * math.cos(a)), w)
        pygame.draw.circle(scr, FG, (cx, cy), 4)
        scr.blit(self.tex.get(r'$\tilde{\omega}$ gyro', 12, FG), (cx - 26, cy + 14))
        t = self.small.render(f'{math.degrees(last["w_meas"]):+6.1f}°/s', True, FG)
        scr.blit(t, (cx - t.get_width() // 2, cy + r + 10))

    # ------------------------------------------------------------ drawing: tab 2

    def cam_rot(self):
        yaw, pitch = math.radians(self.cam[0]), math.radians(self.cam[1])
        back = np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
        right = np.cross(L.UP, back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        return np.array([right, up, back])

    def proj3(self, X, Rc):
        c = Rc @ np.asarray(X)
        depth = self.cam[2] - c[2]
        f = 620
        return (V3D.centerx + f * c[0] / depth, V3D.centery + 30 - f * c[1] / depth), depth

    def draw_box(self, R, color, Rc, solid=True):
        hx, hy, hz = 0.095, 0.055, 0.065
        corners = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        W = corners @ R.T
        P = [self.proj3(w, Rc) for w in W]
        faces = [((0, 1, 3, 2), (-1, 0, 0)), ((4, 5, 7, 6), (1, 0, 0)), ((0, 1, 5, 4), (0, -1, 0)),
                 ((2, 3, 7, 6), (0, 1, 0)), ((0, 2, 6, 4), (0, 0, -1)), ((1, 3, 7, 5), (0, 0, 1))]
        if solid:
            light = np.array([0.3, 0.8, 0.5])
            light /= np.linalg.norm(light)
            back = Rc[2]
            order = []
            for idx, n in faces:
                nw = R @ np.array(n, float)
                if nw @ back <= 0:
                    continue
                depth = np.mean([P[i][1] for i in idx])
                order.append((depth, idx, nw, n))
            order.sort(key=lambda x: -x[0])
            for _, idx, nw, n in order:
                shade = 0.55 + 0.45 * max(0.0, float(nw @ light))
                base = (51, 65, 85) if n == (0, 0, -1) else lighter(color, 0.55)
                col = tuple(int(v * shade) for v in base)
                pygame.draw.polygon(self.screen, col, [P[i][0] for i in idx])
                pygame.draw.polygon(self.screen, color, [P[i][0] for i in idx], 1)
        else:
            for a, b in ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6),
                         (3, 7)):
                pygame.draw.line(self.screen, color, P[a][0], P[b][0], 2)
            # front face marker: where the headset looks
            f = (R @ np.array([0, 0, -hz - 0.04]))
            pygame.draw.line(self.screen, color, self.proj3(R @ np.array([0, 0, -hz]), Rc)[0],
                             self.proj3(f, Rc)[0], 2)

    def arrow3(self, a, b, color, Rc, width=3, label=None, dashed=False, left=False):
        pa, pb = self.proj3(a, Rc)[0], self.proj3(b, Rc)[0]
        if dashed:
            dashed_line(self.screen, color, pa, pb, 6, width)
        else:
            arrow2d(self.screen, color, pa, pb, width, 10)
        if label:
            s = self.tex.get(label, 14, color)
            self.screen.blit(s, (pb[0] - s.get_width() - 6 if left else pb[0] + 6, pb[1] - 16))

    def estimate_R(self):
        """The orientation shown for the selected stage, and its colour."""
        key = self.sel[2]
        if key == 'M4' or self.expr['M5'] is None:
            return self.trk2.R_g, COL_GYRO
        return self.trk2.R_f, COL_FUSED

    def draw_3d(self):
        scr = self.screen
        pygame.draw.rect(scr, VIEW_BG, V3D)
        last = self.st[2]['last']
        if last is None:
            return
        clip = scr.get_clip()
        scr.set_clip(V3D)
        Rc = self.cam_rot()
        key = self.sel[2]
        # floor grid
        for v in np.arange(-0.4, 0.41, 0.1):
            pygame.draw.line(scr, LINE, self.proj3((v, -0.3, -0.4), Rc)[0], self.proj3((v, -0.3, 0.4), Rc)[0], 1)
            pygame.draw.line(scr, LINE, self.proj3((-0.4, -0.3, v), Rc)[0], self.proj3((0.4, -0.3, v), Rc)[0], 1)
        n = self.proj3((0, -0.3, -0.45), Rc)[0]
        scr.blit(self.small.render('N', True, MUTED), (n[0] - 4, n[1] - 7))
        R_est, col = self.estimate_R()
        if self.expr['M4'] is not None:
            if key in ('M5', 'M6') and self.expr['M5'] is not None:
                self.draw_box(self.trk2.R_g, lighter(COL_GYRO, 0.4), Rc, solid=False)
            self.draw_box(R_est, col, Rc, solid=True)
        self.draw_box(last['R'], COL_TRUE, Rc, solid=False)
        o = np.zeros(3)
        if key in ('M5', 'M6') and self.expr['M4'] is not None:
            a = self.trk2.a_lp
            na = np.linalg.norm(a)
            if na > 1e-6:
                u = a / na
                self.arrow3(o, L.UP * 0.25, COL_UP, Rc, 3, r'$up$', left=True)
                self.arrow3(o, u * 0.25, COL_ACC, Rc, 3, r'$\hat{a} = R\,\tilde{a}$')
                phi = math.acos(max(-1.0, min(1.0, float(u @ L.UP))))
                if phi > 1e-3:
                    ax = np.cross(u, L.UP)
                    ax /= np.linalg.norm(ax)
                    pts = [self.proj3(L.rot(ax, -phi * s) @ L.UP * 0.16, Rc)[0] for s in np.linspace(0, 1, 16)]
                    pygame.draw.lines(scr, FG, False, pts, 2)
                    lab = self.tex.get(rf'$\varphi = {math.degrees(phi):.1f}^\circ$', 13, FG)
                    scr.blit(lab, (pts[8][0] + 6, pts[8][1] - 6))
                    self.arrow3(-ax * 0.18, ax * 0.18, COL_AXIS, Rc, 2, r'$t$', dashed=True)
        if key in ('M4', 'M5', 'M6') and self.expr['M4'] is not None:
            w = R_est @ last['w_meas']
            nw = np.linalg.norm(w)
            if nw > 0.05:
                self.arrow3(o, w / nw * min(0.3, 0.06 + nw * 0.05), COL_GYRO, Rc, 2, r'$\tilde{\omega}$')
        if self.pos_demo is not None:
            pd = self.pos_demo
            off = pd['p'] - pd['true']
            d = np.linalg.norm(off)
            shown = off if d < 0.35 else off / d * 0.35
            p = self.proj3(shown, Rc)[0]
            pygame.draw.circle(scr, BAD, p, 6)
            t = self.small.render(f'position {d:.2f} m off' + (' →' if d >= 0.35 else ''), True, BAD)
            scr.blit(t, (p[0] + 8, p[1] - 6))
        scr.set_clip(clip)
        t = self.small.render('drag: orbit · wheel: zoom · C: reset', True, FAINT)
        scr.blit(t, (V3D.x + 8, V3D.bottom - 18))
        lg = [('truth', COL_TRUE)]
        if self.expr['M4'] is not None:
            lg.append(('gyro only (M4)' if col == COL_GYRO else 'fused (M4 + M5)', col))
            if col == COL_FUSED:
                lg.append(('gyro only', lighter(COL_GYRO, 0.4)))
        for i, (name, c) in enumerate(lg):
            pygame.draw.rect(scr, c, (V3D.x + 10, V3D.y + 12 + 16 * i, 10, 10))
            scr.blit(self.small.render(name, True, MUTED), (V3D.x + 26, V3D.y + 9 + 16 * i))

    def vr_rotations(self):
        """(rotation the picture is drawn with, rotation of the ghost, label)."""
        key = self.sel[2]
        last = self.st[2]['last']
        R_est, _ = self.estimate_R()
        if key != 'M6' or self.expr['M4'] is None:
            return R_est, last['R'], 'ghost: where the world really is'
        back = self.latency_ms
        if self.predict_on and self.expr['M6'] is not None:
            Rd = self.trk2.predicted(self.expr['M6'], back / 1000, back)
        else:
            Rd = self.trk2.past(back)
        if Rd is None:
            Rd = self.trk2.R_f
        return Rd, self.trk2.R_f, 'ghost: your estimate now, without the lag'

    def vr_scene(self):
        if hasattr(self, '_scene'):
            return self._scene
        segs = []
        y0 = -1.6
        for v in range(-8, 9):
            segs.append(((v, y0, -8), (v, y0, 8), 1))
            segs.append(((-8, y0, v), (8, y0, v), 1))
        boxes = [((0, -1.1, -3.5), (0.5, 0.5, 0.5)), ((2.5, -1.3, -2), (0.3, 0.3, 0.3)),
                 ((-3, -0.9, 0.5), (0.4, 0.7, 0.4)), ((1.5, -1.2, 3), (0.4, 0.4, 0.4)),
                 ((-2.2, -1.35, -2.8), (0.25, 0.25, 0.25))]
        for (cx, cy, cz), (hx, hy, hz) in boxes:
            c = [(cx + sx * hx, cy + sy * hy, cz + sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            for a, b in ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6),
                         (3, 7)):
                segs.append((c[a], c[b], 2))
        segs.append(((0, y0, -7), (0, 1.8, -7), 2))                # a pole to the north, with a flag
        segs.append(((0, 1.8, -7), (0.8, 1.6, -7), 2))
        segs.append(((0.8, 1.6, -7), (0, 1.4, -7), 2))
        ring = [(1000 * math.sin(a), 0.0, 1000 * math.cos(a)) for a in np.linspace(0, 2 * math.pi, 73)]
        for a, b in zip(ring[:-1], ring[1:]):
            segs.append((a, b, 3))                                 # the horizon, at eye height
        self._scene = [(np.array(a, float), np.array(b, float), w) for a, b, w in segs]
        return self._scene

    def vr_project(self, Rt, X):
        c = Rt @ X
        f = VVR.w / 2
        return c, (VVR.centerx + f * c[0] / -c[2], VVR.centery - f * c[1] / -c[2])

    def draw_vr_lines(self, R, color, scale_w=1.0):
        Rt = R.T
        near = 0.05
        f = VVR.w / 2
        for a, b, w in self.vr_scene():
            ca, cb = Rt @ a, Rt @ b
            if ca[2] > -near and cb[2] > -near:
                continue
            if ca[2] > -near or cb[2] > -near:
                t = (-near - ca[2]) / (cb[2] - ca[2])
                cm = ca + (cb - ca) * t
                if ca[2] > -near:
                    ca = cm
                else:
                    cb = cm
            pa = (VVR.centerx + f * ca[0] / -ca[2], VVR.centery - f * ca[1] / -ca[2])
            pb = (VVR.centerx + f * cb[0] / -cb[2], VVR.centery - f * cb[1] / -cb[2])
            if max(abs(pa[0]), abs(pa[1]), abs(pb[0]), abs(pb[1])) > 1e5:
                continue
            pygame.draw.line(self.screen, color, pa, pb, max(1, int(w * scale_w)))

    def draw_ground(self, R):
        """The ground plane, clipped at the near plane (so the horizon tilts with the estimate)."""
        Rt = R.T
        near = 0.05
        quad = [np.array(p, float) for p in ((-500, -1.6, -500), (500, -1.6, -500), (500, -1.6, 500),
                                             (-500, -1.6, 500))]
        cs = [Rt @ q for q in quad]
        out = []
        for i in range(4):
            a, b = cs[i], cs[(i + 1) % 4]
            ina, inb = a[2] < -near, b[2] < -near
            if ina:
                out.append(a)
            if ina != inb:
                t = (-near - a[2]) / (b[2] - a[2])
                out.append(a + (b - a) * t)
        if len(out) < 3:
            return
        f = VVR.w / 2
        pts = [(VVR.centerx + f * c[0] / -c[2], VVR.centery - f * c[1] / -c[2]) for c in out]
        pts = [(max(-1e5, min(1e5, x)), max(-1e5, min(1e5, y))) for x, y in pts]
        pygame.draw.polygon(self.screen, GROUND, pts)

    def draw_vr(self):
        scr = self.screen
        pygame.draw.rect(scr, SKY, VVR)
        last = self.st[2]['last']
        if last is None:
            return
        clip = scr.get_clip()
        scr.set_clip(VVR)
        Rd, Rghost, label = self.vr_rotations()
        self.draw_ground(Rd)
        self.draw_vr_lines(Rghost, (186, 196, 210), 1.0)
        self.draw_vr_lines(Rd, (51, 65, 85), 1.0)
        pygame.draw.line(scr, FAINT, (VVR.centerx - 6, VVR.centery), (VVR.centerx + 6, VVR.centery), 1)
        pygame.draw.line(scr, FAINT, (VVR.centerx, VVR.centery - 6), (VVR.centerx, VVR.centery + 6), 1)
        scr.set_clip(clip)
        t = self.bold.render('what the eye sees', True, FG)
        scr.blit(t, (VVR.x + 8, VVR.y + 8))
        scr.blit(self.small.render(label, True, MUTED), (VVR.x + 8, VVR.y + 26))
        if self.sel[2] == 'M6':
            pr = 'prediction on' if (self.predict_on and self.expr['M6'] is not None) else 'no prediction'
            scr.blit(self.small.render(f'latency {self.latency_ms} ms · {pr}', True, ACCENT), (VVR.x + 8, VVR.y + 42))
        scr.blit(self.small.render('drag: turn your head · wheel: tilt', True, FAINT), (VVR.x + 8, VVR.bottom - 18))

    # ------------------------------------------------------------ strips

    def draw_plot(self, rect, t, series, title, unit='°', symmetric=True, floor=2.0):
        scr = self.screen
        pygame.draw.rect(scr, PANE_BG, rect)
        pygame.draw.rect(scr, LINE, rect, 1)
        scr.blit(self.small.render(title, True, MUTED), (rect.x + 48, rect.y + 3))
        if len(t) < 2:
            return
        vals = [np.abs(s[0][np.isfinite(s[0])]) for s in series if len(s) < 4 or s[3]]
        vmax = max([floor] + [float(v.max()) for v in vals if len(v)]) * 1.1
        lo, hi = (-vmax, vmax) if symmetric else (0.0, vmax)
        x0, x1 = rect.x + 44, rect.right - 6
        y0, y1 = rect.y + 18, rect.bottom - 4
        tn = t[-1]

        def Y(v):
            return y1 - (v - lo) / (hi - lo) * (y1 - y0)

        if symmetric:
            pygame.draw.line(scr, TRACK, (x0, Y(0)), (x1, Y(0)))
        for v in (lo, hi) if not symmetric else (lo, 0, hi):
            s = self.small.render(f'{v:.0f}{unit}' if abs(v) >= 10 else f'{v:.1f}{unit}', True, FAINT)
            scr.blit(s, (rect.x + 4, Y(v) - 7))
        xs = x1 - (tn - t) / STRIP_SECS * (x1 - x0)
        for s, col, w, *_ in series:
            ok = np.isfinite(s)
            if ok.sum() < 2:
                continue
            ys = np.clip(Y(s), y0 - 2, y1 + 2)
            pts = np.stack([xs, ys], 1)
            # break the line at gaps
            start = None
            for i in range(len(s) + 1):
                good = i < len(s) and ok[i]
                if good and start is None:
                    start = i
                if not good and start is not None:
                    if i - start >= 2:
                        pygame.draw.lines(scr, col, False, pts[start:i].tolist(), w)
                    start = None

    def draw_strips(self):
        top = pygame.Rect(STRIP.x, STRIP.y, STRIP.w, STRIP.h // 2 - 3)
        bot = pygame.Rect(STRIP.x, STRIP.y + STRIP.h // 2 + 3, STRIP.w, STRIP.h // 2 - 3)
        if self.tab == 1:
            rows = self.ring1.rows()
            t = rows[:, 0]
            key = self.sel[1]
            if key == 'M0':
                self.draw_plot(top, t, [(rows[:, 6], COL_TRUE, 2), (rows[:, 5], FG, 1)],
                               'gyro ω̃ (black) and the true angular velocity (grey), °/s', '', floor=20)
                self.draw_plot(bot, t, [(rows[:, 7], COL_GYRO, 1), (rows[:, 8], COL_UP, 1)],
                               'accelerometer: ã_x (blue) and ã_y (green), m/s²', '', floor=10)
                return
            ser, err = [(rows[:, 1], COL_TRUE, 3)], []
            cols = [('M1', 2, COL_GYRO), ('M2', 3, COL_ACC), ('M3', 4, COL_FUSED)]
            upto = L.STAGE_ORDER.index(key)
            labels = []
            for k, c, col in cols:
                if L.STAGE_ORDER.index(k) <= upto and self.expr[k] is not None:
                    mine = k == key
                    ser.append((rows[:, c], col, 1 if k == 'M2' else 2, mine))
                    e = rows[:, c] - rows[:, 1]
                    e = (e + 180) % 360 - 180
                    err.append((e, col, 1 if k == 'M2' else 2, mine))
                    labels.append(f'{k} {rms(e):.2f}°')
            self.draw_plot(top, t, ser, 'angle (°): truth grey, gyro blue, accelerometer orange, fused red '
                                        '(like slide 33)', floor=10)
            goal = f' · goal: M3 below {GOAL_M3}°' if key == 'M3' else ''
            self.draw_plot(bot, t, err, 'error (°), RMS over 20 s: ' + ('  '.join(labels) if labels else '—') + goal,
                           floor=2)
            return
        rows = self.ring2.rows()
        t = rows[:, 0]
        key = self.sel[2]
        has5 = key != 'M4' and self.expr['M5'] is not None
        if key == 'M6':
            self.draw_plot(top, t, [(rows[:, 5], COL_TRUE, 2), (rows[:, 6], COL_FUSED, 2)],
                           f'prediction error at {self.latency_ms} ms (°): no prediction grey, M6 red '
                           f'(avg {np.nanmean(rows[:, 5]) if len(t) else 0:.2f}° vs '
                           f'{np.nanmean(rows[:, 6]) if len(t) and np.isfinite(rows[:, 6]).any() else math.nan:.2f}°)',
                           symmetric=False, floor=1)
        else:
            ser = [(rows[:, 1], COL_GYRO if not has5 else lighter(COL_GYRO, 0.4), 2, not has5)]
            if has5:
                ser.append((rows[:, 3], COL_FUSED, 2))
            c = 3 if has5 else 1
            self.draw_plot(top, t, ser, f'tilt error (°): RMS {rms(rows[:, c]):.2f}° over 20 s',
                           symmetric=False, floor=2)
        ser = [(rows[:, 2], COL_GYRO if not has5 else lighter(COL_GYRO, 0.4), 2, not has5)]
        if has5:
            ser.append((rows[:, 4], COL_FUSED, 2))
        c = 4 if has5 else 2
        self.draw_plot(bot, t, ser, f'yaw error (°): RMS {rms(rows[:, c]):.2f}° over 20 s'
                       + (' — gravity cannot see yaw' if has5 and not self.use_mag else ''), floor=2)

    # ------------------------------------------------------------ title and mode bar

    def draw_title(self):
        scr = self.screen
        names = {1: 'Flatland', 2: 'Rift'}
        st = self.st[self.tab]
        if self.live_mode[self.tab]:
            src = 'your head (H: recorded)'
        else:
            rec = self.rec1 if self.tab == 1 else self.rec2
            src = f'recorded head: {L.note_at(rec.notes, (st["k"] * L.DT) % L.LOOP_SECS)}'
        t = self.big.render(f'IMU · {names[self.tab]}', True, FG)
        scr.blit(t, (PAD, 9))
        sub = f'{src} · t = {st["t_abs"]:.1f} s'
        if self.paused:
            sub += ' · paused'
        elif self.speed > 1 and not self.live_mode[self.tab]:
            sub += f' · ×{self.speed}'
        scr.blit(self.small.render(sub, True, MUTED), (PAD + t.get_width() + 14, 13))
        imu = self.imu1 if self.tab == 1 else self.imu2
        flags = f'bias {"on" if imu.bias_on else "off"} · noise {"on" if imu.noise_on else "off"}'
        if st['b_hat'] is not None:
            flags += ' · calibrated'
        elif st['calib'] is not None:
            flags += ' · calibrating…'
        r = self.small.render(flags, True, MUTED)
        scr.blit(r, (VIEW.right - r.get_width(), 13))

    def draw_modebar(self):
        scr = self.screen
        pygame.draw.rect(scr, PANE_BG, MODE)
        pygame.draw.rect(scr, LINE, MODE, 1)
        st = self.st[self.tab]
        last = st['last']
        imu = self.imu1 if self.tab == 1 else self.imu2
        temp = imu.temperature(st['t_abs'])
        bias = np.degrees(np.linalg.norm(imu.bias(st['t_abs']))) if imu.bias_on else 0.0
        if last is not None:
            if self.tab == 1:
                am = last['a_meas']
                line = (f'θ {math.degrees(last["theta"]):+6.1f}° · ω̃ {math.degrees(last["w_meas"]):+7.1f}°/s · '
                        f'ã ({am[0]:+5.2f}, {am[1]:+5.2f}) m/s², |ã| {np.linalg.norm(am):5.2f} · '
                        f'board {temp:4.1f} °C, bias {bias:.2f}°/s')
            else:
                am = last['a_meas']
                a = am / max(1e-9, np.linalg.norm(am))
                pitch = -math.degrees(math.atan2(a[2], math.hypot(a[0], a[1])))
                roll = -math.degrees(math.atan2(-a[0], a[1]))
                tr = last['R'][1]                       # Rᵀ up, noiseless
                tp = -math.degrees(math.atan2(tr[2], math.hypot(tr[0], tr[1])))
                trl = -math.degrees(math.atan2(-tr[0], tr[1]))
                line = (f'from ã (slides 38, 40): pitch {pitch:+5.1f}° roll {roll:+5.1f}° (truth {tp:+5.1f}°, '
                        f'{trl:+5.1f}°) · |ã| {np.linalg.norm(am):5.2f} · board {temp:4.1f} °C')
                if self.pos_demo is not None:
                    pd = self.pos_demo
                    line = (f'double-integrated position: {np.linalg.norm(pd["p"] - pd["true"]):.2f} m off after '
                            f'{pd["t"]:.1f} s (P stops) [Oculus §VIII]')
            scr.blit(self.small.render(line, True, FG), (MODE.x + 8, MODE.y + 8))
        if self.msg:
            col = self.msg_col
            msg = self.msg
            while self.small.size(msg)[0] > MODE.w - 16 and len(msg) > 8:
                msg = msg[:-2].rstrip() + '…'
            scr.blit(self.small.render(msg, True, col), (MODE.x + 8, MODE.y + 32))

    def say(self, msg, col=FG):
        self.msg, self.msg_col, self.msg_t = msg, col, time.time()

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

    def button(self, label, action, x, y, ready=True, on=None):
        fg = WHITE if ready and on is not False else MUTED
        t = self.small.render(label, True, fg if on is None or on else FG)
        r = pygame.Rect(x, y, t.get_width() + 18, 26)
        if on is None:
            bg = ACCENT if ready else TRACK
        else:
            bg = ACCENT if on else WHITE
        pygame.draw.rect(self.screen, bg, r, border_radius=6)
        if on is False:
            pygame.draw.rect(self.screen, LINE, r, 1, border_radius=6)
            t = self.small.render(label, True, FG)
        self.screen.blit(t, (r.x + 9, r.y + 6))
        self.hit['buttons'].append((r, action))
        return r.right + 6

    def slider(self, name, label, value_text, u, x, y, w=200):
        scr = self.screen
        scr.blit(self.small.render(label, True, FG), (x, y))
        tx = x + 70
        track = pygame.Rect(tx, y + 6, w, 4)
        pygame.draw.rect(scr, TRACK, track, border_radius=2)
        kx = tx + int(u * w)
        pygame.draw.rect(scr, ACCENT, (tx, y + 6, kx - tx, 4), border_radius=2)
        pygame.draw.circle(scr, ACCENT, (kx, y + 8), 7)
        scr.blit(self.small.render(value_text, True, FG), (tx + w + 12, y))
        self.hit['sliders'].append((pygame.Rect(tx - 8, y - 4, w + 16, 22), name, tx, w))

    def magnet_surface(self, m):
        return self.tex.get(L.magnet_tex(m), 13, FG)

    def magnet_width(self, m):
        return max(22, self.magnet_surface(m).get_width() + 12)

    def draw_magnet(self, m, x, y, lifted=False, scale=1.0, good=False):
        w = self.magnet_width(m)
        r = pygame.Rect(x, y, w, MAG_H)
        d = r
        s = self.magnet_surface(m)
        if scale != 1.0:
            d = pygame.Rect(0, 0, int(w * scale), int(MAG_H * scale))
            d.center = r.center
            s = pygame.transform.smoothscale(s, (int(s.get_width() * scale), int(s.get_height() * scale)))
        if lifted:
            pygame.draw.rect(self.screen, LINE, d.move(3, 3), border_radius=3)
        bg = GOOD_BG if good else (233, 241, 255) if m in L.SYMBOLS else WHITE
        pygame.draw.rect(self.screen, bg, d, border_radius=3)
        pygame.draw.rect(self.screen, lighter(GOOD, 0.2) if good else FAINT, d, 1, border_radius=3)
        self.screen.blit(s, (d.centerx - s.get_width() // 2, d.centery - s.get_height() // 2))
        return r

    def draw_sidebar(self):
        scr = self.screen
        pygame.draw.rect(scr, PANE_BG, SB)
        pygame.draw.rect(scr, LINE, SB, 1)
        x0, y = SB.x + 12, SB.y + 10
        # tabs
        for tab, name in ((1, '1 Flatland'), (2, '2 Rift')):
            t = self.bold.render(name, True, WHITE if self.tab == tab else FG)
            r = pygame.Rect(x0 + (0 if tab == 1 else 120), y, 112, 28)
            open_ = tab == 1 or self.unlocked('M4')
            pygame.draw.rect(scr, ACCENT if self.tab == tab else (WHITE if open_ else TRACK), r, border_radius=6)
            if self.tab != tab:
                pygame.draw.rect(scr, LINE, r, 1, border_radius=6)
            scr.blit(t if open_ or self.tab == tab else self.bold.render(name, True, FAINT),
                     (r.centerx - t.get_width() // 2, r.y + 6))
            self.hit['tabs'].append((r, tab))
        # stage chips
        cx = x0 + 252
        for key in L.STAGE_ORDER:
            st = L.STAGE[key]
            r = pygame.Rect(cx, y + 2, 30, 24)
            sel = self.sel[self.tab] == key
            if self.checked.get(key) is True or (key == 'M0' and key in self.done):
                bg, fg = GOOD_BG, GOOD
            elif self.unlocked(key):
                bg, fg = WHITE, FG
            else:
                bg, fg = TRACK, FAINT
            pygame.draw.rect(scr, bg, r, border_radius=5)
            pygame.draw.rect(scr, ACCENT if sel else LINE, r, 2 if sel else 1, border_radius=5)
            t = self.small.render(key[1], True, fg)
            scr.blit(t, (r.centerx - t.get_width() // 2, r.y + 5))
            self.hit['chips'].append((r, key))
            cx += 33
        y += 40
        st = self.stage()
        key = st.key
        # card
        head = self.big.render(f'{key} · {st.title}', True, FG)
        scr.blit(head, (x0, y))
        sl = self.small.render(f'slides {st.slides}', True, FAINT)
        scr.blit(sl, (SB.right - 12 - sl.get_width(), y + 4))
        y += 26
        for lab, text, col in (('Goal', st.goal, FG), ('Do', st.do, FG), ('See', st.see, FG),
                               ('Think', st.think, ACCENT)):
            scr.blit(self.bold.render(lab, True, col), (x0, y))
            for i, ln in enumerate(self.wrap(text, SB.w - 24 - 58)):
                scr.blit(self.small.render(ln, True, col if lab == 'Think' else FG), (x0 + 58, y + 1 + 15 * i))
                last_i = i
            y += 15 * (last_i + 1) + 6
        y += 4
        pygame.draw.line(scr, TRACK, (x0, y), (SB.right - 12, y))
        y += 10
        if key == 'M0':
            self.button('Got it → M1', self.finish_m0, x0, y)
            y += 40
        else:
            y = self.draw_line_editor(st, x0, y)
        y = self.draw_controls(st, x0, y + 4)

    def finish_m0(self):
        self.done.add('M0')
        self.enter_stage('M1')
        self.say('M1: drag magnets into the empty slot to write the gyro update.', MUTED)

    def draw_line_editor(self, st, x0, y):
        scr = self.screen
        key = st.key
        good = self.checked[key] is True
        # the line
        row = pygame.Rect(x0, y, SB.w - 24, 40)
        bg = (240, 253, 244) if good else (248, 250, 252)
        t0 = self.pops.get(key)
        if good and t0 is not None and time.time() - t0 < 0.8:
            f = 1 - (time.time() - t0) / 0.8
            bg = tuple(int(bg[i] + ((187, 247, 208)[i] - bg[i]) * f) for i in range(3))
        pygame.draw.rect(scr, bg, row, border_radius=6)
        pygame.draw.rect(scr, lighter(GOOD, 0.3) if good else LINE, row, 1, border_radius=6)
        self.hit['line_rect'] = row
        x = row.x + 8
        cy = row.centery
        lhs = self.tex.get(st.lhs, 15, FG)
        scr.blit(lhs, (x, cy - lhs.get_height() // 2))
        x += lhs.get_width() + 8
        drag_mag = self.drag is not None
        ordinal = 0
        for b in st.body:
            if isinstance(b, int):
                toks = self.slots[key][b]
                w = sum(self.magnet_width(m) + 3 for m in toks) + 8
                sr = pygame.Rect(x, cy - MAG_H // 2 - 4, max(64, w), MAG_H + 8)
                over = drag_mag and sr.inflate(10, 14).collidepoint(self.drag['pos'])
                if over:
                    pygame.draw.rect(scr, (255, 247, 230), sr, border_radius=4)
                if not toks:
                    dashed_rect(scr, AMBER if drag_mag else FAINT, sr)
                else:
                    pygame.draw.rect(scr, LINE, sr, 1, border_radius=4)
                self.hit['slots'].append((sr, (key, b)))
                mx = sr.x + 4
                for j, m in enumerate(toks):
                    r = self.draw_magnet(m, mx, sr.y + 4, scale=self.pop_scale(key, ordinal), good=good)
                    self.hit['mags'].append((r, (key, b, j)))
                    mx = r.right + 3
                    ordinal += 1
                if over:
                    k = self.insert_index(key, b, self.drag['pos'][0])
                    rs = [rr for rr, (kk, bb, _) in self.hit['mags'] if kk == key and bb == b]
                    ix = rs[k - 1].right + 1 if k > 0 else sr.x + 2
                    pygame.draw.line(scr, ACCENT, (ix, sr.y + 3), (ix, sr.bottom - 3), 3)
                x = sr.right + 6
            else:
                s = self.tex.get(L.magnet_tex(b), 15, FG)
                scr.blit(s, (x, cy - s.get_height() // 2))
                x += s.get_width() + 5
        y = row.bottom + 6
        # what the line does
        e, err, ok, hint = self.expr[key], self.err[key], self.checked[key], self.hint[key]
        if ok is True:
            note, col = '✓ correct', GOOD
        elif err == 'empty':
            note, col = 'drag magnets from the palette into the slot', MUTED
        elif err:
            note, col = f'? {err}', AMBER
        elif ok is False:
            note, col = (f'✗ runs, not right yet — {hint}' if hint else
                         '✗ runs, but not right yet: watch what it does'), BAD
        else:
            note, col = '', MUTED
        if e is not None and e.warn and ok is not True:
            note, col = f'⚠ {e.warn}', AMBER
        for i, ln in enumerate(self.wrap(note, SB.w - 24)[:3]):
            scr.blit(self.small.render(ln, True, col), (x0, y + 15 * i))
        y += 15 * min(3, len(self.wrap(note, SB.w - 24))) + 6
        # palette
        scr.blit(self.small.render('palette (drag; copies stay)', True, FAINT), (x0, y))
        y += 16
        px = x0
        for m in st.palette:
            w = self.magnet_width(m)
            if px + w > SB.right - 12:
                px = x0
                y += MAG_H + 6
            r = self.draw_magnet(m, px, y)
            self.hit['palette'].append((r, m))
            px = r.right + 6
        y += MAG_H + 10
        # next / answer
        bx = x0
        nxt = L.STAGE_ORDER.index(key) + 1
        if good and nxt < len(L.STAGE_ORDER):
            nk = L.STAGE_ORDER[nxt]
            bx = self.button(f'Next → {nk}', lambda nk=nk: self.enter_stage(nk), bx, y)
        if key in self.released and not good:
            bx = self.button('A  answer', self.fill_answer, bx, y)
        if bx != x0:
            y += 32
        return y

    def draw_controls(self, st, x0, y):
        scr = self.screen
        key = st.key
        pygame.draw.line(scr, TRACK, (x0, y), (SB.right - 12, y))
        y += 8
        tab = st.tab
        imu = self.imu1 if tab == 1 else self.imu2
        x = x0
        x = self.button('H your head' if not self.live_mode[tab] else 'H recorded', self.toggle_live, x, y)
        x = self.button('R restart', lambda: self.restart(self.tab), x, y)
        x = self.button('Space pause', self.toggle_pause, x, y, on=self.paused)
        x = self.button('. ×10', self.toggle_speed, x, y, on=self.speed > 1)
        y += 32
        if key in ('M1', 'M2', 'M3', 'M4', 'M5'):
            x = x0
            x = self.button('B bias', self.toggle_bias, x, y, on=imu.bias_on)
            x = self.button('N noise', self.toggle_noise, x, y, on=imu.noise_on)
            cal = self.st[tab]['b_hat'] is not None or self.st[tab]['calib'] is not None
            x = self.button('K calibrate', lambda: self.start_calibration(self.tab), x, y, on=cal)
            if key in ('M3', 'M5'):
                rej = self.reject1 if tab == 1 else self.reject2
                x = self.button('S reject shakes', self.toggle_reject, x, y, on=rej)
            y += 34
        if key in ('M3', 'M5'):
            a = self.alpha1 if tab == 1 else self.alpha2
            u = (-math.log10(max(1e-6, 1 - a)) - 1) / 3
            tau = L.alpha_tau(a)
            self.slider('alpha', '[ ] α', f'{a:.4f}  τ {tau * 1000:.0f} ms' if tau < 1 else f'{a:.4f}  τ {tau:.1f} s',
                        max(0, min(1, u)), x0, y, 190)
            y += 28
        if key == 'M5':
            x = x0
            x = self.button('M magnetometer', self.toggle_mag, x, y, on=self.use_mag)
            x = self.button('G magnet nearby', self.toggle_magnet, x, y, on=self.magnet)
            x = self.button('P position', self.toggle_pos, x, y, on=self.pos_demo is not None)
            y += 34
        if key == 'M6':
            x = self.button('O prediction', self.toggle_predict, x0, y, on=self.predict_on)
            x = self.button('M magnetometer', self.toggle_mag, x, y, on=self.use_mag)
            self.button('P position', self.toggle_pos, x, y, on=self.pos_demo is not None)
            y += 34
            self.slider('latency', 'L latency', f'{self.latency_ms} ms', self.latency_ms / 60, x0, y, 190)
            y += 26
            y = self.draw_prediction_table(x0, y)
        if key == 'M0':
            scr.blit(self.small.render('Keys: H your head · R restart · Space pause · . fast-forward', True, MUTED),
                     (x0, y))
        return y

    def draw_prediction_table(self, x0, y):
        scr = self.screen
        rows = self.ring2.rows()
        if len(rows) < 10:
            return y
        n0, n1 = rows[:, 5], rows[:, 6]

        def stat(v):
            v = v[np.isfinite(v)]
            return (f'{v.mean():.2f}°', f'{v.max():.2f}°') if len(v) else ('—', '—')

        lines = [('', 'average', 'worst'), (f'no prediction, {self.latency_ms} ms', *stat(n0)),
                 (f'your M6, {self.latency_ms} ms', *stat(n1)),
                 ('paper, none, 20 ms', f'{PAPER_20MS[0]:.2f}°', f'{PAPER_20MS[1]:.2f}°'),
                 ('paper, constant rate', f'{PAPER_20MS[2]:.2f}°', f'{PAPER_20MS[3]:.2f}°')]
        for i, (a, b, c) in enumerate(lines):
            col = MUTED if i == 0 or i >= 3 else FG
            scr.blit(self.small.render(a, True, col), (x0, y))
            scr.blit(self.small.render(b, True, col), (x0 + 250, y))
            scr.blit(self.small.render(c, True, col), (x0 + 340, y))
            y += 16
        scr.blit(self.small.render('(errors against your own estimate T later, as in the paper)', True, FAINT),
                 (x0, y))
        return y + 18

    # ------------------------------------------------------------ toggles

    def toggle_live(self):
        tab = self.tab
        self.live_mode[tab] = not self.live_mode[tab]
        last = self.st[tab]['last']
        if tab == 1:
            lv = self.live1
            th = last['theta'] if last else 0.0
            lv.ang[:] = lv.target_ang[:] = th
            lv.dang[:] = 0
            lv.pos[:] = lv.target_pos[:] = lv.dpos[:] = 0
        else:
            lv = self.live2
            R = last['R'] if last else np.eye(3)
            f = -R[:, 2]
            yaw = math.atan2(-f[0], -f[2])
            pitch = math.asin(max(-1.0, min(1.0, f[1])))
            lv.ang[:] = lv.target_ang[:] = (yaw, pitch, 0.0)
            lv.dang[:] = 0
            lv.pos[:] = lv.target_pos[:] = lv.dpos[:] = 0
            lv.R = L.head_rotation(*lv.ang)
            lv.p_hist = []
        self.reset_trackers(tab)
        self.st[tab]['last'] = None
        self.simulate(tab, 1)
        self.reset_trackers(tab)
        if self.live_mode[tab]:
            self.say('your head: drag in the view to turn it, right-drag to shake it. H: back to the recording.', ACCENT)
        else:
            self.restart(tab)
            self.say('the recorded head again', MUTED)

    def toggle_pause(self):
        self.paused = not self.paused

    def toggle_speed(self):
        self.speed = 10 if self.speed == 1 else 1
        if self.live_mode[self.tab] and self.speed > 1:
            self.say('fast-forward works on the recorded head only (H)', AMBER)

    def toggle_bias(self):
        imu = self.imu1 if self.tab == 1 else self.imu2
        imu.bias_on = not imu.bias_on
        self.say(f'gyro bias {"on" if imu.bias_on else "off"}', MUTED)

    def toggle_noise(self):
        imu = self.imu1 if self.tab == 1 else self.imu2
        imu.noise_on = not imu.noise_on
        self.say(f'sensor noise {"on" if imu.noise_on else "off"}', MUTED)

    def toggle_reject(self):
        if self.tab == 1:
            self.reject1 = not self.reject1
            on = self.reject1
        else:
            self.reject2 = not self.reject2
            on = self.reject2
        self.say('reject shakes: skip the accelerometer when |ã| is far from g or jumps [Oculus §V]' if on else
                 'reject shakes off', MUTED)
        self.celebrated.discard(self.sel[self.tab])

    def toggle_mag(self):
        self.use_mag = not self.use_mag
        self.trk2.mag_ref = None
        self.say('magnetometer: the yaw is pulled towards the heading it had when switched on [Oculus §VI]'
                 if self.use_mag else 'magnetometer off: nothing corrects the yaw', MUTED)

    def toggle_magnet(self):
        self.magnet = not self.magnet
        self.say('a magnet on the desk bends the field: the magnetometer now points 35° off' if self.magnet else
                 'magnet removed', AMBER if self.magnet else MUTED)

    def toggle_pos(self):
        if self.pos_demo is not None:
            self.pos_demo = None
            return
        last = self.st[2]['last']
        if last is None:
            return
        self.pos_demo = dict(p=last['pos'].copy(), v=np.zeros(3), true=last['pos'].copy(), t=0.0)
        self.say('position from the accelerometer: subtract gravity, integrate twice. Watch the red dot.', ACCENT)

    def toggle_predict(self):
        self.predict_on = not self.predict_on
        self.celebrated.discard('M6')

    def set_alpha(self, a):
        a = max(0.9, min(0.9999, a))
        if self.tab == 1:
            self.alpha1 = a
        else:
            self.alpha2 = a
        self.celebrated.discard(self.sel[self.tab])

    def step_alpha(self, d):
        a = self.alpha1 if self.tab == 1 else self.alpha2
        u = -math.log10(1 - a) + d * 0.25
        self.set_alpha(1 - 10 ** -u)

    def set_latency(self, ms):
        self.latency_ms = int(max(0, min(60, round(ms / 10) * 10)))
        self.celebrated.discard('M6')

    # ------------------------------------------------------------ input

    def insert_index(self, key, b, x):
        cs = [r.centerx for r, (kk, bb, _) in self.hit.get('mags', []) if kk == key and bb == b]
        return sum(1 for c in cs if c < x)

    def slot_at(self, pos):
        for rect, slot in self.hit.get('slots', []):
            if rect.inflate(10, 14).collidepoint(pos):
                return slot
        return None

    def mag_at(self, pos):
        for rect, where in self.hit.get('mags', []):
            if rect.collidepoint(pos):
                return where
        return None

    def select_tab(self, tab):
        if tab == 2 and not self.unlocked('M4'):
            self.say('the Rift tab opens once M3 is right (or press A on M3 after the answer is released)', AMBER)
            return
        self.tab = tab
        self.enter_stage(self.sel[tab])

    def mouse_down(self, ev):
        pos = ev.pos
        if ev.button in (4, 5):
            return
        for rect, tab in self.hit.get('tabs', []):
            if rect.collidepoint(pos) and ev.button == 1:
                self.select_tab(tab)
                return
        for rect, key in self.hit.get('chips', []):
            if rect.collidepoint(pos) and ev.button == 1:
                if not self.unlocked(key):
                    prev = L.STAGE_ORDER[L.STAGE_ORDER.index(key) - 1]
                    self.say(f'{key} opens once {prev} is right', AMBER)
                elif L.STAGE[key].tab == 2 and not self.unlocked('M4'):
                    self.select_tab(2)
                else:
                    self.enter_stage(key)
                return
        for rect, action in self.hit.get('buttons', []):
            if rect.collidepoint(pos) and ev.button == 1:
                action()
                return
        for rect, name, tx, w in self.hit.get('sliders', []):
            if rect.collidepoint(pos) and ev.button == 1:
                self.slider_drag = (name, tx, w)
                self.slide(pos)
                return
        where = self.mag_at(pos)
        if where is not None:
            key, b, j = where
            m = self.slots[key][b].pop(j)
            self.recompile(key)
            if ev.button == 1:
                self.drag = dict(mag=m, pos=pos)
            return
        for rect, m in self.hit.get('palette', []):
            if rect.collidepoint(pos) and ev.button == 1:
                self.drag = dict(mag=m, pos=pos)
                return
        # views
        if self.tab == 1 and VIEW.collidepoint(pos) and self.sel[1] in L.STAGE_ORDER:
            if not self.live_mode[1]:
                self.toggle_live()
            self.view_drag = ('tilt' if ev.button == 1 else 'shake', pos)
            self.view_input(pos)
        elif self.tab == 2 and VVR.collidepoint(pos):
            if not self.live_mode[2]:
                self.toggle_live()
            self.view_drag = ('turn' if ev.button == 1 else 'shake', pos)
        elif self.tab == 2 and V3D.collidepoint(pos) and ev.button == 1:
            self.view_drag = ('orbit', pos)

    def slide(self, pos):
        name, tx, w = self.slider_drag
        u = max(0.0, min(1.0, (pos[0] - tx) / w))
        if name == 'alpha':
            self.set_alpha(1 - 10 ** (-1 - 3 * u))
        elif name == 'latency':
            self.set_latency(u * 60)

    def view_input(self, pos):
        mode, last = self.view_drag
        if self.tab == 1:
            px, py = VIEW.centerx, VIEW.y + 330
            if mode == 'tilt':
                dx, dy = pos[0] - px, py - pos[1]
                th = math.atan2(-dx, max(10, dy))
                self.live1.target_ang[0] = max(-80 * L.DEG, min(80 * L.DEG, th))
            else:
                self.live1.target_pos[0] = max(-0.04, min(0.04, (pos[0] - last[0]) / (PX_PER_M * SHAKE_GAIN)))
            return
        dx, dy = pos[0] - last[0], pos[1] - last[1]
        if mode == 'orbit':
            self.cam[0] -= dx * 0.4
            self.cam[1] = max(-80.0, min(80.0, self.cam[1] + dy * 0.4))
            self.view_drag = (mode, pos)
        elif mode == 'turn':
            ta = self.live2.target_ang
            ta[0] -= dx * 0.35 * L.DEG
            ta[1] = max(-80 * L.DEG, min(80 * L.DEG, ta[1] - dy * 0.35 * L.DEG))
            self.view_drag = (mode, pos)
        elif mode == 'shake':
            self.live2.target_pos[0] = max(-0.05, min(0.05, dx / 1500))
            self.live2.target_pos[1] = max(-0.05, min(0.05, -dy / 1500))

    def mouse_up(self, ev):
        if self.slider_drag:
            self.slider_drag = None
            return
        if self.view_drag:
            if self.view_drag[0] == 'shake':
                if self.tab == 1:
                    self.live1.target_pos[0] = 0.0
                else:
                    self.live2.target_pos[:] = 0.0
            self.view_drag = None
            return
        if not self.drag:
            return
        d, self.drag = self.drag, None
        slot = self.slot_at(ev.pos)
        if slot is None:
            return
        key, b = slot
        self.slots[key][b].insert(self.insert_index(key, b, ev.pos[0]), d['mag'])
        self.recompile(key)

    def mouse_motion(self, ev):
        if self.drag:
            self.drag['pos'] = ev.pos
        elif self.slider_drag:
            self.slide(ev.pos)
        elif self.view_drag:
            self.view_input(ev.pos)

    def mouse_wheel(self, ev):
        pos = self.mouse
        if self.tab == 2 and V3D.collidepoint(pos):
            self.cam[2] = max(0.4, min(3.0, self.cam[2] * (0.9 if ev.y > 0 else 1.1)))
        elif self.tab == 2 and VVR.collidepoint(pos):
            if not self.live_mode[2]:
                self.toggle_live()
            ta = self.live2.target_ang
            ta[2] = max(-60 * L.DEG, min(60 * L.DEG, ta[2] + ev.y * 4 * L.DEG))

    def key_down(self, ev):
        k = ev.key
        mods = getattr(ev, 'mod', 0) or pygame.key.get_mods()
        if k == pygame.K_ESCAPE:
            if self.drag:
                self.drag = None
                return
            raise SystemExit
        if k == pygame.K_1:
            self.select_tab(1)
        elif k == pygame.K_2:
            self.select_tab(2)
        elif k == pygame.K_SPACE:
            self.toggle_pause()
        elif k == pygame.K_PERIOD:
            self.toggle_speed()
        elif k == pygame.K_r:
            self.restart(self.tab)
            self.say('recording restarted', MUTED)
        elif k == pygame.K_h:
            self.toggle_live()
        elif k == pygame.K_b:
            self.toggle_bias()
        elif k == pygame.K_n:
            self.toggle_noise()
        elif k == pygame.K_k:
            self.start_calibration(self.tab)
        elif k == pygame.K_LEFTBRACKET:
            self.step_alpha(-1)
        elif k == pygame.K_RIGHTBRACKET:
            self.step_alpha(1)
        elif k == pygame.K_s:
            self.toggle_reject()
        elif k == pygame.K_a:
            self.fill_answer()
        elif k == pygame.K_c:
            self.cam = [35.0, 20.0, 0.95]
        elif self.tab == 2 and k == pygame.K_m:
            self.toggle_mag()
        elif self.tab == 2 and k == pygame.K_g:
            self.toggle_magnet()
        elif self.tab == 2 and k == pygame.K_p:
            self.toggle_pos()
        elif self.tab == 2 and k == pygame.K_l:
            self.set_latency(self.latency_ms + (-10 if mods & pygame.KMOD_SHIFT else 10))
        elif self.tab == 2 and k == pygame.K_o:
            self.toggle_predict()
        elif k in (pygame.K_BACKSPACE, pygame.K_DELETE):
            key = self.sel[self.tab]
            if key == 'M0':
                return
            if k == pygame.K_DELETE:
                self.slots[key] = [[] for _ in self.slots[key]]
            else:
                slot = self.slot_at(self.mouse)
                b = slot[1] if slot and slot[0] == key else 0
                if self.slots[key][b]:
                    self.slots[key][b].pop()
            self.recompile(key)

    # ------------------------------------------------------------ main loop

    def draw_drag(self):
        if self.drag:
            x, y = self.drag['pos']
            self.draw_magnet(self.drag['mag'], x - 12, y - MAG_H // 2, lifted=True)

    def draw(self):
        self.screen.fill(BG)
        self.hit = {'tabs': [], 'chips': [], 'buttons': [], 'slots': [], 'mags': [], 'palette': [], 'sliders': [],
                    'line_rect': None}
        self.draw_title()
        if self.tab == 1:
            self.draw_flatland()
        else:
            self.draw_3d()
            self.draw_vr()
        self.draw_strips()
        self.draw_modebar()
        self.draw_sidebar()
        self.draw_particles()
        self.draw_banner()
        self.draw_drag()
        pygame.display.flip()

    def handle(self, ev):
        if ev.type == pygame.QUIT:
            raise SystemExit
        if hasattr(ev, 'pos'):
            self.mouse = ev.pos
        if ev.type == pygame.MOUSEBUTTONDOWN:
            self.mouse_down(ev)
        elif ev.type == pygame.MOUSEBUTTONUP:
            self.mouse_up(ev)
        elif ev.type == pygame.MOUSEMOTION:
            self.mouse_motion(ev)
        elif ev.type == pygame.MOUSEWHEEL:
            self.mouse_wheel(ev)
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
    ap.add_argument('--answers', action='store_true', help='fill every stage that has a released answer')
    ap.add_argument('--stage', choices=L.STAGE_ORDER[1:], help='start at this stage (needs the earlier answers)')
    args = ap.parse_args()
    App(args).loop()


if __name__ == '__main__':
    main()
