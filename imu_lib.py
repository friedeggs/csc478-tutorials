"""Given code for Tutorial 8 (2026-11-05, Inertial Measurement Units).

Everything the demo needs except the six lines you build with magnets:
rotations, the head motions (recorded and live), a simulated 1000 Hz IMU
(gyro, accelerometer, magnetometer; Lecture 7 slides 10, 21, 42), the magnet
parser with its unit checks, and the given parts of the filters from LaValle
et al., "Head Tracking for the Oculus Rift", ICRA 2014.

Conventions
    Flatland (tab 1): the head tilts by theta (rad, counter-clockwise) in the
        x-y plane, y up. The 2-axis accelerometer reads the specific force in
        the head's frame, so at rest atan2(a_x, a_y) = theta (slide 29).
    Rift (tab 2): world y is up, the headset looks along -z. R maps head
        coordinates to world coordinates. The gyro and the accelerometer
        measure in head coordinates.
"""
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANSWERS_DIR = HERE / 'answers'

RATE = 1000                 # samples per second, as the Rift's sensor board [LaValle §II]
DT = 1.0 / RATE
G = 9.81
DEG = math.pi / 180
LOOP_SECS = 60.0            # both recordings loop after a minute
UP = np.array([0.0, 1.0, 0.0])
INCLINATION = 60 * DEG      # the magnetic field points north (-z) and down
M_WORLD = np.array([0.0, -math.sin(INCLINATION), -math.cos(INCLINATION)])


# ---------------------------------------------------------------- rotations

def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rot(axis, angle):
    """Rotation by angle (rad) about axis (Rodrigues)."""
    a = np.asarray(axis, float)
    n = math.sqrt(a @ a)
    if n < 1e-12 or angle == 0:
        return np.eye(3)
    k = a / n
    K = skew(k)
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def exp_so3(v):
    """Rotation vector (axis * angle, rad) to rotation matrix."""
    v = np.asarray(v, float)
    return rot(v, math.sqrt(v @ v))


def log_so3(R):
    """Rotation matrix to rotation vector (rad)."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    a = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if a < 1e-9:
        return w / 2
    if a > math.pi - 1e-6:                     # half turn: axis from R + I
        M = (R + np.eye(3)) / 2
        k = M[np.argmax(np.diag(M))]
        return k / np.linalg.norm(k) * a
    return w * (a / (2 * math.sin(a)))


def log_so3_batch(R):
    """log_so3 for an (N, 3, 3) stack of small rotations."""
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    a = np.arccos(c)
    w = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    s = np.where(a < 1e-9, 0.5, a / (2 * np.sin(np.maximum(a, 1e-9))))
    return w * s[:, None]


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def head_rotation(yaw, pitch, roll):
    """Turn (about the world's vertical), then look up / down, then tilt the head."""
    return rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)


def head_rotation_batch(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    n = len(yaw)
    Ry = np.zeros((n, 3, 3)); Rx = np.zeros((n, 3, 3)); Rz = np.zeros((n, 3, 3))
    Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 1, 1], Ry[:, 2, 0], Ry[:, 2, 2] = cy, sy, 1, -sy, cy
    Rx[:, 0, 0], Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2] = 1, cp, -sp, sp, cp
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = cr, -sr, sr, cr, 1
    return Ry @ Rx @ Rz


def angle_of(R):
    """Rotation angle of R in degrees."""
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def tilt_error(R_est, R_true):
    """Degrees between where each orientation thinks 'up' is, in head coordinates."""
    a, b = R_est[1], R_true[1]                 # rows: Rᵀ up
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))


def heading(R):
    f = -R[:, 2]                               # where the headset looks
    return math.atan2(-f[0], -f[2])            # 0 = north (-z), positive = turned left


def yaw_error(R_est, R_true):
    """Degrees between the two headings, about the vertical."""
    d = heading(R_est) - heading(R_true)
    return math.degrees((d + math.pi) % (2 * math.pi) - math.pi)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------- recordings

def min_jerk(keys, t):
    """Piecewise minimum-jerk path through keyframes [(time, value), ...]: zero velocity and
    acceleration at every key, so the motion is smooth and its derivatives are continuous."""
    kt = np.array([k[0] for k in keys], float)
    kv = np.array([k[1] for k in keys], float)
    i = np.clip(np.searchsorted(kt, t, side='right') - 1, 0, len(kt) - 2)
    s = np.clip((t - kt[i]) / (kt[i + 1] - kt[i]), 0, 1)
    b = s ** 3 * (10 - 15 * s + 6 * s * s)
    return kv[i] + (kv[i + 1] - kv[i]) * b


def envelope(t, t0, t1, ramp=0.3):
    up = min_jerk([(t0, 0), (t0 + ramp, 1)], t)
    down = min_jerk([(t1 - ramp, 1), (t1, 0)], t)
    return np.where(t < t0 + ramp, up, np.where(t > t1 - ramp, down, 1.0))


def shake(t, t0, t1, amp, hz):
    """Side to side: a sine inside a smooth envelope (metres)."""
    return amp * envelope(t, t0, t1) * np.sin(2 * math.pi * hz * (t - t0))


def second_derivative(x):
    return np.gradient(np.gradient(x, DT, axis=0), DT, axis=0)


NECK = 0.18                 # Flatland: the IMU sits 18 cm above the neck's pivot


@dataclass
class FlatRecording:
    t: np.ndarray
    theta: np.ndarray       # true angle (rad)
    omega: np.ndarray       # true angular velocity (rad/s)
    acc: np.ndarray         # (N, 2) linear acceleration of the IMU, world frame (m/s²)
    pivot: np.ndarray       # (N,) neck x position (m)
    notes: list = field(default_factory=list)   # (time, what happens)


def flat_recording():
    """60 s of a head tilting side to side, with a still start, quick tilts and two shakes."""
    t = np.arange(int(LOOP_SECS * RATE)) * DT
    keys = [(0, 0), (3, 0), (5, 25), (7, -20), (8, -20), (10, 30), (11.5, 0), (14.5, 0),
            (15.5, -30), (16.0, -30), (16.45, 30), (16.9, -30), (17.35, 30), (17.8, 0),
            (19, 0), (21, 35), (24, -35), (26, 10), (27, 10), (31, 10), (33, -15),
            (35, 20), (37, -25), (38, -25), (38.4, 25), (38.8, -25), (39.2, 0), (42, 0),
            (44, 30), (46, 30), (49, -30), (51, -30), (54, 5), (57, 0), (60, 0)]
    theta = min_jerk(keys, t) * DEG
    pivot = shake(t, 11.8, 14.2, 0.015, 3.0) + shake(t, 27.2, 30.8, 0.012, 3.5) + shake(t, 46.2, 48.8, 0.015, 3.0)
    p = np.stack([pivot - NECK * np.sin(theta), NECK * np.cos(theta)], 1)
    notes = [(0, 'holding still'), (3, 'slow tilts'), (11.8, 'shaking side to side'), (14.5, 'quick tilts'),
             (19, 'slow tilts'), (27.2, 'tilted and shaking'), (31, 'slow tilts'), (37, 'quick tilts'),
             (42, 'slow tilts'), (46.2, 'tilted and shaking'), (49, 'slow tilts'), (57, 'holding still')]
    return FlatRecording(t, theta, np.gradient(theta, DT), second_derivative(p), pivot, notes)


HEAD_OFFSET = np.array([0.0, 0.12, -0.08])   # Rift: IMU relative to the neck's pivot, head frame


@dataclass
class RiftRecording:
    t: np.ndarray
    R: np.ndarray           # (N, 3, 3) true orientation
    omega: np.ndarray       # (N, 3) true angular velocity, head frame (rad/s)
    acc: np.ndarray         # (N, 3) linear acceleration of the IMU, world frame (m/s²)
    pos: np.ndarray         # (N, 3) IMU position (m)
    notes: list = field(default_factory=list)


def rift_recording():
    """60 s of looking around: level first (yaw only), then look down and turn, nods, a head
    roll, a quick turn at about 270°/s, a lean forward, then free looking around."""
    t = np.arange(int(LOOP_SECS * RATE)) * DT
    yaw = min_jerk([(0, 0), (2, 0), (4.5, 60), (7, -60), (9.5, 30), (11, 0), (12.5, 0), (13.5, 0),
                    (16, 70), (18.5, -50), (20, 0), (30, 0), (30.6, 90), (31.6, 90), (32.2, 0),
                    (33, 0), (40, 0), (43, 50), (46, -40), (49, 20), (52, -30), (55, 0), (60, 0)], t)
    pitch = min_jerk([(0, 0), (11, 0), (12.5, -40), (18.5, -40), (20, 0), (22, 0), (22.6, 20),
                      (23.2, -20), (23.8, 20), (24.4, -20), (25, 0), (40, 0), (43, 15), (46, -25),
                      (49, 10), (52, -10), (55, 0), (60, 0)], t)
    roll = min_jerk([(0, 0), (26, 0), (27, 20), (28.5, -20), (29.5, 0), (40, 0), (44, 10), (48, -10),
                     (52, 5), (55, 0), (60, 0)], t)
    R = head_rotation_batch(yaw * DEG, pitch * DEG, roll * DEG)
    rel = np.einsum('nji,njk->nik', R[:-1], R[1:])
    omega = np.vstack([log_so3_batch(rel) / DT, np.zeros((1, 3))])
    omega[-1] = omega[-2]
    lean = min_jerk([(0, 0), (34, 0), (35.5, -0.15), (37.5, -0.15), (39, 0), (60, 0)], t)
    pivot = np.stack([shake(t, 37.6, 39.6, 0.012, 3.0), np.zeros_like(t), lean], 1)
    pos = pivot + R @ HEAD_OFFSET
    notes = [(0, 'holding still'), (2, 'looking around, level'), (11, 'looking down'),
             (13.5, 'turning while looking down'), (20, 'level again'), (22, 'nodding'),
             (26, 'tilting the head'), (30, 'a quick turn'), (34, 'leaning forward'),
             (37.6, 'shaking'), (40, 'looking around'), (55, 'holding still')]
    return RiftRecording(t, R, omega, second_derivative(pos), pos, notes)


def note_at(notes, t):
    s = ''
    for t0, what in notes:
        if t >= t0:
            s = what
    return s


class LiveHead:
    """'Your head': the truth follows the mouse through a critically damped spring, so the
    angular velocity and the acceleration stay finite. Angles in rad, positions in m."""

    W = 22.0                                  # spring's natural frequency (rad/s)

    def __init__(self, n_angles, n_pos):
        self.ang = np.zeros(n_angles)
        self.dang = np.zeros(n_angles)
        self.pos = np.zeros(n_pos)
        self.dpos = np.zeros(n_pos)
        self.target_ang = np.zeros(n_angles)
        self.target_pos = np.zeros(n_pos)

    def step(self):
        w = self.W
        dd = w * w * (self.target_ang - self.ang) - 2 * w * self.dang
        self.dang = self.dang + dd * DT
        self.ang = self.ang + self.dang * DT
        ddp = w * w * (self.target_pos - self.pos) - 2 * w * self.dpos
        self.dpos = self.dpos + ddp * DT
        self.pos = self.pos + self.dpos * DT
        return dd, ddp


class LiveFlat(LiveHead):
    def __init__(self):
        super().__init__(1, 1)

    def samples(self, n):
        """n samples of (theta, omega, acc (2,), pivot x)."""
        out = []
        for _ in range(n):
            dd, ddp = self.step()
            th, w = self.ang[0], self.dang[0]
            # IMU at pivot + NECK (-sin th, cos th): differentiate twice
            ax = ddp[0] - NECK * (math.cos(th) * dd[0] - math.sin(th) * w * w)
            ay = -NECK * (math.sin(th) * dd[0] + math.cos(th) * w * w)
            out.append((th, w, np.array([ax, ay]), self.pos[0]))
        return out


class LiveRift(LiveHead):
    def __init__(self):
        super().__init__(3, 3)
        self.R = np.eye(3)
        self.p_hist = []

    def samples(self, n):
        """n samples of (R, omega (head frame), acc (world), IMU position)."""
        out = []
        for _ in range(n):
            self.step()
            R = head_rotation(*self.ang)
            omega = log_so3(self.R.T @ R) / DT
            self.R = R
            p = self.pos + R @ HEAD_OFFSET
            self.p_hist = (self.p_hist + [p])[-3:]
            if len(self.p_hist) == 3:
                a = (self.p_hist[2] - 2 * self.p_hist[1] + self.p_hist[0]) / (DT * DT)
            else:
                a = np.zeros(3)
            out.append((R, omega, a, p))
        return out


# ---------------------------------------------------------------- the IMU

class IMU:
    """Simulated MEMS sensors (slide 10: gyro = truth + bias + noise; slide 21: the
    accelerometer reads gravity, pointing up, plus the head's own acceleration).

    The gyro bias grows as the board warms up [LaValle Fig. 3]: temperature rises
    from 28 °C towards 38 °C with a time constant of a minute."""

    def __init__(self, dims, seed=478, bias_dps=2.0, warm_dps=0.5, noise_dps=3.0, acc_noise=0.3,
                 mag_noise=0.01):
        self.rng = np.random.default_rng(seed)
        self.dims = dims
        if dims == 1:
            self.bias_dir = np.array([1.0])
        else:
            self.bias_dir = np.array([1.0, -0.75, 0.6])
        self.bias_dps, self.warm_dps, self.noise_dps = bias_dps, warm_dps, noise_dps
        self.acc_noise, self.mag_noise = acc_noise, mag_noise
        self.bias_on = True
        self.noise_on = True

    @staticmethod
    def temperature(t_abs):
        return 28 + 10 * (1 - math.exp(-t_abs / 60))

    def bias(self, t_abs):
        """Gyro bias (rad/s) at time t_abs since the board was switched on."""
        warm = (self.temperature(t_abs) - 28) / 10
        return self.bias_dir * (self.bias_dps + self.warm_dps * warm) * DEG

    def gyro(self, omega, t_abs):
        w = np.atleast_1d(np.asarray(omega, float)).copy()
        if self.bias_on:
            w += self.bias(t_abs)
        if self.noise_on:
            w += self.rng.normal(0, self.noise_dps * DEG, self.dims)
        return w

    def accel_flat(self, theta, acc):
        """Specific force (acceleration minus gravity's pull, so +g up at rest) in head coordinates."""
        fx, fy = acc[0], acc[1] + G
        c, s = math.cos(theta), math.sin(theta)
        a = np.array([fx * c + fy * s, -fx * s + fy * c])
        if self.noise_on:
            a += self.rng.normal(0, self.acc_noise, 2)
        return a

    def accel(self, R, acc):
        a = R.T @ (acc + G * UP)
        if self.noise_on:
            a += self.rng.normal(0, self.acc_noise, 3)
        return a

    def magnet(self, R, field_world):
        m = R.T @ field_world
        if self.noise_on:
            m += self.rng.normal(0, self.mag_noise, 3)
        return m


def magnet_nearby_field():
    """The Earth's field plus a magnet on the desk to the left: the horizontal part turns by ~35°."""
    f = M_WORLD + np.array([-0.35, 0.0, 0.0])
    return f / np.linalg.norm(f)


# ---------------------------------------------------------------- given parts of the filters

def tilt_terms(R, acc):
    """[LaValle §V] The accelerometer reading brought into the world frame (â = R ã), the tilt
    error φ (angle between â and up) and the horizontal tilt axis t = â × up / ‖â × up‖."""
    a_hat = R @ acc
    n = np.linalg.norm(a_hat)
    if n < 1e-9:
        return a_hat, 0.0, np.array([1.0, 0.0, 0.0])
    u = a_hat / n
    phi = math.acos(max(-1.0, min(1.0, float(u @ UP))))
    ax = np.cross(u, UP)
    na = np.linalg.norm(ax)
    t = ax / na if na > 1e-12 else np.array([1.0, 0.0, 0.0])
    return a_hat, phi, t


def magnet_heading(R, mag):
    """[LaValle §VI] Heading of the magnetometer reading, brought into the world frame and
    projected onto the horizontal plane."""
    m = R @ mag
    return math.atan2(m[0], m[2])


MAG_GAIN = 0.002


def yaw_correction(R, mag, ref):
    """[LaValle eq. 12] Turn the estimate about the vertical by a small part of the heading error."""
    return rot(UP, -MAG_GAIN * wrap(magnet_heading(R, mag) - ref)) @ R


class ShakeGate:
    """[LaValle §V] Skip the accelerometer when it jumps: sudden changes are motion, not drift.
    A sample is rejected when its length is far from g or it is far from its recent average."""

    def __init__(self, dims, tau=0.2, max_len=1.0, max_jump=1.5):
        self.lp = None
        self.k = DT / tau
        self.max_len, self.max_jump = max_len, max_jump
        self.dims = dims

    def ok(self, acc):
        if self.lp is None:
            self.lp = acc.copy()
        jump = np.linalg.norm(acc - self.lp)
        self.lp = self.lp + self.k * (acc - self.lp)
        return abs(np.linalg.norm(acc) - G) < self.max_len and jump < self.max_jump


def alpha_tau(alpha):
    """Time constant (s) of the complementary filter at 1000 Hz: τ = Δt α / (1 − α)."""
    return math.inf if alpha >= 1 else DT * alpha / (1 - alpha)


# ---------------------------------------------------------------- magnets

# id: (mathtext, plain, kind). kind: 'var', 'op', 'paren', 'comma', 'post', 'fn'
MAGNETS = {
    'theta': (r'$\theta$', 'θ', 'var'),
    'omega': (r'$\tilde{\omega}$', 'ω̃', 'var'),
    'dt': (r'$\Delta t$', 'Δt', 'var'),
    'ax': (r'$\tilde{a}_x$', 'ã_x', 'var'),
    'ay': (r'$\tilde{a}_y$', 'ã_y', 'var'),
    'alpha': (r'$\alpha$', 'α', 'var'),
    'oma': (r'$(1-\alpha)$', '(1 − α)', 'var'),
    'thg': (r'$\theta_{gyro}$', 'θ_gyro', 'var'),
    'tha': (r'$\theta_{acc}$', 'θ_acc', 'var'),
    'R': (r'$R$', 'R', 'var'),
    'dR': (r'$\Delta R$', 'ΔR', 'var'),
    'T': (r'$T$', 'T', 'var'),
    'phi': (r'$\varphi$', 'φ', 'var'),
    't': (r'$t$', 't', 'var'),
    '+': (r'$+$', '+', 'op'),
    '-': (r'$-$', '−', 'op'),
    '*': (r'$\cdot$', '·', 'op'),
    '/': (r'$\div$', '÷', 'op'),
    '(': (r'$($', '(', 'paren'),
    ')': (r'$)$', ')', 'paren'),
    ',': (r'$,$', ',', 'comma'),
    'T^': (r'$^\top$', 'ᵀ', 'post'),
    'atan2': (r'$\mathrm{atan2}$', 'atan2', 'fn'),
    'Rot': (r'$\mathrm{Rot}$', 'Rot', 'fn'),
    'Exp': (r'$\mathrm{Exp}$', 'Exp', 'fn'),
}
SYMBOLS = {k for k, v in MAGNETS.items() if v[2] != 'var'}


def magnet_tex(m):
    return MAGNETS[m][0]


def plain(toks):
    return ' '.join(MAGNETS[m][1] for m in toks)


# units: (rad, s, m) exponents
U1 = (0, 0, 0)
RAD = (1, 0, 0)
RADPS = (1, -1, 0)
SEC = (0, 1, 0)
MPS2 = (0, -2, 1)
UNIT_NAME = {RAD: 'an angle (rad)', RADPS: 'an angular velocity (rad/s)', SEC: 'a time (s)',
             MPS2: 'an acceleration (m/s²)', U1: 'a plain number'}


def unit_name(u):
    if u in UNIT_NAME:
        return UNIT_NAME[u]
    parts = []
    for name, e in zip(('rad', 's', 'm'), u):
        if e:
            parts.append(name if e == 1 else f'{name}^{e}')
    return f"something in {'·'.join(parts)}"


def umul(a, b, sign=1):
    return tuple(x + sign * y for x, y in zip(a, b))


# variables per tab: id -> (shape, unit, description). shape: 's' scalar, 'v' 3-vector, 'm' rotation
VARS_FLAT = {
    'theta': ('s', RAD, 'the previous angle estimate'),
    'omega': ('s', RADPS, 'the gyro reading'),
    'dt': ('s', SEC, 'the time step, 1 ms'),
    'ax': ('s', MPS2, "the accelerometer's x reading"),
    'ay': ('s', MPS2, "the accelerometer's y reading"),
    'alpha': ('s', U1, 'the blend weight'),
    'oma': ('s', U1, 'one minus the blend weight'),
    'thg': ('s', RAD, "M1's update: θ + ω̃·Δt"),
    'tha': ('s', RAD, "M2's angle from gravity"),
}
VARS_RIFT = {
    'R': ('m', U1, 'the previous orientation estimate'),
    'dR': ('m', U1, 'the rotation the gyro measured in this step'),
    'omega': ('v', RADPS, 'the gyro reading (head frame)'),
    'T': ('s', SEC, 'the latency, sensor to eye'),
    'dt': ('s', SEC, 'the time step, 1 ms'),
    'phi': ('s', RAD, 'the tilt error'),
    't': ('v', U1, 'the tilt axis'),
    'alpha': ('s', U1, 'the blend weight'),
    'oma': ('s', U1, 'one minus the blend weight'),
}


class ParseError(Exception):
    pass


class Expr:
    """A parsed and type-checked line: .fn(env) evaluates it; .shape, .unit; .warn is a unit
    mismatch that still runs (shown in amber)."""

    def __init__(self, fn, shape, unit, warn, text):
        self.fn, self.shape, self.unit, self.warn, self.text = fn, shape, unit, warn, text


def parse(tokens, variables):
    """Parse magnets into an Expr. Raises ParseError with a sentence for students."""
    toks = list(tokens)
    pos = [0]
    warns = []

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        m = peek()
        pos[0] += 1
        return m

    def name(m):
        return MAGNETS[m][1]

    def node(fn, shape, unit, text):
        return (fn, shape, unit, text)

    def atom():
        m = peek()
        if m is None:
            raise ParseError('something is missing at the end' if toks else 'empty')
        kind = MAGNETS[m][2]
        if kind == 'var':
            take()
            if m not in variables:
                raise ParseError(f'{name(m)} is not known in this tab')
            shape, unit, _ = variables[m]
            return node(lambda env, m=m: env[m], shape, unit, name(m))
        if m == '(':
            take()
            e = expr()
            if peek() != ')':
                raise ParseError('a "(" is never closed')
            take()
            return node(e[0], e[1], e[2], f'({e[3]})')
        if m == '-':
            take()
            e = unary()
            if e[1] == 'm':
                raise ParseError('a rotation cannot be negated; its inverse is Rᵀ')
            f = e[0]
            return node(lambda env, f=f: -f(env), e[1], e[2], f'−{e[3]}')
        if kind == 'fn':
            take()
            return call(m)
        if m == ')':
            raise ParseError('a ")" closes nothing')
        if m == ',':
            raise ParseError('a "," outside atan2( , )')
        if m == 'T^':
            raise ParseError('ᵀ goes right after a matrix')
        raise ParseError(f'{name(m)} needs something on its left')

    def call(f):
        if peek() != '(':
            raise ParseError(f'{f} needs its brackets')
        take()
        args = [expr()]
        while peek() == ',':
            take()
            args.append(expr())
        if peek() != ')':
            raise ParseError(f'{f}( ) is never closed')
        take()
        if f == 'atan2':
            if len(args) != 2:
                raise ParseError('atan2 takes two numbers')
            (fa, sa, ua, ta), (fb, sb, ub, tb) = args
            if sa != 's' or sb != 's':
                raise ParseError('atan2 takes two numbers')
            if ua != ub:
                warns.append(f'atan2({ta}, {tb}): {unit_name(ua)} and {unit_name(ub)} — use two of a kind')
            return node(lambda env: math.atan2(fa(env), fb(env)), 's', RAD, f'atan2({ta}, {tb})')
        if f == 'Rot':
            if len(args) != 2 or args[0][1] != 'v' or args[1][1] != 's':
                raise ParseError('Rot(axis, angle)')
            (fa, _, _, ta), (fb, _, ub, tb) = args
            if ub != RAD:
                warns.append(f'the angle {tb} is {unit_name(ub)}, not an angle')
            return node(lambda env: rot(fa(env), fb(env)), 'm', U1, f'Rot({ta}, {tb})')
        if f == 'Exp':
            if len(args) != 1 or args[0][1] != 'v':
                raise ParseError('Exp( ) takes a rotation vector: axis × angle')
            fa, _, ua, ta = args[0]
            if ua != RAD:
                warns.append(f'{ta} is {unit_name(ua)}; Exp needs a rotation vector in rad')
            return node(lambda env: exp_so3(fa(env)), 'm', U1, f'Exp({ta})')
        raise ParseError(f'unknown function {f}')

    def postfix():
        e = atom()
        while peek() == 'T^':
            take()
            if e[1] != 'm':
                raise ParseError('ᵀ transposes a matrix; this is not one')
            f = e[0]
            e = node(lambda env, f=f: f(env).T, 'm', e[2], f'{e[3]}ᵀ')
        if peek() is not None and MAGNETS[peek()][2] in ('var', 'fn') or peek() == '(':
            raise ParseError(f'put an operator between {e[3]} and {name(peek())}')
        return e

    def unary():
        return postfix()

    def term():
        e = unary()
        while peek() in ('*', '/'):
            op = take()
            b = unary()
            e = combine(op, e, b)
        return e

    def expr():
        e = term()
        while peek() in ('+', '-'):
            op = take()
            b = term()
            e = combine(op, e, b)
        return e

    def combine(op, a, b):
        fa, sa, ua, ta = a
        fb, sb, ub, tb = b
        text = f'{ta} {name(op)} {tb}'
        if op in '+-':
            if sa == 'm' or sb == 'm':
                raise ParseError('rotations do not add up; chain them with ·')
            if sa != sb:
                raise ParseError(f'{ta} {name(op)} {tb}: a number and a vector')
            if ua != ub:
                warns.append(f'{ta} {name(op)} {tb}: {unit_name(ua)} {"plus" if op == "+" else "minus"} '
                             f'{unit_name(ub)} — the units do not match')
            if op == '+':
                return node(lambda env: fa(env) + fb(env), sa, ua, text)
            return node(lambda env: fa(env) - fb(env), sa, ua, text)
        if op == '/':
            if sb != 's':
                raise ParseError('divide by a number only')
            if sa == 'm':
                raise ParseError('a rotation divided by a number is no rotation')
            return node(lambda env: fa(env) / fb(env), sa, umul(ua, ub, -1), text)
        # multiplication
        if sa == 'm' and sb == 'm':
            return node(lambda env: fa(env) @ fb(env), 'm', U1, text)
        if sa == 'm' and sb == 'v':
            return node(lambda env: fa(env) @ fb(env), 'v', ub, text)
        if 'm' in (sa, sb):
            raise ParseError('a number times a rotation matrix is no rotation')
        if sa == 'v' and sb == 'v':
            raise ParseError('two vectors cannot be multiplied here')
        return node(lambda env: fa(env) * fb(env), 'v' if 'v' in (sa, sb) else 's', umul(ua, ub), text)

    if not toks:
        raise ParseError('empty')
    e = expr()
    if peek() is not None:
        m = peek()
        raise ParseError('a ")" closes nothing' if m == ')' else f'{name(m)} is left over')
    return Expr(e[0], e[1], e[2], warns[0] if warns else None, e[3])


# ---------------------------------------------------------------- stages

@dataclass
class Stage:
    key: str
    tab: int
    title: str
    lhs: str                # mathtext of the left-hand side
    body: list              # fixed magnet ids and slot numbers
    palette: list
    want: tuple             # (shape, unit) of the whole line
    goal: str
    do: str
    see: str
    think: str
    slides: str

    @property
    def n_slots(self):
        return sum(1 for b in self.body if isinstance(b, int))


STAGES = [
    Stage('M0', 1, 'Meet the IMU', '', [], [], None,
          'know what the gyro and the accelerometer measure',
          'Press H for "your head": drag in the view to tilt the head, right-drag to shake it '
          'sideways. Then press H again to watch the recorded head.',
          'The dial is the gyro (ω̃, in °/s; grey: the truth); the green arrow on the head is the accelerometer (ã), drawn '
          'in the head\'s own frame. Both update 1000 times a second.',
          'Held still, which way does ã point? Why up and not down? What happens to it while you shake?',
          '4–9, 21'),
    Stage('M1', 1, 'Dead reckoning', r'$\theta \;\leftarrow$', [0],
          ['theta', 'omega', 'dt', 'ax', 'ay', '+', '-', '*'], ('s', RAD),
          'turn the gyro\'s angular velocity into an angle',
          'Fill the update: the new angle from the old one and the gyro reading (slide 28).',
          'The blue needle follows the grey head, then slides away. Switch the bias (B) and the noise (N) '
          'on and off: a ramp or a growing wobble. K calibrates (holds still for 2 s first); » fast-forwards.',
          'With a bias of 2°/s, how far off after one minute? Why does calibrating slow the drift without '
          'stopping it?',
          '10–17, 28'),
    Stage('M2', 1, 'Tilt from gravity', r'$\theta_{acc} \;=$', ['atan2', '(', 0, ',', 1, ')'],
          ['ax', 'ay', 'omega', 'theta', '-'], ('s', RAD),
          'read the angle off the direction of gravity',
          'Fill the two slots of atan2 (slide 29): which reading goes across, which goes up?',
          'The orange needle: no drift at all, but noisy. While the head shakes it swings by tens of degrees.',
          'What does the accelerometer measure while the head shakes? Why is the grey head not tilting then?',
          '21–25, 29–30'),
    Stage('M3', 1, 'Complementary filter', r'$\theta \;\leftarrow$', [0],
          ['alpha', 'oma', 'thg', 'tha', 'theta', '+', '-', '*', '(', ')'], ('s', RAD),
          'keep the gyro\'s short term and the accelerometer\'s long term',
          'Blend θ_gyro (your M1) and θ_acc (your M2) with the weight α (slide 32). Then move the α slider '
          'until the error stays below 1.5° for 20 s.',
          'The red needle stays on the grey head: smooth and no drift. α = 1 is the blue needle, α = 0 the '
          'orange one; τ is the time over which the accelerometer wins. Try "reject shakes" [Oculus §V].',
          'At 1000 Hz, which α gives τ = 1 s? What goes wrong with τ = 10 ms? With τ = 100 s?',
          '31–33'),
    Stage('M4', 2, 'Body frame', r'$R \;\leftarrow$', [0],
          ['R', 'dR', '*', 'T^'], ('m', U1),
          'integrate the gyro in 3D',
          'Chain the old orientation R and this step\'s rotation ΔR = Exp(ω̃ Δt). The order matters.',
          'The headset (blue) and its grey ghost; on the right, what the eye sees. The recorded head looks '
          'around, looks down (12 s), then turns. Bias and noise start off; switch them on (B, N) to see the drift.',
          'Both orders agree for the first 12 s. Why? In which frame does the gyro measure ω? [Oculus §III]',
          '11, 28'),
    Stage('M5', 2, 'Tilt correction', r'$R \;\leftarrow$', ['Rot', '(', 't', ',', 0, ')', '*', 'R'],
          ['oma', 'alpha', 'phi', '*', '-'], ('m', U1),
          'use gravity to level the horizon',
          'Fill the angle: turn the estimate a fraction of the tilt error φ about the tilt axis t, the 3D '
          'version of M3 (θ_gyro + (1 − α)(θ_acc − θ_gyro)).',
          'â = R ã (orange) should point up (green); φ is the angle between them, t (violet) the axis. The '
          'tilt error drops to 0; the yaw error keeps drifting. Try the magnetometer (M) and a magnet (G).',
          'Turn your head about the vertical (H, drag): does â change? Which drift can gravity never fix?',
          '26, 34–41, 46–47'),
    Stage('M6', 2, 'Predict', r'$R_{pred} \;=$', ['R', '*', 'Exp', '(', 0, ')'],
          ['omega', 'T', 'dt', '*', 'alpha'], ('m', U1),
          'show the world where the head will be when the picture reaches the eye',
          'Fill the rotation vector: keep turning at the gyro\'s rate for the latency T [Oculus §VII]. '
          'Then move the latency slider (L / Shift+L).',
          'Without M6 the room lags behind the ghost on every turn; with it, it sticks. The table compares '
          'your errors with the paper\'s.',
          'Why does prediction overshoot when the head reverses? What would assuming constant angular '
          'acceleration add?',
          'Oculus §VII'),
]
STAGE = {s.key: s for s in STAGES}
STAGE_ORDER = [s.key for s in STAGES]


def line_tokens(stage, slots):
    """The whole line as magnets: the fixed parts with the slots' magnets put in."""
    out = []
    for b in stage.body:
        out.extend(slots[b] if isinstance(b, int) else [b])
    return out


def variables_for(stage):
    return VARS_FLAT if stage.tab == 1 else VARS_RIFT


def compile_line(stage, slots):
    """(Expr or None, message). The message says what is missing or wrong in words."""
    if any(not s for s in slots):
        missing = sum(1 for s in slots if not s)
        return None, 'empty' if missing == len(slots) else 'fill the other slot'
    try:
        e = parse(line_tokens(stage, slots), variables_for(stage))
    except ParseError as err:
        return None, str(err)
    want_shape, want_unit = stage.want
    if e.shape != want_shape:
        what = {'s': 'a number', 'v': 'a vector', 'm': 'a rotation'}
        return None, f'this gives {what[e.shape]}; the line needs {what[want_shape]}'
    if e.warn is None and e.unit != want_unit and stage.tab == 1:
        e.warn = f'this is {unit_name(e.unit)}; the angle needs rad'
    return e, None


# ---------------------------------------------------------------- checks (salted hashes)

_SALT = 'csc478-imu'


def _probes(stage, n=4):
    rng = np.random.default_rng(1000 + STAGE_ORDER.index(stage.key))
    envs = []
    for _ in range(n):
        if stage.tab == 1:
            env = {k: float(rng.uniform(0.2, 1.5)) for k in VARS_FLAT}
            env['alpha'] = float(rng.uniform(0.6, 0.95))
            env['oma'] = 1 - env['alpha']
        else:
            env = {'R': exp_so3(rng.normal(0, 1, 3)), 'dR': exp_so3(rng.normal(0, 0.5, 3)),
                   'omega': rng.normal(0, 2, 3), 'T': float(rng.uniform(0.02, 0.06)),
                   'dt': float(rng.uniform(0.001, 0.005)), 'phi': float(rng.uniform(0.05, 0.5)),
                   't': (lambda v: v / np.linalg.norm(v))(rng.normal(0, 1, 3)),
                   'alpha': float(rng.uniform(0.6, 0.95))}
            env['oma'] = 1 - env['alpha']
        envs.append(env)
    return envs


def digest(stage, slots):
    """Salted hash of what the line computes on fixed random inputs (so equivalent forms agree)."""
    e, _ = compile_line(stage, slots)
    if e is None:
        return None
    try:
        vals = [np.asarray(e.fn(env), float).ravel() for env in _probes(stage)]
    except (ArithmeticError, ValueError, FloatingPointError):
        return None
    q = np.concatenate(vals)
    if not np.all(np.isfinite(q)):
        return None
    sig = ','.join(str(int(v)) for v in np.round(q * 1e4))
    return hashlib.sha256(f'{_SALT}|{stage.key}|{sig}'.encode()).hexdigest()[:16]


_ACCEPT = {
    'M1': {'c8372151c9044a8e'},
    'M2': {'53a8c16b60ad4f66'},
    'M3': {'50056214bda99e07'},
    'M4': {'4f02a8b1a57ce33c'},
    'M5': {'a4cf41b3657758b1'},
    'M6': {'750ae73702c6b6a6'},
}
_HINTS = {   # digests of sensible wrong lines -> what to look at
    '6b85ad7a9b0acb14': 'the needle turns the wrong way: which sign does ω̃ have when the head tilts left?',
    '8060b0f175588637': 'a thousand times too fast: ω̃ is per second, one step lasts Δt',
    'c6ddf663a9003e68': 'this is only the change in one step; add it to the previous angle',
    'e45d4868b1af18ca': '90° off while upright: atan2(across, up) — which reading is across the head?',
    'de20454a87e171fa': 'α and (1 − α) swapped: with α near 1 this is the noisy orange needle',
    '9a05c40ed1faa327': 'θ alone leaves the gyro out: use θ_gyro, the angle after the gyro step',
    '4d4c7b03cc0ae71f': "right while looking straight ahead, wrong after looking down: ΔR is measured in the "
                        "head's frame, so it acts first (on the right)",
    'fe9638c234c3d003': 'the headset turns backwards: ΔRᵀ undoes the step',
    '8f528a622c6d7c0e': 'the minus pushes â away from up: the tilt runs away',
    '2bedf69243ae0aaa': 'α ≈ 1: almost the whole tilt error is removed every sample, so the horizon jitters '
                        'with the noise',
    '0400c0c60bd00e4b': 'the whole error every sample: the horizon jitters with the accelerometer noise',
    'b3e99bb5b4f4c6c3': 'this predicts one sample (1 ms) ahead; the picture is late by T',
}


def check(stage, slots):
    """(True / False / None, hint). None while the line does not run."""
    d = digest(stage, slots)
    if d is None:
        return None, None
    if d in _ACCEPT.get(stage.key, ()):
        return True, None
    return False, _HINTS.get(d)


ANSWER_FILES = {'M1': 'u1_dead_reckoning', 'M2': 'u2_accel_angle', 'M3': 'u3_complementary',
                'M4': 'u4_body_frame', 'M5': 'u5_tilt_correction', 'M6': 'u6_predict'}


def released_answers():
    """Stages whose answer file is in answers/ (released with git add -f). IMU_NO_ANSWERS=1
    ignores them (the student view)."""
    out = {}
    if os.environ.get('IMU_NO_ANSWERS'):
        return out
    for key, stem in ANSWER_FILES.items():
        p = ANSWERS_DIR / f'{stem}.json'
        if p.exists():
            try:
                out[key] = [list(s) for s in json.loads(p.read_text())[key]]
            except (OSError, ValueError, KeyError, TypeError):
                pass
    return out


# ---------------------------------------------------------------- trackers (run your lines)

def _finite(x):
    return bool(np.all(np.isfinite(x)))


class FlatTracker:
    """Tab 1: runs your M1 (blue, gyro only), M2 (orange) and M3 (red, fused) on every sample."""

    def __init__(self):
        self.reset()

    def reset(self, theta0=0.0):
        self.th_g = theta0
        self.th_a = theta0
        self.th_f = theta0
        self.gate = ShakeGate(2)
        self.rejected = False

    def step(self, w, acc, m1, m2, m3, alpha, reject):
        env = {'theta': self.th_g, 'omega': w, 'dt': DT, 'ax': acc[0], 'ay': acc[1],
               'alpha': alpha, 'oma': 1 - alpha}
        try:
            if m1 is not None:
                v = m1.fn(env)
                self.th_g = v if math.isfinite(v) and abs(v) < 1e6 else 0.0
            if m2 is not None:
                self.th_a = m2.fn(env)
            if m1 is not None and m2 is not None and m3 is not None:
                env['theta'] = self.th_f
                thg = m1.fn(env)
                ok = self.gate.ok(acc)
                self.rejected = reject and not ok
                if self.rejected:
                    v = thg
                else:
                    env.update(thg=thg, tha=self.th_a)
                    v = m3.fn(env)
                self.th_f = v if math.isfinite(v) and abs(v) < 1e6 else 0.0
        except (ArithmeticError, ValueError):
            pass


class RiftTracker:
    """Tab 2: your M4 alone (blue, gyro only) and M4 + M5 (+ the given magnetometer correction)
    (red), and the history M6 needs."""

    HIST = 128                                # samples kept for prediction (> 60 ms)

    def __init__(self):
        self.reset()

    def reset(self, R0=None):
        R0 = np.eye(3) if R0 is None else R0.copy()
        self.R_g = R0.copy()
        self.R_f = R0.copy()
        self.gate = ShakeGate(3)
        self.rejected = False
        self.a_hat = UP * G
        self.a_lp = UP * G
        self.phi = 0.0
        self.axis = np.array([1.0, 0, 0])
        self.mag_ref = None
        self.n = 0
        self.hist_R = []
        self.hist_w = []

    def step(self, w, acc, mag, m4, m5, alpha, reject, use_mag):
        dR = exp_so3(w * DT)
        self.n += 1
        if m4 is None:
            return
        try:
            Rg = m4.fn({'R': self.R_g, 'dR': dR})
            Rf = m4.fn({'R': self.R_f, 'dR': dR})
            if m5 is not None:
                a_hat, phi, axis = tilt_terms(Rf, acc)
                self.a_hat, self.phi, self.axis = a_hat, phi, axis
                self.a_lp = self.a_lp + 0.02 * (a_hat - self.a_lp)
                ok = self.gate.ok(acc)
                self.rejected = reject and not ok
                if not self.rejected:
                    Rf = m5.fn({'R': Rf, 't': axis, 'phi': phi, 'alpha': alpha, 'oma': 1 - alpha})
            else:
                self.a_hat = Rf @ acc
                self.a_lp = self.a_lp + 0.02 * (self.a_hat - self.a_lp)
            if use_mag:
                if self.mag_ref is None:
                    self.mag_ref = magnet_heading(Rf, mag)
                Rf = yaw_correction(Rf, mag, self.mag_ref)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            return
        if self.n % 200 == 0:                 # keep them rotations despite rounding
            Rg, Rf = orthonormal(Rg), orthonormal(Rf)
        self.R_g = Rg if _finite(Rg) else np.eye(3)
        self.R_f = Rf if _finite(Rf) else np.eye(3)
        self.hist_R.append(self.R_f)
        self.hist_w.append(w)
        if len(self.hist_R) > self.HIST:
            del self.hist_R[0], self.hist_w[0]

    def predicted(self, m6, T, back=0):
        """M6 applied to the estimate `back` samples ago (None if not available)."""
        if back >= len(self.hist_R):
            return None
        R, w = self.hist_R[-1 - back], self.hist_w[-1 - back]
        if m6 is None:
            return R
        try:
            Rp = m6.fn({'R': R, 'omega': w, 'T': T, 'dt': DT})
        except (ArithmeticError, ValueError):
            return R
        return Rp if _finite(Rp) else R

    def past(self, back):
        return None if back >= len(self.hist_R) else self.hist_R[-1 - back]


def orthonormal(R):
    U, _, Vt = np.linalg.svd(R)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    return Q
