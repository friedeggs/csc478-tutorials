"""Tutorial 5 - given code for the Kalman filter demo (2026-10-08-kalman_demo.py).

Roads and the scripted car, the noisy position sensor, the blanks of the right
pane (tokens, equations, shape checks), the filter that runs whatever the
blanks say, and the statistics in the mode bar.

Conventions: world coordinates in metres, x to the right, y up. The state is
x = (p_x, p_y, v_x, v_y) as a column vector; measurements y = (p_x, p_y).
Covariances: Q (process noise, given) and R = sigma_R^2 I (sensor noise).
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANSWERS_DIR = HERE / 'answers'

DT = 0.1                     # sensor period (10 Hz); the filter runs once per frame
WORLD_W, WORLD_H = 120.0, 75.0
ROAD_WIDTH = 7.0
CHI2_95 = 5.991              # 95 % region of a 2-D Gaussian: d^2 < 5.991
GATE = 9.21                  # 99 % region, the K6 gate threshold
PRIOR_SPEED, PRIOR_SPEED_STD = 10.0, 3.0   # K0: cars enter at about the speed limit
DIVERGED = 1e6               # |mu| or |Sigma| above this counts as diverged


# ---------------------------------------------------------------- roads

def catmull_rom(pts, closed, n_per=60):
    """Uniform Catmull-Rom spline through pts, sampled n_per times per segment."""
    P = np.asarray(pts, float)
    if closed:
        E, segs = np.vstack([P[-1], P, P[0], P[1]]), len(P)
    else:
        E, segs = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]]), len(P) - 1
    t = np.linspace(0, 1, n_per, endpoint=False)[:, None]
    out = []
    for i in range(segs):
        p0, p1, p2, p3 = E[i:i + 4]
        out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    if not closed:
        out.append(P[-1][None])
    return np.vstack(out)


class Road:
    """A road centreline sampled every DS metres, with the car's speed profile.

    Speed: at most v_max, at most sqrt(a_lat / curvature) in bends, and
    changing by at most a_lon m/s^2 (so the car brakes before a bend)."""
    DS = 0.25

    def __init__(self, pts, closed, v_max=14.0, a_lat=2.5, a_lon=2.0, tunnel=None):
        self.closed = closed
        raw = catmull_rom(pts, closed) if len(pts) > 2 else np.asarray(pts, float)
        if closed:
            raw = np.vstack([raw, raw[:1]])
        seg = np.hypot(*np.diff(raw, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = cum[-1]
        s = np.arange(0.0, self.length, self.DS)
        if not closed:
            s = np.append(s, self.length)
        self.s = s
        self.xy = np.c_[np.interp(s, cum, raw[:, 0]), np.interp(s, cum, raw[:, 1])]
        if closed:
            d = np.roll(self.xy, -1, axis=0) - np.roll(self.xy, 1, axis=0)
        else:
            d = np.gradient(self.xy, axis=0)
        self.tangent = d / np.linalg.norm(d, axis=1, keepdims=True)
        heading = np.unwrap(np.arctan2(self.tangent[:, 1], self.tangent[:, 0]))
        kappa = np.gradient(heading, self.DS)
        w = int(3.0 / self.DS)                       # smooth over ~3 m
        kernel = np.ones(w) / w
        if closed:
            kappa = np.convolve(np.r_[kappa[-w:], kappa, kappa[:w]], kernel, 'same')[w:-w]
        else:
            kappa = np.convolve(kappa, kernel, 'same')
        self.kappa = kappa
        v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-9)))
        n, step = len(v), 2 * a_lon * self.DS
        for _ in range(2 if closed else 1):          # accel / brake limits
            for i in range(1, n + (1 if closed else 0)):
                v[i % n] = min(v[i % n], np.sqrt(v[i - 1] ** 2 + step))
            for i in range(n - 2, -2 if closed else -1, -1):
                v[i % n] = min(v[i % n], np.sqrt(v[(i + 1) % n] ** 2 + step))
        self.speed = v
        self.tunnel = None if tunnel is None else tuple(self.nearest_s(p) for p in tunnel)

    def nearest_s(self, p):
        return float(self.s[np.argmin(np.hypot(*(self.xy - p).T))])

    def _interp(self, s, arr):
        s = np.asarray(s, float)
        if self.closed:
            s = np.mod(s, self.length)
            ss = np.append(self.s, self.length)
            arr = np.vstack([arr, arr[:1]]) if arr.ndim == 2 else np.append(arr, arr[0])
        else:
            ss = self.s
        if arr.ndim == 1:
            return np.interp(s, ss, arr)
        return np.stack([np.interp(s, ss, arr[:, j]) for j in range(arr.shape[1])], axis=-1)

    def at(self, s):
        """Position, unit tangent and speed at arc length s (arrays allowed)."""
        return self._interp(s, self.xy), self._interp(s, self.tangent), self._interp(s, self.speed)

    def in_tunnel(self, s):
        if self.tunnel is None:
            return np.zeros(np.shape(s), bool)
        a, b = self.tunnel
        s = np.mod(s, self.length) if self.closed else np.asarray(s)
        return (s >= a) & (s <= b) if a <= b else (s >= a) | (s <= b)

    def tunnel_polyline(self):
        if self.tunnel is None:
            return None
        a, b = self.tunnel
        if b < a:
            b += self.length
        return self.at(np.arange(a, b, 0.5))[0]


HIGHWAY = [(6.0, 14.0), (114.0, 60.0)]
RING = [(22, 10), (60, 9), (96, 11), (110, 22), (108, 40), (94, 47), (79, 43), (67, 50),
        (62, 63), (47, 67), (29, 63), (14, 51), (10, 30)]
TUNNEL = ((60.0, 9.0), (96.0, 11.0))

SCENES = {
    1: dict(name='Highway', road=lambda: Road(HIGHWAY, False, v_max=9.0), frames=400, q=0.3,
            outlier_p=0.0, about='straight road at constant speed: the constant velocity model is exact'),
    2: dict(name='Ring road', road=lambda: Road(RING, True), frames=900, q=6.0, outlier_p=0.0,
            about='bends and braking: the car does not move at constant velocity'),
    3: dict(name='Tunnel', road=lambda: Road(RING, True, tunnel=TUNNEL), frames=900, q=6.0,
            outlier_p=0.0, about='no measurements inside the tunnel: predict only'),
    4: dict(name='Busy street', road=lambda: Road(RING, True), frames=900, q=6.0, outlier_p=0.08,
            about='8 % of detections are false alarms 10-30 m away (extension K6)'),
}


@dataclass
class Scene:
    """One run: the true trajectory and the random numbers of the sensor.

    The noise is drawn once per seed, so moving the sigma_true knob rescales
    the same noise instead of drawing new noise."""
    key: int
    seed: int
    name: str = ''
    about: str = ''
    q_default: float = 1.0
    road: Road = None
    pos: np.ndarray = None        # (N, 2) true position
    vel: np.ndarray = None        # (N, 2) true velocity
    blind: np.ndarray = None      # (N,) True where the sensor sees nothing (tunnel)
    noise: np.ndarray = None      # (N, 2) standard normal
    outlier: np.ndarray = None    # (N, 2) offset of a false detection, or nan

    @property
    def n(self):
        return len(self.pos)

    def measurement(self, k, sigma_true):
        if self.blind[k]:
            return None
        if not np.isnan(self.outlier[k, 0]):
            return self.pos[k] + self.outlier[k]
        return self.pos[k] + sigma_true * self.noise[k]


def make_scene(key, seed):
    cfg = SCENES[key]
    road = cfg['road']()
    s, ss = 0.0, []
    for _ in range(cfg['frames']):
        ss.append(s)
        for _ in range(10):                          # midpoint rule, 10 sub-steps per frame
            h = DT / 10
            s += h * road.at(s + 0.5 * h * road.at(s)[2])[2]
        if not road.closed and s > road.length:
            break
    ss = np.array(ss)
    pos, tan, v = road.at(ss)
    rng = np.random.default_rng(seed)
    n = len(ss)
    noise = rng.standard_normal((n, 2))
    is_out = rng.random(n) < cfg['outlier_p']
    ang, rad = rng.uniform(0, 2 * np.pi, n), rng.uniform(10, 30, n)
    outlier = np.where(is_out[:, None], np.c_[rad * np.cos(ang), rad * np.sin(ang)], np.nan)
    outlier[0] = np.nan                              # the first detection starts the track
    return Scene(key, seed, cfg['name'], cfg['about'], cfg['q'], road, pos, v[:, None] * tan,
                 road.in_tunnel(ss), noise, outlier)


def q_matrix(q, dt=DT):
    """Process noise of the constant velocity model: white-noise acceleration q (m/s^2)."""
    I = np.eye(2)
    return q ** 2 * np.block([[dt ** 4 / 4 * I, dt ** 3 / 2 * I], [dt ** 3 / 2 * I, dt ** 2 * I]])


# ---------------------------------------------------------------- blanks of the right pane

TOKENS = ['A', 'At', 'H', 'Ht', 'Q', 'R', 'K', 'Sinv', 'Sp', 'mup', 'I', 'none']
CELLS = {'K1': ['0', '1', 'dt'], 'K3': ['0', '1']}
SYMBOLS = {   # plain-text label, mathtext
    'A': ('A', r'$A$'), 'At': ('Aᵀ', r'$A^{\mathrm{T}}$'), 'H': ('H', r'$H$'), 'Ht': ('Hᵀ', r'$H^{\mathrm{T}}$'),
    'Q': ('Q', r'$Q$'), 'R': ('R', r'$R$'), 'K': ('K', r'$K$'), 'Sinv': ('S⁻¹', r'$S^{-1}$'),
    'Sp': ('Σ⁻', r'$\Sigma^-$'), 'mup': ('μ⁻', r'$\mu^-$'), 'I': ('I', r'$I$'), 'none': ('∅', r'$\emptyset$'),
    'mu': ('μ', r'$\mu$'), 'Sigma': ('Σ', r'$\Sigma$'), 'y': ('y', r'$y$'), 'r': ('r', r'$r$'),
    'rT': ('rᵀ', r'$r^{\mathrm{T}}$'), 'S': ('S', r'$S$'), 'd2': ('d²', r'$d^2$'),
    '0': ('0', r'$0$'), '1': ('1', r'$1$'), 'dt': ('Δt', r'$\Delta t$'),
}
WHERE = {'A': 'K1', 'At': 'K1', 'H': 'K3', 'Ht': 'K3', 'mup': 'K2', 'Sp': 'K2', 'Sinv': 'K4',
         'K': 'K5', 'r': 'K4', 'rT': 'K4'}


@dataclass
class Equation:
    """lhs = sum of signed products. A factor is a symbol name, an int (that
    blank of the equation), or ('sum', terms) for a bracket."""
    key: str
    lhs: str
    terms: list
    shape: tuple
    n_blanks: int
    known: set                    # symbols available when this line runs


BASE = {'A', 'At', 'H', 'Ht', 'Q', 'R', 'I'}
EQUATIONS = {
    'K2a': Equation('K2a', 'mup', [(1, [0, 'mu'])], (4, 1), 1, BASE | {'mu', 'Sigma'}),
    'K2b': Equation('K2b', 'Sp', [(1, [0, 'Sigma', 1]), (1, [2])], (4, 4), 3,
                    BASE | {'mu', 'Sigma', 'mup'}),
    'K4a': Equation('K4a', 'r', [(1, ['y']), (-1, [0, 1])], (2, 1), 2, BASE | {'mup', 'Sp', 'y'}),
    'K4b': Equation('K4b', 'S', [(1, [0, 'Sp', 1]), (1, [2])], (2, 2), 3,
                    BASE | {'mup', 'Sp', 'y', 'r'}),
    'K6': Equation('K6', 'd2', [(1, ['rT', 0, 'r'])], (1, 1), 1,
                   BASE | {'mup', 'Sp', 'y', 'r', 'rT', 'Sinv'}),
    'K5a': Equation('K5a', 'K', [(1, ['Sp', 0, 1])], (4, 2), 2, BASE | {'mup', 'Sp', 'y', 'r', 'Sinv'}),
    'K5b': Equation('K5b', 'mu', [(1, ['mup']), (1, [0, 'r'])], (4, 1), 1,
                    BASE | {'mup', 'Sp', 'y', 'r', 'Sinv', 'K'}),
    'K5c': Equation('K5c', 'Sigma', [(1, [('sum', [(1, ['I']), (-1, [0, 1])]), 'Sp'])], (4, 4), 2,
                    BASE | {'mup', 'Sp', 'y', 'r', 'Sinv', 'K'}),
}


@dataclass
class Step:
    key: str
    title: str
    phase: str                    # 'predict' or 'correct'
    eqs: tuple = ()               # equation keys; empty for the matrix steps K1, K3
    n_cells: int = 0
    slides: str = ''


STEPS = [
    Step('K1', 'motion model', 'predict', n_cells=8, slides='112'),
    Step('K2', 'predict', 'predict', eqs=('K2a', 'K2b'), slides='109, 111'),
    Step('K3', 'observation model', 'correct', n_cells=8, slides='110'),
    Step('K4', 'innovation', 'correct', eqs=('K4a', 'K4b'), slides='124'),
    Step('K6', 'gate (ext.)', 'correct', eqs=('K6',), slides='124'),
    Step('K5', 'correct', 'correct', eqs=('K5a', 'K5b', 'K5c'), slides='109, 111'),
]
STEP = {s.key: s for s in STEPS}
ANSWER_FILES = {'K1': 'k1_motion_model', 'K2': 'k2_predict', 'K3': 'k3_observation',
                'K4': 'k4_innovation', 'K5': 'k5_correct', 'K6': 'k6_gate'}
_SALT = 'csc478-kalman'
_ACCEPT = {   # sha256(_SALT|step|blanks)[:16] of the accepted fillings
    'K1': {'ca7f42c388c058e9'}, 'K2': {'9b935618a19ccd36'}, 'K3': {'b50ebf72ec20edf5'},
    'K4': {'efa4c3d2473e1978'}, 'K5': {'3a6abbd5132c5784'}, 'K6': {'8ac51e89b1cb142d'},
}


def blank_slots(step):
    """(slot key, count) of the blanks of a step, e.g. [('K2a', 1), ('K2b', 3)]."""
    if step.n_cells:
        return [(step.key, step.n_cells)]
    return [(e, EQUATIONS[e].n_blanks) for e in step.eqs]


def empty_blanks():
    return {k: [None] * n for s in STEPS for k, n in blank_slots(s)}


def step_signature(step, blanks):
    return '|'.join(','.join(blanks[k]) for k, _ in blank_slots(step))


def check_step(step, blanks):
    sig = step_signature(step, blanks)
    h = hashlib.sha256(f'{_SALT}|{step.key}|{sig}'.encode()).hexdigest()[:16]
    return h in _ACCEPT[step.key]


def released_answers():
    """Steps whose answer file is in answers/ (released with git add -f)."""
    out = {}
    for key, stem in ANSWER_FILES.items():
        p = ANSWERS_DIR / f'{stem}.json'
        if p.exists():
            try:
                out[key] = json.loads(p.read_text())
            except (OSError, ValueError):
                pass
    return out


class EvalError(Exception):
    pass


def label(name):
    return SYMBOLS[name][0]


def shape_str(m):
    return f'{m.shape[0]}×{m.shape[1]}'


def _eval_terms(terms, blanks, ctx):
    """Evaluate a sum of signed products. Returns (matrix or None, label)."""
    parts = []                    # (sign, matrix or 'I', label) in the order written
    for sign, factors in terms:
        val, lab, has_ident = None, [], False
        for f in factors:
            if isinstance(f, tuple):
                m, l = _eval_terms(f[1], blanks, ctx)
                if m is None:
                    continue
                name_label = f'({l})'
            else:
                name = blanks[f] if isinstance(f, int) else f
                if name == 'none':
                    continue
                if name == 'I':                      # identity: no-op in a product
                    has_ident = True
                    continue
                if name not in ctx:
                    where = WHERE.get(name)
                    if where is None or name in ('mu', 'Sigma'):
                        raise EvalError(f'{label(name)} is not known on this line')
                    raise EvalError(f'{label(name)} is not known yet on this line; it comes from {where}')
                m, name_label = ctx[name], label(name)
            if val is None:
                val = m
            elif val.shape[1] != m.shape[0]:
                raise EvalError(f'{"".join(lab)} is {shape_str(val)} but {name_label} is {shape_str(m)}: '
                                f'cannot multiply')
            else:
                val = val @ m
            lab.append(name_label)
        if val is not None:
            parts.append((sign, val, ''.join(lab)))
        elif has_ident:
            parts.append((sign, 'I', 'I'))
    mats = [p for p in parts if not isinstance(p[1], str)]
    if not mats:
        if parts:
            raise EvalError('I on its own has no size here')
        return None, ''
    shape = mats[0][1].shape
    total, tl = None, ''
    for sign, val, lab in parts:
        if isinstance(val, str):
            if shape[0] != shape[1]:
                raise EvalError(f'cannot add I to {mats[0][2]}, which is {shape[0]}×{shape[1]} (not square)')
            val = np.eye(shape[0])
        if total is None:
            total, tl = sign * val, ('−' if sign < 0 else '') + lab
        elif total.shape != val.shape:
            raise EvalError(f'{tl} is {shape_str(total)} but {lab} is {shape_str(val)}: cannot add')
        else:
            total, tl = total + sign * val, tl + (' − ' if sign < 0 else ' + ') + lab
    return total, tl


def evaluate(eq, blanks, ctx):
    """Value of one equation with the student's tokens; raises EvalError."""
    ctx = {k: v for k, v in ctx.items() if k in eq.known}
    val, lab = _eval_terms(eq.terms, blanks, ctx)
    if val is None:
        raise EvalError(f'the right side of {label(eq.lhs)} is empty')
    if val.shape != eq.shape:
        raise EvalError(f'{label(eq.lhs)} must be {eq.shape[0]}×{eq.shape[1]} '
                        f'but {lab} is {shape_str(val)}')
    return val


def cell_matrix(step_key, cells):
    vals = {'0': 0.0, '1': 1.0, 'dt': DT}
    if step_key == 'K1':
        A = np.zeros((4, 4))
        A[:2, :2] = np.eye(2)
        A[:, 2:] = np.array([vals[c] for c in cells]).reshape(4, 2)
        return A
    return np.array([vals[c] for c in cells]).reshape(2, 4)


def _dummy_ctx(A, H):
    rng = np.random.default_rng(0)

    def spd(n):
        M = rng.standard_normal((n, n))
        return M @ M.T + n * np.eye(n)
    ctx = {'Q': spd(4), 'R': spd(2), 'mu': rng.standard_normal((4, 1)), 'Sigma': spd(4),
           'mup': rng.standard_normal((4, 1)), 'Sp': spd(4), 'y': rng.standard_normal((2, 1)),
           'r': rng.standard_normal((2, 1)), 'Sinv': spd(2), 'K': rng.standard_normal((4, 2)), 'I': None}
    ctx['rT'] = ctx['r'].T
    if A is not None:
        ctx['A'], ctx['At'] = A, A.T
    if H is not None:
        ctx['H'], ctx['Ht'] = H, H.T
    return ctx


@dataclass
class Model:
    """What the current blanks define: A, H, and the status of every step."""
    blanks: dict
    A: np.ndarray = None
    H: np.ndarray = None
    status: dict = field(default_factory=dict)    # step key -> (state, message)

    def ok(self, key):
        return self.status[key][0] == 'ok'


def build_model(blanks):
    """Check every step; status is 'todo' (no blank filled), 'partial', 'error' or 'ok'."""
    m = Model(blanks)
    for key in ('K1', 'K3'):
        cells = blanks[key]
        if all(c is not None for c in cells):
            setattr(m, 'A' if key == 'K1' else 'H', cell_matrix(key, cells))
    ctx = _dummy_ctx(m.A, m.H)
    for step in STEPS:
        slots = blank_slots(step)
        filled = [t is not None for k, _ in slots for t in blanks[k]]
        if not any(filled):
            m.status[step.key] = ('todo', '')
            continue
        if not all(filled):
            n = filled.count(False)
            m.status[step.key] = ('partial', f'{n} blank{"s" if n > 1 else ""} left')
            continue
        state = ('ok', '')
        for e in step.eqs:
            try:
                evaluate(EQUATIONS[e], blanks[e], ctx)
            except EvalError as err:
                state = ('error', f'{label(EQUATIONS[e].lhs)}: {err}')
                break
        m.status[step.key] = state
    return m


# ---------------------------------------------------------------- the filter

@dataclass
class Frame:
    k: int
    y: np.ndarray = None            # (2,) measurement, None in the tunnel
    est: np.ndarray = None          # (2,) the position estimate shown
    mu_pred: np.ndarray = None      # (4, 1) prediction (prior), if K1 + K2 run
    S_pred: np.ndarray = None
    mu: np.ndarray = None           # (4, 1) posterior (= prediction when not corrected)
    Sigma: np.ndarray = None
    corrected: bool = False
    r: np.ndarray = None            # (2, 1) innovation, if K4 runs
    S: np.ndarray = None
    d2: float = None                # r^T S^-1 r with the student's r and S
    gated: bool = False             # K6 rejected this measurement
    K: np.ndarray = None
    error: str = None               # numerical failure (divergence, singular S)


@dataclass
class Params:
    sigma_true: float = 2.5
    sigma_R: float = 2.5
    q: float = 1.0


def _finite(*arrays):
    return all(a is None or (np.all(np.isfinite(a)) and np.abs(a).max() < DIVERGED) for a in arrays)


class KalmanRun:
    """Runs the filter defined by the blanks over a scene, frame by frame."""

    def __init__(self, scene, model, params):
        self.scene, self.model, self.params = scene, model, params
        self.frames = []
        self.failed = None          # (frame, message) of the first numerical failure

    def prior(self, y):
        _, tan, _ = self.scene.road.at(0.0)
        mu = np.r_[y, PRIOR_SPEED * tan].reshape(4, 1)
        sr = self.params.sigma_R
        return mu, np.diag([sr ** 2, sr ** 2, PRIOR_SPEED_STD ** 2, PRIOR_SPEED_STD ** 2])

    def compute(self, upto):
        while len(self.frames) <= min(upto, self.scene.n - 1):
            self.frames.append(self._step(len(self.frames)))

    def _step(self, k):
        m, p = self.model, self.params
        y = self.scene.measurement(k, p.sigma_true)
        f = Frame(k, y)
        prev = self.frames[-1] if self.frames else None
        running = m.ok('K1') and m.ok('K2') and self.failed is None
        if not running:
            f.est = y if y is not None else (prev.est if prev else None)
            return f
        if k == 0:
            f.mu, f.Sigma = self.prior(y)
            f.est = f.mu[:2, 0]
            return f
        R = p.sigma_R ** 2 * np.eye(2)
        ctx = {'A': m.A, 'At': m.A.T, 'Q': q_matrix(p.q), 'R': R, 'mu': prev.mu, 'Sigma': prev.Sigma}
        if m.H is not None:
            ctx['H'], ctx['Ht'] = m.H, m.H.T
        b = m.blanks
        try:
            ctx['mup'] = f.mu_pred = evaluate(EQUATIONS['K2a'], b['K2a'], ctx)
            ctx['Sp'] = f.S_pred = evaluate(EQUATIONS['K2b'], b['K2b'], ctx)
            f.mu, f.Sigma = f.mu_pred, f.S_pred
            if y is not None and m.ok('K3') and m.ok('K4'):
                ctx['y'] = y.reshape(2, 1)
                ctx['r'] = f.r = evaluate(EQUATIONS['K4a'], b['K4a'], ctx)
                f.S = evaluate(EQUATIONS['K4b'], b['K4b'], ctx)
                ctx['rT'] = f.r.T
                try:
                    ctx['Sinv'] = np.linalg.inv(f.S)
                    f.d2 = float((f.r.T @ ctx['Sinv'] @ f.r)[0, 0])
                except np.linalg.LinAlgError:
                    raise EvalError('S is singular, so S⁻¹ does not exist')
                if m.ok('K6'):
                    f.gated = float(evaluate(EQUATIONS['K6'], b['K6'], ctx)[0, 0]) > GATE
                if m.ok('K5') and not f.gated:
                    ctx['K'] = f.K = evaluate(EQUATIONS['K5a'], b['K5a'], ctx)
                    f.mu = evaluate(EQUATIONS['K5b'], b['K5b'], ctx)
                    f.Sigma = evaluate(EQUATIONS['K5c'], b['K5c'], ctx)
                    f.corrected = True
            if not _finite(f.mu, f.Sigma, f.K, f.S):
                raise EvalError('the estimate diverged (numbers blew up)')
        except (EvalError, np.linalg.LinAlgError, FloatingPointError) as e:
            f.error = str(e)
            f.mu = f.Sigma = f.mu_pred = f.S_pred = f.K = f.S = f.r = f.d2 = None
            f.corrected = f.gated = False
            self.failed = (k, f.error)
            f.est = y
            return f
        f.est = f.mu[:2, 0]
        return f


def mahalanobis2(d, C):
    """d^T C^-1 d for a 2-vector, or None if C is not positive definite."""
    C = 0.5 * (C + C.T)
    try:
        if np.linalg.eigvalsh(C).min() <= 1e-12:
            return None
        return float(d @ np.linalg.solve(C, d))
    except np.linalg.LinAlgError:
        return None


def stats(run, upto):
    """RMSE of the raw measurements and of the estimate, 95 % coverage, mean NIS."""
    sc = run.scene
    me, ee, cov, nis = [], [], [], []
    for f in run.frames[:upto + 1]:
        p = sc.pos[f.k]
        if f.y is not None:
            me.append(np.sum((f.y - p) ** 2))
        if f.est is not None:
            ee.append(np.sum((f.est - p) ** 2))
        if f.Sigma is not None:
            d2 = mahalanobis2(p - f.mu[:2, 0], f.Sigma[:2, :2])
            if d2 is not None:
                cov.append(d2 < CHI2_95)
        if f.d2 is not None:
            nis.append(f.d2)

    def rms(a):
        return float(np.sqrt(np.mean(a))) if a else None
    return dict(meas_rmse=rms(me), est_rmse=rms(ee),
                coverage=float(np.mean(cov)) if cov else None,
                nis=float(np.mean(nis)) if nis else None)
