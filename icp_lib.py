"""Tutorial 6 - given code for the ICP demo (2026-10-15-icp_demo.py).

The bunny scans and their scenes, the blocks of the right pane (variable tiles,
the Align script's lines, the ICP loop's command blocks), the evaluator that
runs whatever the blocks say, and the hashed checks.

Conventions: points in cm, one point per row, so a cloud is an (N, 3) array.
Q is the target (fixed, blue), P the source (moves, magenta). R, t move P onto
Q: a point p goes to R p + t, which for a whole cloud is P @ R.T + t.
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANSWERS_DIR = HERE / 'answers'
BUNNY = HERE / 'assets' / 'icp_bunny.npz'

NOISE = 0.05                 # sensor noise per coordinate, cm (0.5 mm)
N_DEFAULT = 800              # points per cloud
MAX_PASSES = 100             # safety stop for the ICP loop
TAU_DEFAULT = 1e-4           # the loop's threshold tau on E, in cm^2 (0.01 mm^2)


# ---------------------------------------------------------------- geometry

def rotation(axis, deg):
    """Rotation matrix about `axis` by `deg` degrees (Rodrigues)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def is_rotation(M, tol=1e-3):
    return M.shape == (3, 3) and np.allclose(M.T @ M, np.eye(3), atol=tol) and np.linalg.det(M) > 0


def angle_of(M):
    """Rotation angle of a rotation matrix, in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))))


def axis_angle(M):
    """(axis, degrees) of a rotation matrix."""
    deg = angle_of(M)
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    if np.linalg.norm(v) < 1e-9:
        if deg < 90:
            return np.array([0.0, 0.0, 1.0]), 0.0
        w, V = np.linalg.eigh((M + M.T) / 2)          # 180 degrees: axis = eigenvector of 1
        return V[:, -1], deg
    return v / np.linalg.norm(v), deg


def random_axis(rng):
    a = rng.normal(size=3)
    return a / np.linalg.norm(a)


def closest_points(P, Q, chunk=2048):
    """Index into Q of the closest point of Q for every row of P (brute force)."""
    q2 = (Q ** 2).sum(1)
    out = np.empty(len(P), int)
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk]
        d2 = (p ** 2).sum(1)[:, None] - 2 * p @ Q.T + q2[None, :]
        out[s:s + chunk] = d2.argmin(1)
    return out


# ---------------------------------------------------------------- scenes

def load_bunny():
    d = np.load(BUNNY)
    return d['points'].astype(float), d['normals'].astype(float)


@dataclass
class SceneSpec:
    key: int
    name: str
    blurb: str
    start_deg: float
    start_shift: float           # cm
    known_pairs: bool = False
    views_deg: float = 0.0       # > 0: two scans from viewpoints this far apart
    axis: tuple = (0.3, 1.0, 0.25)


SCENES = {
    1: SceneSpec(1, 'Known pairs', 'P is Q, point for point (plus noise): the pairs are known',
                 40, 6.0, known_pairs=True),
    2: SceneSpec(2, 'Two scans', 'P and Q are sampled independently: no point has a true partner',
                 25, 3.0),
    3: SceneSpec(3, 'Far start', 'Two scans, but P starts turned far away',
                 120, 3.0, axis=(0.2, 1.0, 0.1)),
    4: SceneSpec(4, 'Two views', 'Each scan only sees the side of the bunny facing its sensor',
                 20, 2.0, views_deg=70),
}


@dataclass
class Scene:
    spec: SceneSpec
    seed: int
    Q: np.ndarray                # target, fixed
    P0: np.ndarray               # source at its start pose
    Z: np.ndarray                # true correction: P0 @ Z.T + z is where P belongs
    z: np.ndarray

    @property
    def P_true(self):
        return self.P0 @ self.Z.T + self.z


def _sample(rng, pts, nrm, n, view=None):
    idx = np.arange(len(pts))
    if view is not None:
        idx = idx[nrm @ view > 0.1]               # points facing the sensor
    pick = rng.choice(idx, min(n, len(idx)), replace=False)
    return pts[pick] + rng.normal(scale=NOISE, size=(len(pick), 3))


def make_scene(key, seed, n=N_DEFAULT, start_deg=None, rng_axis=False):
    """Scene `key` with `n` points per cloud. start_deg / rng_axis override the start pose."""
    spec = SCENES[key]
    rng = np.random.default_rng([seed, key])
    pts, nrm = load_bunny()
    if spec.known_pairs:
        Q = _sample(rng, pts, nrm, n)
        P = Q + rng.normal(scale=NOISE, size=Q.shape)
    elif spec.views_deg:
        h = np.radians(spec.views_deg / 2)
        v1 = np.array([np.sin(h), 0.0, np.cos(h)])
        v2 = np.array([-np.sin(h), 0.0, np.cos(h)])
        Q = _sample(rng, pts, nrm, n, v1)
        P = _sample(rng, pts, nrm, n, v2)
    else:
        Q = _sample(rng, pts, nrm, n)
        P = _sample(rng, pts, nrm, n)
    deg = spec.start_deg if start_deg is None else start_deg
    axis = random_axis(rng) if rng_axis else np.array(spec.axis)
    shift_dir = random_axis(rng) if rng_axis else np.array([1.0, 0.35, -0.2])
    shift = spec.start_shift * shift_dir / np.linalg.norm(shift_dir)
    # move P away from its true place about its own centroid: P0 = S (P - c) + c + shift
    S = rotation(axis, deg)
    c = P.mean(0)
    P0 = (P - c) @ S.T + c + shift
    # the true correction undoes that: P = S^T (P0 - c - shift) + c
    Z = S.T
    z = c - (c + shift) @ S          # = c - S^T (c + shift)
    return Scene(spec, seed, Q, P0, Z, z)


def pose_error(scene, A, b):
    """True error of P = P0 @ A.T + b: (rotation error in degrees or None, RMS point error in cm)."""
    P = scene.P0 @ A.T + b
    rms = float(np.sqrt(((P - scene.P_true) ** 2).sum(1).mean()))
    E = A @ scene.Z.T
    return (angle_of(E) if is_rotation(A, 1e-2) else None), rms


def mean_sq(P, Qp):
    """E = mean over pairs of ||q - p||^2 (rows of P and Qp are the pairs)."""
    return float(((Qp - P) ** 2).sum(1).mean())


# ---------------------------------------------------------------- blocks: variable tiles

# kind decides the tile's shape and which slots accept it
TILES = {   # name: (kind, plain label, mathtext)
    'P':   ('cloud', 'P', r'$P$'),
    'Q':   ('cloud', 'Q', r'$Q$'),
    'pc':  ('cloud', "p'", r"$p'$"),
    'qc':  ('cloud', "q'", r"$q'$"),
    'muP': ('point', 'mu_P', r'$\mu_P$'),
    'muQ': ('point', 'mu_Q', r'$\mu_Q$'),
    't':   ('point', 't', r'$t$'),
    'W':   ('matrix', 'W', r'$W$'),
    'U':   ('matrix', 'U', r'$U$'),
    'D':   ('matrix', 'D', r'$D$'),
    'Vt':  ('matrix', 'V^T', r'$V^\top$'),
    'R':   ('matrix', 'R', r'$R$'),
}
TILE_ORDER = ['P', 'Q', 'pc', 'qc', 'muP', 'muQ', 't', 'W', 'U', 'D', 'Vt', 'R']
KIND_NAME = {'cloud': 'a point cloud', 'point': 'a point', 'matrix': 'a 3×3 matrix'}


def tile_kind(name):
    return TILES[name][0]


# ---------------------------------------------------------------- blocks: the Align script

@dataclass(frozen=True)
class Line:
    key: str                  # slot key, also the variable(s) it defines
    step: str                 # I1 ... I4
    parts: tuple              # mathtext pieces with None where a slot goes
    kinds: tuple              # kind of each slot
    defines: tuple            # variables this line defines
    row: int                  # display row in the script
    col: int = 0              # 0 = left, 1 = right half of the row


ALIGN_LINES = [
    Line('muP', 'I1', (r'$\mu_P \leftarrow$ mean of', None), ('cloud',), ('muP',), 0, 0),
    Line('muQ', 'I1', (r'$\mu_Q \leftarrow$ mean of', None), ('cloud',), ('muQ',), 0, 1),
    Line('pc', 'I1', (r"$p' \leftarrow P\, -$", None), ('point',), ('pc',), 1, 0),
    Line('qc', 'I1', (r"$q' \leftarrow Q\, -$", None), ('point',), ('qc',), 1, 1),
    Line('W', 'I2', (r'$W \leftarrow \Sigma$', None, None, r'$^\top$'), ('cloud', 'cloud'), ('W',), 2),
    Line('svd', 'I3', (r'$U, D, V^\top \leftarrow$ SVD of', None), ('matrix',), ('U', 'D', 'Vt'), 3),
    Line('R', 'I3', (r'$R \leftarrow$', None, None), ('matrix', 'matrix'), ('R',), 4),
    Line('t', 'I4', (r'$t \leftarrow$', None, r'$-$', None, None), ('point', 'matrix', 'point'), ('t',), 5),
]
LINE = {ln.key: ln for ln in ALIGN_LINES}


# ---------------------------------------------------------------- blocks: the ICP loop

# The loop's four lines are built from word magnets and symbol magnets, one
# line per step of slide 109. A line is a list of magnet ids; parse_line turns it
# into an action, or explains why it cannot run.
WORDS = ['align', 'and', 'by', 'closest', 'decreased', 'each', 'mean', 'move', 'or', 'pair', 'pairs',
         'sum', 'to', 'with']
SYMBOLS = {   # id: (plain, mathtext)
    'P': ('P', r'$P$'), 'Q': ('Q', r'$Q$'), 'p': ('p', r'$p$'), 'q': ('q', r'$q$'),
    'R': ('R', r'$R$'), 't': ('t', r'$t$'), 'E': ('E', r'$E$'),
    'arrow': ('←', r'$\leftarrow$'), 'plus': ('+', r'$+$'), 'minus': ('−', r'$-$'),
    'comma': (',', r'$,$'), 'norm': ('‖', r'$\Vert$'), 'norm2': ('‖²', r'$\Vert^2$'),
    'inf': ('∞', r'$\infty$'), 'zero': ('0', r'$0$'), 'tau': ('τ', r'$\tau$'),
    'gt': ('>', r'$>$'), 'lt': ('<', r'$<$'),
}
SYMBOL_ORDER = ['P', 'Q', 'p', 'q', 'R', 't', 'E', 'arrow', 'plus', 'minus', 'comma', 'norm', 'norm2',
                'inf', 'zero', 'tau', 'gt', 'lt']
LOOP_SLOTS = 4


def magnet_plain(m):
    return SYMBOLS[m][0] if m in SYMBOLS else m


def magnet_tex(m):
    return SYMBOLS[m][1] if m in SYMBOLS else m


def line_text(toks):
    return ' '.join(magnet_plain(m) for m in toks)


class ParseError(Exception):
    pass


@dataclass(frozen=True)
class Action:
    kind: str                 # 'pair', 'align', 'move', 'error'
    variant: str = ''         # pair: closest / index; align: '' / inverse; error: mean / sum
    rot: bool = True          # move: apply R
    tsign: int = 1            # move: +t, -t or no t
    says: str = ''            # what the line does, in words (shown under it)

    @property
    def sig(self):
        if self.kind == 'move':
            return f'move:{int(self.rot)}:{self.tsign}'
        return f'{self.kind}:{self.variant}'


def parse_line(toks):
    """The action a line of magnets describes. Raises ParseError with a hint."""
    t = [m for m in toks if m != 'comma']                  # commas are only for reading
    if not t:
        raise ParseError('empty line')
    if t.count('arrow') > 1:
        raise ParseError('one ← per line: one thing gets a new value')
    if 'arrow' in t:
        i = t.index('arrow')
        lhs, rhs = t[:i], t[i + 1:]
        if lhs == ['E']:
            return _parse_error(rhs)
        if sorted(lhs) == ['R', 't']:
            return _parse_align(rhs)
        if lhs in (['P'], ['p']):
            return _parse_move_formula(lhs[0], rhs)
        if lhs in (['Q'], ['q']):
            raise ParseError('Q is the target: it stays where it is. Only P moves')
        if lhs in (['R'], ['t']):
            raise ParseError('R and t come out together: write R, t ← …')
        if not lhs:
            raise ParseError('nothing on the left of ←: what gets the new value?')
        raise ParseError(f'cannot give "{line_text(lhs)}" a new value')
    if t[0] == 'pair':
        return _parse_pair(t[1:])
    if t[0] == 'move':
        return _parse_move_words(t[1:])
    if t[0] == 'align' or 'align' in t:
        raise ParseError('align gives you R and t: keep them with R, t ← …')
    if 'mean' in t or 'sum' in t or 'norm' in t:
        raise ParseError('keep the result: E ← …')
    raise ParseError('start with a verb (pair, move) or with "… ←"')


def _parse_pair(t):
    if t[:1] == ['each']:
        t = t[1:]
    if any(m in ('P', 'Q') for m in t):
        raise ParseError('P and Q are whole clouds: pairs are made of single points p and q')
    if t[:1] != ['p'] and t[:1] != ['q']:
        raise ParseError('pair which points? e.g. pair each … with …')
    if t[:1] == ['q']:
        raise ParseError('every point of the cloud that moves needs a partner: start with each p')
    rest = t[1:]
    if rest[:1] != ['with']:
        raise ParseError('pair each p with …?')
    rest = rest[1:]
    if rest == ['closest', 'q']:
        return Action('pair', 'closest', says='each p gets the q closest to it')
    if rest == ['q']:
        return Action('pair', 'index', says='p₁ with q₁, p₂ with q₂, … (same index: nothing says which q)')
    if rest in (['p'], ['closest', 'p']):
        raise ParseError('a p needs a partner from the other cloud, Q')
    raise ParseError('pair each p with … which q?')


def _parse_align(rhs):
    if rhs[:1] != ['align']:
        raise ParseError('what computes R and t from the pairs? (your Align script)')
    rest = rhs[1:]
    if rest in (['pairs'], ['P', 'to', 'Q'], ['P', 'to', 'pairs'], ['p', 'to', 'q']):
        return Action('align', '', says='R, t that move P onto its partners')
    if rest in (['Q', 'to', 'P'], ['q', 'to', 'p'], ['pairs', 'to', 'P']):
        return Action('align', 'inverse', says='R, t that move the partners in Q onto P')
    raise ParseError('align what to what?')


def _parse_move_formula(x, rhs):
    """x <- sum of terms: R x, x, t (each with a sign)."""
    terms, sign, cur = [], 1, []
    if rhs[:1] in (['plus'], ['minus']):
        sign, rhs = (1 if rhs[0] == 'plus' else -1), rhs[1:]
    for m in rhs + ['plus']:
        if m in ('plus', 'minus'):
            if not cur:
                raise ParseError('a + or − needs something on both sides')
            terms.append((sign, cur))
            sign, cur = (1 if m == 'plus' else -1), []
        else:
            cur.append(m)
    rot, has_x, tsign = False, False, 0
    for sg, term in terms:
        if term in (['R', x], [x]):
            if has_x:
                raise ParseError(f'{x} appears twice on the right')
            if sg < 0:
                raise ParseError(f'−{line_text(term)} would flip the bunny inside out')
            has_x, rot = True, term[0] == 'R'
        elif term == ['t']:
            if tsign:
                raise ParseError('t appears twice')
            tsign = sg
        elif x.lower() == 'p' and any(m in ('P', 'p') for m in term) and x not in term:
            raise ParseError('use the same letter on both sides of ←')
        else:
            raise ParseError(f'"{line_text(term)}" is not something P can be moved by')
    if not has_x:
        raise ParseError(f'the new {x} has to be made from the old {x}')
    if not rot and not tsign:
        raise ParseError(f'{x} ← {x} does not move anything')
    return _move_action(rot, tsign)


def _parse_move_words(t):
    if t[:1] in (['Q'], ['q']):
        raise ParseError('Q is the target: it stays where it is. Only P moves')
    if t[:1] not in (['P'], ['p']):
        raise ParseError('move what?')
    rest = t[1:]
    if rest[:1] != ['by']:
        raise ParseError('move P by …?')
    by = rest[1:]
    if sorted(by) == ['R', 't']:
        return _move_action(True, 1)
    if by == ['R']:
        return _move_action(True, 0)
    if by == ['t']:
        return _move_action(False, 1)
    raise ParseError('move P by what?')


def _move_action(rot, tsign):
    what = {(True, 1): 'p → R p + t', (True, 0): 'p → R p (turn only)', (False, 1): 'p → p + t (shift only)',
            (True, -1): 'p → R p − t', (False, -1): 'p → p − t (shift only)'}[(rot, tsign)]
    return Action('move', rot=rot, tsign=tsign, says=f'moves every point: {what}')


def _parse_error(rhs):
    agg = rhs[0] if rhs[:1] in (['mean'], ['sum']) else None
    body = rhs[1:] if agg else rhs
    if len(body) == 5 and body[0] == 'norm' and body[4] == 'norm2' and body[2] == 'minus':
        a, b = body[1], body[3]
        if {a, b} in ({'p', 'q'}, {'P', 'Q'}):
            if agg is None:
                raise ParseError('that is one number per pair; E has to be one number for all pairs')
            how = 'mean' if agg == 'mean' else 'sum'
            return Action('error', agg, says=f'{how} over all pairs of the squared distance')
        raise ParseError('E measures how far each p is from its partner q')
    if body[:1] == ['norm'] and body[-1:] == ['norm']:
        raise ParseError('close with ‖² : E uses squared distances (slide 97)')
    raise ParseError('E ← … how far each p is from its partner q, over all pairs')



# ---- the start value of E and the loop condition (slide 109: error := inf; while (...))

START_VALUES = {'inf': float('inf'), 'zero': 0.0, 'tau': TAU_DEFAULT}


@dataclass(frozen=True)
class Init:
    value: str                # 'inf', 'zero' or 'tau'
    says: str = ''

    @property
    def sig(self):
        return f'init:{self.value}'


def parse_init(toks):
    t = [m for m in toks if m != 'comma']
    if not t:
        raise ParseError('empty line')
    if 'arrow' not in t:
        raise ParseError('give E its first value: E ← …')
    i = t.index('arrow')
    lhs, rhs = t[:i], t[i + 1:]
    if lhs != ['E']:
        raise ParseError('this line only starts E: E ← …')
    if len(rhs) == 1 and rhs[0] in START_VALUES:
        says = {'inf': 'E starts infinitely large', 'zero': 'E starts at 0',
                'tau': 'E starts at τ'}[rhs[0]]
        return Init(rhs[0], says)
    if any(m in ('mean', 'sum', 'norm') for m in rhs):
        raise ParseError('there are no pairs yet before the loop: just pick a start value')
    raise ParseError('E starts at one value')


@dataclass(frozen=True)
class Cond:
    op: str                   # 'and', 'or' or '' (a single clause)
    clauses: tuple            # each 'dec', 'gt:tau', 'lt:tau', 'gt:zero', 'lt:zero'
    says: str = ''

    @property
    def sig(self):
        return f'cond:{self.op}:' + ','.join(sorted(self.clauses))


def _clause(t):
    if t == ['E', 'decreased']:
        return 'dec'
    if len(t) == 3 and t[0] == 'E' and t[1] in ('gt', 'lt') and t[2] in ('tau', 'zero'):
        return f'{t[1]}:{t[2]}'
    if len(t) == 3 and t[2] == 'E' and t[1] in ('gt', 'lt') and t[0] in ('tau', 'zero'):
        return f'{"lt" if t[1] == "gt" else "gt"}:{t[0]}'                 # tau < E  is  E > tau
    if 'E' not in t:
        raise ParseError('the condition is about E')
    if t[:1] == ['E'] and len(t) == 1:
        raise ParseError('E what? a condition is true or false')
    raise ParseError(f'"{line_text(t)}" is not something that is true or false')


CLAUSE_WORDS = {'dec': 'E went down', 'gt:tau': 'E > τ', 'lt:tau': 'E < τ', 'gt:zero': 'E > 0', 'lt:zero': 'E < 0'}


def parse_cond(toks):
    t = [m for m in toks if m != 'comma']
    if not t:
        raise ParseError('empty line')
    if 'arrow' in t:
        raise ParseError('a condition is true or false: no ← here')
    if 'and' in t and 'or' in t:
        raise ParseError('use either and or or, not both')
    op = 'and' if 'and' in t else ('or' if 'or' in t else '')
    parts, cur = [], []
    for m in t + [op or 'END']:
        if m == op or m == 'END':
            if not cur:
                raise ParseError(f'"{op}" needs a condition on both sides')
            parts.append(_clause(cur))
            cur = []
        else:
            cur.append(m)
    if len(set(parts)) < len(parts):
        raise ParseError('the same condition twice')
    says = f' {op} '.join(CLAUSE_WORDS[c] for c in parts)
    return Cond(op, tuple(parts), f'repeat while {says}')


def cond_holds(cond, st, tau=TAU_DEFAULT):
    """(bool, explanation) of the loop condition in state st. Before the first pass
    there is no earlier E to compare with, so 'E decreased' counts as true."""
    def fmt(v):
        return '∞' if not np.isfinite(v) else f'{v * 100:.4g} mm²'
    vals, why = [], []
    for c in cond.clauses:
        if c == 'dec':
            if st.passes == 0:
                v, w = True, 'E decreased: yes (nothing to compare with yet)'
            else:
                v = st.E < st.E_prev
                w = f'E decreased: {"yes" if v else "no"} ({fmt(st.E_prev)} → {fmt(st.E)})'
        else:
            cmp, ref = c.split(':')
            r = tau if ref == 'tau' else 0.0
            v = st.E > r if cmp == 'gt' else st.E < r
            w = f'{CLAUSE_WORDS[c]}: {"yes" if v else "no"} (E = {fmt(st.E)})'
        vals.append(v)
        why.append(w)
    ok = all(vals) if cond.op != 'or' else any(vals)
    return ok, f' {cond.op or ""} '.join(why).replace('  ', ' ')

# ---------------------------------------------------------------- steps, checks, answers

@dataclass(frozen=True)
class Step:
    key: str
    title: str
    slides: str
    lines: tuple = ()         # Align lines; empty for I5


STEPS = [
    Step('I1', 'centroids', '100', ('muP', 'muQ', 'pc', 'qc')),
    Step('I2', 'cross-cov.', '102', ('W',)),
    Step('I3', 'rotation', '102–103', ('svd', 'R')),
    Step('I4', 'translation', '103', ('t',)),
    Step('I5', 'the ICP loop', '109'),
]
STEP = {s.key: s for s in STEPS}
ANSWER_FILES = {'I1': 'i1_centre', 'I2': 'i2_cross_covariance', 'I3': 'i3_rotation',
                'I4': 'i4_translation', 'I5': 'i5_icp_loop'}
_SALT = 'csc478-icp'
_ACCEPT = {   # sha256(_SALT|step|signature)[:16] of the accepted fillings
    'I1': {'7fcf351940ab0015'},
    'I2': {'6cafdfa5999cd6be', '855e44bace46fa74', '90f65b5532d7cba0'},
    'I3': {'11bc50e424afef20'},
    'I4': {'e67b2017ad963f0b'},
    'I5': {'cccc79aa54f6a350', '4cbcb32924602e89'},   # mean or sum: same minimum
    'I5stop': {'48740df086f7dc40'},   # E ← ∞; while E decreased and E > τ (either order)
}
EQUIVALENT_W = {('qc', 'P'), ('Q', 'pc')}     # same W as q' p'^T: centred points sum to 0


def empty_fill():
    """Slot contents: one list per Align line plus the loop; None = empty."""
    f = {ln.key: [None] * len(ln.kinds) for ln in ALIGN_LINES}
    f['loop'] = [[] for _ in range(LOOP_SLOTS)]
    f['init'], f['cond'] = [], []
    return f


def step_slots(step):
    return list(step.lines) if step.lines else ['init', 'cond', 'loop']


def step_filled(step, fill):
    if not step.lines:
        return bool(fill['init']) and bool(fill['cond']) and all(len(ln) > 0 for ln in fill['loop'])
    return all(x is not None for k in step_slots(step) for x in fill[k])


def step_empty(step, fill):
    if not step.lines:
        return not fill['init'] and not fill['cond'] and all(len(ln) == 0 for ln in fill['loop'])
    return all(x is None for k in step_slots(step) for x in fill[k])


def parse_loop(fill):
    """[(Action or None, error message or None)] for the four lines of the loop."""
    out = []
    for toks in fill['loop']:
        try:
            out.append((parse_line(toks), None))
        except ParseError as e:
            out.append((None, None if not toks else str(e)))
    return out


def parse_stop(fill):
    """(Init or None, error), (Cond or None, error) for the start value and the condition."""
    out = []
    for key, fn in (('init', parse_init), ('cond', parse_cond)):
        try:
            out.append((fn(fill[key]), None))
        except ParseError as e:
            out.append((None, None if not fill[key] else str(e)))
    return out


def body_signature(fill):
    return '|'.join(a.sig if a else 'invalid' for a, _ in parse_loop(fill))


def stop_signature(fill):
    return '|'.join(a.sig if a else 'invalid' for a, _ in parse_stop(fill))


def signature(step, fill):
    if not step.lines:     # I5: what the lines do, not how they are worded
        return body_signature(fill) + '||' + stop_signature(fill)
    return '|'.join(','.join(str(x) for x in fill[k]) for k in step_slots(step))


def _hash(step_key, sig):
    return hashlib.sha256(f'{_SALT}|{step_key}|{sig}'.encode()).hexdigest()[:16]


def check_loop(fill):
    """I5 in two parts: (the four lines inside the loop right, start value and condition right)."""
    return (_hash('I5', body_signature(fill)) in _ACCEPT['I5'],
            _hash('I5stop', stop_signature(fill)) in _ACCEPT['I5stop'])


def check_step(step, fill):
    if step.key == 'I5':
        return all(check_loop(fill))
    return _hash(step.key, signature(step, fill)) in _ACCEPT[step.key]


def released_answers():
    """Steps whose answer file is in answers/ (released with git add -f)."""
    out = {}
    for key, stem in ANSWER_FILES.items():
        p = ANSWERS_DIR / f'{stem}.json'
        if p.exists():
            try:
                out[key] = json.loads(p.read_text())[key]
            except (OSError, ValueError, KeyError):
                pass
    return out


# ---------------------------------------------------------------- the evaluator

class EvalError(Exception):
    def __init__(self, msg, where=None):
        super().__init__(msg)
        self.where = where    # line key or loop slot index


DEFINED_IN = {v: ln for ln in ALIGN_LINES for v in ln.defines}


def _value(name, env, line):
    if name in env:
        return env[name]
    src = DEFINED_IN[name]
    raise EvalError(f'{TILES[name][1]} is not computed yet here (it comes from {src.step}, below)', line.key)


def run_align(fill, P, Qp, upto=None):
    """Run the Align script on row-paired clouds P, Qp.

    Returns env: every variable computed, in order, until the first line that is
    not filled (env['_stop'] = that line's key) or the end (env['_stop'] = None).
    Raises EvalError for a line that cannot be computed. `upto` stops after that line.
    """
    env = {'P': P, 'Q': Qp, '_stop': None}
    for ln in ALIGN_LINES:
        args = fill[ln.key]
        if any(a is None for a in args):
            env['_stop'] = ln.key
            return env
        vals = [_value(a, env, ln) for a in args]
        if ln.key in ('muP', 'muQ'):
            env[ln.key] = vals[0].mean(0)
        elif ln.key == 'pc':
            env['pc'] = P - vals[0]
        elif ln.key == 'qc':
            env['qc'] = Qp - vals[0]
        elif ln.key == 'W':
            env['W'] = vals[0].T @ vals[1]          # sum over rows of a_k b_k^T
        elif ln.key == 'svd':
            U, s, Vt = np.linalg.svd(vals[0])
            env.update(U=U, D=np.diag(s), Vt=Vt, sigma=s)
        elif ln.key == 'R':
            env['R'] = vals[0] @ vals[1]
        elif ln.key == 't':
            env['t'] = vals[0] - vals[1] @ vals[2]
        if not all(np.all(np.isfinite(env[v])) for v in ln.defines):
            raise EvalError('the result is not finite', ln.key)
        if ln.key == upto:
            env['_stop'] = 'upto'
            return env
    return env


def align_ready(fill):
    return all(step_filled(STEP[k], fill) for k in ('I1', 'I2', 'I3', 'I4'))


@dataclass
class IcpState:
    """The variables of the ICP script: the current P, its pairs, R, t, E."""
    P: np.ndarray
    idx: np.ndarray = None          # Q[idx] are the partners of the rows of P
    R: np.ndarray = None
    t: np.ndarray = None
    E: float = float('inf')
    E_prev: float = float('inf')
    passes: int = 0
    errors: list = field(default_factory=list)   # E after each pass
    env: dict = None                # the last Align run (for the live values)


def pair_index(n_p, n_q):
    return np.arange(n_p) % n_q


def run_action(act, st, Q, fill, where=None):
    """Execute the action of one line on the state. Returns a short description, or
    ('move', R, t) for a move (the caller animates it and then applies it)."""
    if act.kind == 'pair':
        if act.variant == 'closest':
            st.idx = closest_points(st.P, Q)
            return 'paired each p with its closest q'
        st.idx = pair_index(len(st.P), len(Q))
        return 'paired p_k with q_k (same index)'
    if act.kind == 'align':
        if st.idx is None:
            raise EvalError('no pairs yet: something has to pair the points first', where)
        if not align_ready(fill):
            raise EvalError('align uses your Align script (tab 1), and it is not complete yet', where)
        if act.variant == 'inverse':
            env = run_align(fill, Q[st.idx], st.P)
        else:
            env = run_align(fill, st.P, Q[st.idx])
        st.env, st.R, st.t = env, env['R'], env['t']
        return f'aligned: R turns {angle_of(st.R):.2f}°' if is_rotation(st.R) else 'aligned (R is not a rotation!)'
    if act.kind == 'move':
        if st.R is None:
            raise EvalError('R, t are not computed yet: nothing to move by', where)
        R = st.R if act.rot else np.eye(3)
        t = act.tsign * st.t
        check_move(st.P, R, t, where)
        return ('move', R, t)
    if act.kind == 'error':
        if st.idx is None:
            raise EvalError('no pairs yet: E needs pairs', where)
        E = mean_sq(st.P, Q[st.idx])
        st.E_prev, st.E = st.E, (E * len(st.P) if act.variant == 'sum' else E)
        return f'E = {st.E * 100:.3f} mm²'
    raise EvalError(f'unknown action {act}', where)


FLY_OFF = 1e3                # cm: a move that sends P this far away is refused


def check_move(P, R, t, where=None):
    """Refuse a move that would send P off to (near) infinity, e.g. when R is not a rotation."""
    P2 = P @ R.T + t
    if not np.all(np.isfinite(P2)) or np.abs(P2).max() > FLY_OFF:
        why = 'R is not a rotation' if not is_rotation(R) else 't is huge'
        raise EvalError(f'this move would send P off to infinity: {why}', where)
    return P2


# ---------------------------------------------------------------- reference run (verification)

def reference_icp(scene, fill, tau=TAU_DEFAULT, P=None):
    """Run the ICP script headless with the given fill. Returns the final state and (A, b)."""
    st = IcpState(P=(scene.P0 if P is None else P).copy())
    A, b = np.eye(3), np.zeros(3)
    (init, e1), (cond, e2) = parse_stop(fill)
    if init is None or cond is None:
        raise EvalError(e1 or e2 or 'empty line')
    st.E = START_VALUES[init.value]
    while st.passes < MAX_PASSES and cond_holds(cond, st, tau)[0]:
        for act, err in parse_loop(fill):
            if act is None:
                raise EvalError(err or 'empty line')
            r = run_action(act, st, scene.Q, fill)
            if isinstance(r, tuple):
                _, R, t = r
                st.P = st.P @ R.T + t
                A, b = R @ A, R @ b + t
        st.passes += 1
        st.errors.append(st.E)
    return st, A, b


def answer_fill():
    """The full answer key as a fill (instructor checkout only)."""
    f = empty_fill()
    for key, d in released_answers().items():
        for k, v in d.items():
            f[k] = [list(x) if isinstance(x, list) else x for x in v]
    return f
