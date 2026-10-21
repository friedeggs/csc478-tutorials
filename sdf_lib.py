"""Tutorial 7 - given code for the SDF demo (2026-10-22-sdf_demo.py).

Cameras and depth maps, the pose-and-scale wrapper that turns your torus_sdf
into a model PyTorch can fit, the sphere tracer that draws any SDF, the CSG
magnets and their evaluator, the parametric mug, and marching cubes with an
empty table of cases (you draw them in the demo).

Conventions: lengths in cm, z points up, one point per row, so a set of
points is an (N, 3) tensor. An SDF maps (..., 3) points to (...) distances:
negative inside, zero on the surface, positive outside.
"""
import importlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ANSWERS_DIR = HERE / 'answers'
MUG_OBJ = HERE / 'assets' / 'mug.obj'
torch.set_default_dtype(torch.float32)

ANSWERS = {'torus_sdf': ('T1', 'answers.s1_torus_sdf'), 'fit': ('T2', 'answers.s2_fit')}
JSON_ANSWERS = {'C1': 's3_csg_ops', 'C2': 's3_csg_ops', 'C3': 's3_csg_ops', 'C4': 's4_mug', 'X': 's5_mc_cases'}
answers_used = set()          # names whose released answer was called (for the demo's chips)

TORUS_R, TORUS_r = 3.0, 1.0   # the torus in its own frame: ring radius R, tube radius r
NOISE = 0.03                  # depth noise, cm (0.3 mm)
N_FIT = 3000                  # points used for fitting
SUCCESS_CM = 0.1              # chamfer distance below this: the fit has succeeded (1 mm)


def answer(name):
    """Return the released answer for the TODO `name`.

    Answers are released during the tutorial as files in answers/ (git pull).
    Until then this raises NotImplementedError. Setting the environment
    variable SDF_NO_ANSWERS=1 ignores released answers."""
    tag, module = ANSWERS[name]
    if os.environ.get('SDF_NO_ANSWERS'):
        raise NotImplementedError(f'{tag} {name}() is a TODO')
    if module not in sys.modules:
        importlib.invalidate_caches()   # notice files added since the last try
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name not in (module, 'answers'):
            raise
        raise NotImplementedError(f'{tag} {name}() is a TODO') from None
    answers_used.add(name)
    return getattr(mod, name)


def released_answers():
    """Sidebar steps whose answer file is in answers/ (released with git add -f)."""
    out = {}
    if os.environ.get('SDF_NO_ANSWERS'):
        return out
    for key, stem in JSON_ANSWERS.items():
        p = ANSWERS_DIR / f'{stem}.json'
        if p.exists():
            try:
                data = json.loads(p.read_text())
                out[key] = data if key == 'X' else data[key]
            except (OSError, ValueError, KeyError):
                pass
    return out


# ---------------------------------------------------------------- rotations

def skew(w):
    z = torch.zeros((), dtype=w.dtype)
    return torch.stack([torch.stack([z, -w[2], w[1]]), torch.stack([w[2], z, -w[0]]),
                        torch.stack([-w[1], w[0], z])])


def rotvec_matrix(w):
    """Rotation matrix of the axis-angle vector w (angle = |w| in radians); differentiable, also at 0."""
    return torch.linalg.matrix_exp(skew(w))


def rotvec_of(axis, deg):
    a = np.asarray(axis, float)
    return a / np.linalg.norm(a) * math.radians(deg)


def axis_angle(w):
    w = np.asarray(w, float)
    ang = np.linalg.norm(w)
    if ang < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return w / ang, math.degrees(ang)


# ---------------------------------------------------------------- cameras

class Camera:
    """A pinhole camera at `pos` looking at `target`, z up. Depth maps store z-depth (along the view axis)."""

    def __init__(self, pos, target, width, height, fov_deg):
        self.pos = np.asarray(pos, float)
        f = np.asarray(target, float) - self.pos
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1.0])
        if np.linalg.norm(r) < 1e-6:
            r = np.array([1.0, 0, 0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.fwd, self.right, self.up = f, r, u
        self.W, self.H = width, height
        self.focal = 0.5 * width / math.tan(math.radians(fov_deg) / 2)

    def resized(self, width, height):
        c = Camera.__new__(Camera)
        c.__dict__.update(self.__dict__)
        c.W, c.H = width, height
        c.focal = self.focal * width / self.W
        return c

    def rays(self):
        """(H*W, 3) unit directions and their dot with the view axis (to turn ray length into z-depth)."""
        v, u = np.mgrid[0:self.H, 0:self.W]
        a = (u + 0.5 - self.W / 2) / self.focal
        b = (v + 0.5 - self.H / 2) / self.focal
        d = self.fwd[None] + a.reshape(-1, 1) * self.right[None] - b.reshape(-1, 1) * self.up[None]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return d / n, (1 / n[:, 0])

    def project(self, X):
        """Pixel coordinates (N, 2) and z-depth (N,) of points X (N, 3)."""
        Y = np.atleast_2d(X) - self.pos
        z = Y @ self.fwd
        zs = np.maximum(z, 1e-6)
        x = self.W / 2 + self.focal * (Y @ self.right) / zs
        y = self.H / 2 - self.focal * (Y @ self.up) / zs
        return np.stack([x, y], 1), z

    def backproject(self, depth):
        """Surface points (N, 3) of the pixels of a z-depth map that saw something (finite depth)."""
        d, cos = self.rays()
        z = depth.reshape(-1)
        ok = np.isfinite(z)
        return self.pos[None] + d[ok] * (z[ok] / cos[ok])[:, None]


def ring_cameras(center, radius=24.0, size=112, fov=36.0, views=None):
    """Six depth cameras around the object; views = (azimuth, elevation) pairs in degrees.

    Default: five around at 30 degrees up and one looking down at 70 degrees.
    """
    c = np.asarray(center, float)
    cams = []
    views = views or [(a, 30.0) for a in range(15, 375, 72)] + [(200.0, 70.0)]
    for az, el in views:
        a, e = math.radians(az), math.radians(el)
        pos = c + radius * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        cams.append(Camera(pos, c, size, size, fov))
    return cams


# ---------------------------------------------------------------- primitives

def reference_torus(p, R, r):
    """The torus SDF, for making the scene and the self-test (T1 asks you to write it)."""
    q = torch.stack([p[..., :2].norm(dim=-1) - R, p[..., 2]], dim=-1)
    return q.norm(dim=-1) - r


def cylinder_sdf(p, radius, z0, z1):
    """A solid cylinder about the z axis, from height z0 to z1."""
    dr = p[..., :2].norm(dim=-1) - radius
    zc, hh = (z0 + z1) / 2, (z1 - z0) / 2
    dz = (p[..., 2] - zc).abs() - hh
    outside = torch.stack([dr.clamp(min=0), dz.clamp(min=0)], -1).norm(dim=-1)
    return outside + torch.maximum(dr, dz).clamp(max=0)


def sphere_sdf(p, c, radius):
    return (p - torch.as_tensor(c, dtype=p.dtype)).norm(dim=-1) - radius


# ---------------------------------------------------------------- models

class Placed(nn.Module):
    """shape(p) placed in the world: p = S⁻¹ Rᵀ (x − t), scaled back by min(s).

    The parameters are the rotation (axis-angle, 3 numbers), the translation (3)
    and, if a scale is given, the log of the scale along each axis of the
    shape's own frame (3). An uneven scale bends distances: see forward().
    """

    def __init__(self, shape, rotvec=(0, 0, 0), t=(0, 0, 0), scale=None):
        super().__init__()
        self.shape = shape
        self.rotvec = nn.Parameter(torch.tensor(rotvec, dtype=torch.float32))
        self.t = nn.Parameter(torch.tensor(t, dtype=torch.float32))
        self.log_s = None if scale is None else nn.Parameter(torch.tensor(np.log(scale), dtype=torch.float32))

    def R(self):
        return rotvec_matrix(self.rotvec)

    def local(self, x):
        p = (x - self.t) @ self.R()
        if self.log_s is not None:
            p = p / self.log_s.exp()
        return p

    def forward(self, x):
        p = self.local(x)
        d = self.shape(p)
        if self.log_s is None:
            return d
        if not (torch.is_grad_enabled() and p.requires_grad):
            return d * self.log_s.exp().min()             # a distance bound: enough for drawing
        # For fitting: the first-order distance f / |grad f| in the world, so the loss
        # cannot shrink just by squashing the shape (which would shrink min(s)).
        (g,) = torch.autograd.grad(d.sum(), p, create_graph=True)
        return d / (g / self.log_s.exp()).norm(dim=-1).clamp(min=1e-6)

    def pose_numbers(self):
        with torch.no_grad():
            axis, deg = axis_angle(self.rotvec.numpy())
            s = None if self.log_s is None else self.log_s.exp().numpy().copy()
            return dict(t=self.t.numpy().copy(), axis=axis, deg=deg, s=s)


def torus_model(torus_fn, rotvec=(0, 0, 0), t=(0, 0, 0), scale=(1, 1, 1)):
    """Your torus (R = 3 cm, r = 1 cm in its own frame) with a pose and a 3-axis scale to fit."""
    return Placed(lambda p: torus_fn(p, TORUS_R, TORUS_r), rotvec, t, scale)


TORUS_VIEWS = [(a, 35.0 if k % 2 == 0 else -35.0) for k, a in enumerate(range(10, 370, 60))]   # above and below
TORUS_TRUE = dict(rotvec=tuple(rotvec_of((0.5, -0.4, 0.75), 35.0)), t=(2.0, -1.0, 1.5), scale=(1.4, 0.9, 0.7))


# ---------------------------------------------------------------- sphere tracing

@torch.no_grad()
def sphere_trace(sdf, origin, dirs, t_near, t_far, steps=80, eps=2e-3, bound=None):
    """March each ray x = origin + t d by the SDF's value until it reaches the surface.

    origin (3,), dirs (N, 3) numpy. Returns t (N,), inf where the ray missed.
    Works with distance bounds too (min/max CSG, uneven scale), just slower.
    bound = (centre, radius): a sphere around the shape; rays that miss it are skipped.
    """
    d = torch.as_tensor(dirs, dtype=torch.float32)
    o = torch.as_tensor(origin, dtype=torch.float32).expand_as(d)
    t = torch.full((len(d),), float(t_near))
    alive = torch.arange(len(d))
    if bound is not None:
        oc = torch.as_tensor(np.asarray(origin, float) - np.asarray(bound[0], float), dtype=torch.float32)
        b = d @ oc
        disc = b * b - (oc @ oc - float(bound[1]) ** 2)
        enter = -b - disc.clamp(min=0).sqrt()
        alive = torch.nonzero(disc > 0)[:, 0]
        t[alive] = torch.maximum(enter[alive], torch.tensor(float(t_near)))
        t_far = min(t_far, float(np.linalg.norm(oc.numpy())) + float(bound[1]))
    hit = torch.zeros(len(d), dtype=torch.bool)
    for _ in range(steps):
        if len(alive) == 0:
            break
        ta = t[alive]
        f = sdf(o[alive] + ta[:, None] * d[alive])
        f = torch.nan_to_num(f, nan=1e3)
        done = f.abs() < eps * (1 + 0.02 * ta)
        hit[alive[done]] = True
        t[alive] = ta + f.clamp(min=-0.5) * 0.95
        keep = ~done & (t[alive] < t_far) & (t[alive] > 0)
        alive = alive[keep]
    t[~hit] = float('inf')
    return t.numpy()


@torch.no_grad()
def sdf_normals(sdf, X, h=1e-2):
    """Unit gradients of the SDF at points X (N, 3), by central differences."""
    if len(X) == 0:
        return np.zeros((0, 3))
    x = torch.as_tensor(X, dtype=torch.float32)
    g = []
    for k in range(3):
        e = torch.zeros(3)
        e[k] = h
        g.append(sdf(x + e) - sdf(x - e))
    g = torch.stack(g, 1).numpy()
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)


def render_depth(sdf, cam, t_near=1.0, t_far=60.0, noise=0.0, rng=None, bound=None):
    """z-depth map (H, W) of an SDF seen by cam; inf where nothing is hit."""
    d, cos = cam.rays()
    t = sphere_trace(sdf, cam.pos, d, t_near, t_far, bound=bound)
    z = t * cos
    if noise and rng is not None:
        z = z + rng.normal(0, noise, z.shape) * np.isfinite(z)
    return z.reshape(cam.H, cam.W)


@dataclass
class Observations:
    """Depth maps from known cameras and the surface points they back-project to."""
    cams: list
    depths: list
    points: np.ndarray            # all back-projected points (N, 3)
    view: np.ndarray              # camera index of each point (N,)
    fit_idx: np.ndarray           # the subset used for fitting

    def fit_points(self):
        return torch.as_tensor(self.points[self.fit_idx], dtype=torch.float32)


def observe(depths, cams, seed=0, n_fit=N_FIT):
    pts, view = [], []
    for k, (cam, dm) in enumerate(zip(cams, depths)):
        P = cam.backproject(dm)
        pts.append(P)
        view.append(np.full(len(P), k))
    pts, view = np.concatenate(pts), np.concatenate(view)
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(pts), min(n_fit, len(pts)), replace=False)
    return Observations(cams, depths, pts, view, np.sort(fit_idx))


def torus_scene(seed=478):
    """The torus to find: six depth maps of TORUS_TRUE with 0.3 mm noise."""
    true = torus_model(reference_torus, **TORUS_TRUE)
    center = np.array(TORUS_TRUE['t'])
    cams = ring_cameras(center, radius=22.0, views=TORUS_VIEWS)
    rng = np.random.default_rng(seed)
    depths = [render_depth(true, c, noise=NOISE, rng=rng) for c in cams]
    return observe(depths, cams, seed), center


# ---------------------------------------------------------------- CSG magnets (tab 2)

OP_MAGNETS = ['min(', 'max(', 'abs(', 'a', 'b', ',', ')', '−', '+']
MUG_MAGNETS = ['union(', 'intersect(', 'subtract(', 'body', 'cavity', 'handle', ',', ')']
OP_ROWS = [('C1', 'union'), ('C2', 'intersect'), ('C3', 'subtract')]
OP_ARITY = {'min': 2, 'max': 2, 'abs': 1, 'union': 2, 'intersect': 2, 'subtract': 2}
PART_NAMES = ['body', 'cavity', 'handle']


class ParseError(Exception):
    pass


def parse_expr(toks, atoms):
    """Magnets -> expression tree: ('atom', x), ('neg', e), ('add', e, e), ('call', f, [e, ...]).

    Raises ParseError with a hint (never the answer) when the magnets do not
    make an expression yet.
    """
    toks = list(toks)
    if not toks:
        raise ParseError('empty')
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        pos[0] += 1
        return toks[pos[0] - 1]

    def expr():
        e = term()
        while peek() == '+':
            take()
            if peek() in (None, ',', ')'):
                raise ParseError('+ needs something on both sides')
            e = ('add', e, term())
        return e

    def term():
        t = peek()
        if t is None:
            raise ParseError('something is missing at the end')
        if t == '−':
            take()
            if peek() in (None, ',', ')'):
                raise ParseError('− needs something after it')
            return ('neg', term())
        if t in atoms:
            take()
            if peek() in atoms or (peek() or '').endswith('('):
                raise ParseError(f'{t} {peek()}: put a , between two things')
            return ('atom', t)
        if t.endswith('('):
            take()
            name = t[:-1]
            args = [expr()]
            while peek() == ',':
                take()
                args.append(expr())
            if peek() != ')':
                raise ParseError(f'{name}( needs a closing )' if peek() is None else f'{name}( … : expected , or ) before {peek()}')
            take()
            n = OP_ARITY[name]
            if len(args) != n:
                raise ParseError(f'{name}( takes {n} thing{"s" if n > 1 else ""}, separated by ,' if n > 1
                                 else f'{name}( takes one thing')
            return ('call', name, args)
        if t == ')':
            raise ParseError(') without its (')
        if t == ',':
            raise ParseError(', only goes between the two things inside ( )')
        raise ParseError(f'{t} does not belong in this line')

    e = expr()
    if pos[0] != len(toks):
        raise ParseError(f'one expression per line: {toks[pos[0]]} … is left over')
    return e


def expr_text(e):
    if e[0] == 'atom':
        return e[1]
    if e[0] == 'neg':
        return '−' + expr_text(e[1])
    if e[0] == 'add':
        return f'{expr_text(e[1])} + {expr_text(e[2])}'
    return f'{e[1]}({", ".join(expr_text(a) for a in e[2])})'


def expr_atoms(e):
    if e[0] == 'atom':
        return {e[1]}
    if e[0] == 'neg':
        return expr_atoms(e[1])
    if e[0] == 'add':
        return expr_atoms(e[1]) | expr_atoms(e[2])
    return set().union(*(expr_atoms(a) for a in e[2]))


def evaluate(e, env, ops=None):
    """Value of an expression tree; env maps atoms to tensors, ops maps union/... to trees in a, b."""
    kind = e[0]
    if kind == 'atom':
        return env[e[1]]
    if kind == 'neg':
        return -evaluate(e[1], env, ops)
    if kind == 'add':
        return evaluate(e[1], env, ops) + evaluate(e[2], env, ops)
    args = [evaluate(a, env, ops) for a in e[2]]
    name = e[1]
    if name == 'min':
        return torch.minimum(*args)
    if name == 'max':
        return torch.maximum(*args)
    if name == 'abs':
        return args[0].abs()
    return evaluate(ops[name], {'a': args[0], 'b': args[1]})


REFERENCE_OPS = {'union': ('call', 'min', [('atom', 'a'), ('atom', 'b')]),
                 'intersect': ('call', 'max', [('atom', 'a'), ('atom', 'b')]),
                 'subtract': ('call', 'max', [('atom', 'a'), ('neg', ('atom', 'b'))])}
REFERENCE_MUG = ('call', 'subtract', [('call', 'union', [('atom', 'body'), ('atom', 'handle')]), ('atom', 'cavity')])
_RNG_CHECK = np.random.default_rng(7)
_AB = torch.as_tensor(_RNG_CHECK.normal(0, 2, (2, 4000)), dtype=torch.float32)


def parse_op(toks):
    """(tree, error) for one of C1-C3; the tree may only use a and b."""
    try:
        e = parse_expr(toks, ('a', 'b'))
    except ParseError as err:
        return None, str(err)
    return e, None


def parse_mug(toks):
    try:
        e = parse_expr(toks, PART_NAMES)
    except ParseError as err:
        return None, str(err)
    missing = [p for p in PART_NAMES if p not in expr_atoms(e)]
    if missing:
        return e, f'runs, but {" and ".join(missing)} {"is" if len(missing) == 1 else "are"} not in the mug yet'
    return e, None


def _calls(e):
    if e[0] == 'call':
        yield e[0], e[1]
        for a in e[2]:
            yield from _calls(a)
    elif e[0] in ('neg', 'add'):
        for a in e[1:]:
            yield from _calls(a)


def check_op(name, tree):
    """Does the tree give the same value as the reference CSG operation for many random a, b?"""
    if tree is None:
        return False
    env = {'a': _AB[0], 'b': _AB[1]}
    return bool(torch.allclose(evaluate(tree, env), evaluate(REFERENCE_OPS[name], env), atol=1e-5))


def describe_op(tree):
    """What an operation does, in words (shown under its row)."""
    if tree is None:
        return ''
    env = {'a': _AB[0], 'b': _AB[1]}
    for name, ref in REFERENCE_OPS.items():
        if torch.allclose(evaluate(tree, env), evaluate(ref, env), atol=1e-5):
            return {'union': 'inside a or inside b: the nearer of the two surfaces',
                    'intersect': 'inside a and inside b: the farther of the two',
                    'subtract': 'inside a and outside b: a with b carved out'}[name]
    swapped = {'a': _AB[1], 'b': _AB[0]}
    if torch.allclose(evaluate(tree, env), evaluate(REFERENCE_OPS['subtract'], swapped), atol=1e-5):
        return 'inside b and outside a: b with a carved out'
    return 'a new shape: see the viewer'


# ---------------------------------------------------------------- the mug (tabs 2-4)

MUG_PARAMS = [('rb', 'body radius'), ('h', 'height'), ('w', 'wall'), ('b', 'base'),
              ('Rh', 'handle radius'), ('rh', 'handle tube'), ('zh', 'handle height')]
MUG_START = dict(rb=4.0, h=9.5, w=0.6, b=0.8, Rh=2.6, rh=0.6, zh=5.0)
MUG_RANGE = dict(rb=(2.5, 6.0), h=(5.0, 14.0), w=(0.2, 1.5), b=(0.2, 2.0), Rh=(1.2, 4.0), rh=(0.2, 1.2),
                 zh=(2.0, 9.0))
MUG_VIEWS = [(a, 25.0) for a in range(20, 380, 90)] + [(65.0, 70.0), (245.0, -45.0)]
OPS_DEMO = {'a': ((-1.0, 0.0, 0.0), 2.0), 'b': ((1.3, 0.0, 0.9), 1.6)}    # two spheres for C1-C3


def mug_parts(p, P, torus_fn):
    """The three primitives in the mug's own frame (z up, handle towards +x)."""
    body = cylinder_sdf(p, P['rb'], 0.0, P['h'])
    cavity = cylinder_sdf(p, P['rb'] - P['w'], P['b'], P['h'] + 1.0)
    q = torch.stack([p[..., 0] - P['rb'], p[..., 2] - P['zh'], p[..., 1]], -1)   # handle ring in the xz plane
    handle = torus_fn(q, P['Rh'], P['rh'])
    return {'body': body, 'cavity': cavity, 'handle': handle}


class MugSDF(nn.Module):
    """Your mug: C4's tree over the three primitives, combined with your C1-C3, then placed (pose)."""

    def __init__(self, torus_fn, ops, tree, params=None, rotvec=(0, 0, 0), t=(0, 0, 0)):
        super().__init__()
        self.torus_fn, self.ops, self.tree = torus_fn, ops, tree
        P = dict(MUG_START, **(params or {}))
        self.p = nn.ParameterDict({k: nn.Parameter(torch.tensor(float(P[k]))) for k, _ in MUG_PARAMS})
        self.rotvec = nn.Parameter(torch.tensor(rotvec, dtype=torch.float32))
        self.t = nn.Parameter(torch.tensor(t, dtype=torch.float32))

    def R(self):
        return rotvec_matrix(self.rotvec)

    def local(self, x):
        return (x - self.t) @ self.R()

    def parts(self, x):
        return mug_parts(self.local(x), dict(self.p.items()), self.torus_fn)

    def forward(self, x):
        return evaluate(self.tree, self.parts(x), self.ops)

    def numbers(self):
        with torch.no_grad():
            axis, deg = axis_angle(self.rotvec.numpy())
            return dict({k: float(v) for k, v in self.p.items()}, t=self.t.numpy().copy(), axis=axis, deg=deg)

    def pose_params(self):
        return [self.rotvec, self.t]

    def shape_params(self):
        return list(self.p.values())

    def bound(self):
        """A sphere around the mug (for the sphere tracer)."""
        with torch.no_grad():
            P = {k: float(v) for k, v in self.p.items()}
            c_local = np.array([P['rh'] / 2, 0.0, P['h'] / 2])
            c = self.R().numpy() @ c_local + self.t.numpy()
            r = math.hypot(P['rb'] + P['Rh'] + P['rh'], P['h'] / 2 + 1.0) + 0.5
        return c, r


def check_mug(tree, ops_ok):
    """C4 is right when its tree gives the reference mug (with the reference C1-C3) at many points."""
    if tree is None:
        return False
    X = torch.as_tensor(_RNG_CHECK.uniform([-5, -5, -1], [8, 5, 11], (4000, 3)), dtype=torch.float32)
    parts = mug_parts(X, {k: torch.tensor(v) for k, v in MUG_START.items()}, reference_torus)
    return bool(torch.allclose(evaluate(tree, parts, REFERENCE_OPS), evaluate(REFERENCE_MUG, parts, REFERENCE_OPS),
                               atol=1e-5))


# ---------------------------------------------------------------- meshes

def load_obj(path):
    V, F = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith('v '):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('f '):
            F.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return np.array(V, float), np.array(F, int)


def write_obj(path, V, F, comment=''):
    lines = [f'# {c}' for c in comment.splitlines()] if comment else []
    lines += [f'v {x:.4f} {y:.4f} {z:.4f}' for x, y, z in V]
    lines += [f'f {a + 1} {b + 1} {c + 1}' for a, b, c in F]
    Path(path).write_text('\n'.join(lines) + '\n')


def rasterize_depth(V, F, cam):
    """z-depth map (H, W) of a triangle mesh seen by cam (a small numpy z-buffer); inf = background."""
    uv, z = cam.project(V)
    T = F[(z[F] > 0.1).all(1)]
    tu, tz = uv[T], z[T]                                   # (T, 3, 2), (T, 3)
    lo = np.ceil(tu.min(1) - 0.5).astype(int)
    hi = np.floor(tu.max(1) - 0.5).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [cam.W - 1, cam.H - 1])
    n = np.maximum(hi - lo + 1, 0)
    cnt = n[:, 0] * n[:, 1]
    keep = cnt > 0
    T_idx = np.repeat(np.nonzero(keep)[0], cnt[keep])
    start = np.repeat(np.cumsum(cnt[keep]) - cnt[keep], cnt[keep])
    k = np.arange(len(T_idx)) - start
    nx = n[T_idx, 0]
    px = lo[T_idx, 0] + k % nx
    py = lo[T_idx, 1] + k // nx
    a, b, c = tu[T_idx, 0], tu[T_idx, 1], tu[T_idx, 2]
    q = np.stack([px + 0.5, py + 0.5], 1)
    den = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    ok = np.abs(den) > 1e-12
    den = np.where(ok, den, 1)
    l1 = ((q[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (q[:, 1] - a[:, 1])) / den
    l2 = ((b[:, 0] - a[:, 0]) * (q[:, 1] - a[:, 1]) - (q[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / den
    l0 = 1 - l1 - l2
    inside = ok & (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
    zt = tz[T_idx]
    inv = l0 / zt[:, 0] + l1 / zt[:, 1] + l2 / zt[:, 2]      # perspective-correct depth
    depth = np.full(cam.H * cam.W, np.inf)
    np.minimum.at(depth, (py * cam.W + px)[inside], 1 / inv[inside])
    return depth.reshape(cam.H, cam.W)


def mug_scene(seed=478):
    """Six depth maps of assets/mug.obj with 0.3 mm noise, from cameras whose poses are known."""
    V, F = load_obj(MUG_OBJ)
    center = (V.min(0) + V.max(0)) / 2
    cams = ring_cameras(center, radius=26.0, views=MUG_VIEWS)
    rng = np.random.default_rng(seed)
    depths = []
    for cam in cams:
        d = rasterize_depth(V, F, cam)
        depths.append(d + rng.normal(0, NOISE, d.shape) * np.isfinite(d))
    return observe(depths, cams, seed), center, (V, F)


def chamfer(model, obs, bound=None):
    """Mean distance (cm) between the depth points and the model's surface, both ways.

    Points to surface: |f| / |grad f| at each depth point. Surface to points: the
    model is rendered from the same cameras and each of its points is matched
    with the nearest depth point.
    """
    from scipy.spatial import cKDTree
    X = torch.as_tensor(obs.points[obs.fit_idx], dtype=torch.float32).requires_grad_(True)
    f = model(X)
    (g,) = torch.autograd.grad(f.sum(), X)
    d1 = (f.detach().abs() / g.norm(dim=-1).clamp(min=1e-6)).numpy()
    Y = []
    for cam in obs.cams:
        small = cam.resized(cam.W // 2, cam.H // 2)
        Y.append(small.backproject(render_depth(model, small, bound=bound)))
    Y = np.concatenate(Y)
    if len(Y) == 0:
        return float('inf')
    d2, _ = cKDTree(obs.points).query(Y)
    return 0.5 * (float(np.mean(d1)) + float(np.mean(d2)))


# ---------------------------------------------------------------- marching cubes (tab 4)
#
# Corner i of a cell sits at offset (i & 1, i >> 1 & 1, i >> 2 & 1). A sign
# pattern ("configuration") is an 8-bit number: bit i set = corner i inside.

CORNERS = np.array([(i & 1, i >> 1 & 1, i >> 2 & 1) for i in range(8)])
EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
EDGE_INDEX = {e: k for k, e in enumerate(EDGES)}
EDGE_AXIS = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
FACES = []                                 # 6 faces, each 4 corners in cyclic order
for _a in range(3):
    _u, _v = [b for b in range(3) if b != _a]
    for _s in (0, 1):
        FACES.append([(_s << _a) | (cu << _u) | (cv << _v) for cu, cv in ((0, 0), (1, 0), (1, 1), (0, 1))])


def edge_of(a, b):
    return EDGE_INDEX[(min(a, b), max(a, b))]


def inside_corners(cfg):
    return [i for i in range(8) if cfg >> i & 1]


def crossing_edges(cfg):
    """Edges whose two corners have different signs: the surface crosses them."""
    return [k for k, (a, b) in enumerate(EDGES) if (cfg >> a & 1) != (cfg >> b & 1)]


def face_segments(cfg):
    """The segments the surface must draw on the cube's faces, as pairs of edges.

    A face with two sign changes gets one segment. A face whose inside corners
    sit diagonally (four sign changes) gets two, one around each inside corner:
    inside corners are kept apart, the same rule for both cells that share the
    face, so neighbouring cells always meet and the mesh has no cracks.
    """
    segs = set()
    for face in FACES:
        ins = [cfg >> c & 1 for c in face]
        es = [edge_of(face[k], face[(k + 1) % 4]) for k in range(4)]
        cut = [k for k in range(4) if ins[k] != ins[(k + 1) % 4]]
        if len(cut) == 2:
            segs.add(frozenset((es[cut[0]], es[cut[1]])))
        elif len(cut) == 4:
            for k in range(4):
                if ins[k]:
                    segs.add(frozenset((es[k - 1], es[k])))
    return segs


def loops(cfg):
    """The face segments joined into closed loops of edges (the borders of the surface pieces)."""
    nbr = {}
    for s in face_segments(cfg):
        a, b = tuple(s)
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    out, seen = [], set()
    for start in sorted(nbr):
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [e for e in nbr[cur] if e != prev and (e not in seen or (e == start and len(loop) > 2))]
            if not nxt or nxt[0] == start:
                break
            prev, cur = cur, nxt[0]
            loop.append(cur)
            seen.add(cur)
        out.append(loop)
    return out


def fan(loop):
    return [(loop[0], loop[k], loop[k + 1]) for k in range(1, len(loop) - 1)]


def _rotations():
    """The 24 rotations and 48 rotations + reflections of the cube, as corner permutations."""
    def perm(M):
        out = []
        for c in CORNERS:
            q = M @ (c - 0.5) + 0.5
            out.append(int(round(q[0])) | int(round(q[1])) << 1 | int(round(q[2])) << 2)
        return tuple(out)
    rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    mats = [np.eye(3, dtype=int)]
    seen = {perm(mats[0])}
    for M in mats:
        for G in (rx, ry, rz):
            N = G @ M
            if perm(N) not in seen:
                seen.add(perm(N))
                mats.append(N)
    rot = sorted(seen)
    mirror = [perm(np.diag([-1, 1, 1]) @ M) for M in mats]
    return rot, sorted(set(rot) | set(mirror))


ROTATIONS, SYMMETRIES = _rotations()


def apply_perm(g, cfg):
    return sum(1 << g[i] for i in range(8) if cfg >> i & 1)


def _orbit(cfg, group):
    out = set()
    for g in group:
        c = apply_perm(g, cfg)
        out |= {c, 255 - c}
    return out


def _class_number(cfg):
    """Lorensen's case number 0-14 (slide 191) for a configuration."""
    ins = inside_corners(cfg)
    if len(ins) > 4:
        ins = inside_corners(255 - cfg)
    n = len(ins)
    dist = lambda a, b: bin(a ^ b).count('1')  # noqa: E731
    adj = [(a, b) for i, a in enumerate(ins) for b in ins[i + 1:] if dist(a, b) == 1]
    deg = [sum(1 for e in adj if c in e) for c in ins]
    if n <= 1:
        return n
    if n == 2:
        return {1: 2, 2: 3, 3: 4}[dist(*ins)]
    if n == 3:
        return {2: 5, 1: 6, 0: 7}[len(adj)]
    if n == 4 and len(adj) == 4:
        return 8                                       # four corners of one face
    if n == 4 and 3 in deg:
        return 9                                       # a corner and its three neighbours
    if n == 4 and len(adj) == 2:
        return 10 if max(deg) == 1 else 12            # two opposite edges / an L and a far corner
    if n == 4 and len(adj) == 0:
        return 13
    return 11                                          # a path of three edges (11 and 14 are mirror images)


def _classes():
    cls = [0] * 256
    seen = {}
    for cfg in range(256):
        key = min(_orbit(cfg, ROTATIONS))
        cls[cfg] = _class_number(cfg)
        if cls[cfg] == 11:
            seen.setdefault(key, len(seen))
            if seen[key] == 1:
                cls[cfg] = 14
    return cls


CLASS_OF = _classes()
CLASS_INFO = {
    0: 'all corners on one side: nothing to draw', 1: 'one corner', 2: 'one edge (two corners)',
    3: 'two corners, diagonal on a face', 4: 'two opposite corners', 5: 'three corners of a face',
    6: 'an edge and a far corner', 7: 'three corners, none touching', 8: 'four corners of a face',
    9: 'a corner and its three neighbours', 10: 'two opposite edges', 11: 'a zig-zag of four corners',
    12: 'three corners of a face and a far corner', 13: 'four corners, none touching',
    14: 'the mirror image of 11'}
STUDENT_CLASSES = (1, 2, 5, 8, 9, 11, 14)           # drawn by hand; 14 comes free with 11 (a mirror)
GIVEN_CLASSES = (3, 4, 6, 7, 10, 12, 13)            # given: ambiguous faces, or rare on smooth shapes
MIRROR_OF = {14: 11}


def class_representative(k):
    """The configuration a class is drawn on (the answer file uses it)."""
    return min(c for c in range(256) if CLASS_OF[c] == k and bin(c).count('1') <= 4)


def map_triangles(tris, src, dst):
    """Move triangles drawn for configuration src to configuration dst of the same shape (or its mirror)."""
    for g in SYMMETRIES:
        c = apply_perm(g, src)
        if c == dst or 255 - c == dst:
            return [tuple(edge_of(g[EDGES[e][0]], g[EDGES[e][1]]) for e in t) for t in tris]
    raise ValueError('not the same class')


def check_triangles(cfg, tris):
    """(ok, message, bad) for triangles (edge triples) drawn for configuration cfg.

    Any surface is accepted whose border on the cube's faces is exactly the
    face segments: every triangle side is either such a segment (used once)
    or shared by two triangles inside the cube. bad = the offending edge pair.
    """
    need = crossing_edges(cfg)
    used = {e for t in tris for e in t}
    for t in tris:
        for e in t:
            if e not in need:
                return False, 'both corners of this edge are on the same side: the surface does not cross it', (e,)
    count = {}
    for t in tris:
        for k in range(3):
            s = frozenset((t[k], t[(k + 1) % 3]))
            count[s] = count.get(s, 0) + 1
    segs = face_segments(cfg)
    on_face = lambda s: any(all(set(EDGES[e]) <= set(f) for e in s) for f in FACES)  # noqa: E731
    for s, c in count.items():
        if c > 2:
            return False, 'three triangles share this side: the surface folds onto itself', tuple(s)
        if c == 1 and s not in segs and on_face(s):
            return False, 'this side runs along a face of the cube between the wrong points', tuple(s)
        if c == 2 and on_face(s):
            return False, ('two triangles share this side, but it lies on a face of the cube: the surface '
                           'has to end there, and the next cell carries it on'), tuple(s)
    missing = [e for e in need if e not in used]
    if missing:
        return None, f'{len(missing)} crossing point{"s" if len(missing) > 1 else ""} not used yet', tuple(missing)
    for s, c in count.items():
        if c == 1 and s not in segs:
            return False, 'this side is a loose border: the surface has a hole here', tuple(s)
    for s in segs:
        if s not in count:
            return False, 'the surface must reach the cube face between these two points', tuple(s)
    return True, 'every inside corner is cut off from every outside corner', ()


def _polygon_triangulations(poly):
    """Every triangulation of a small polygon (a list of edges), one at a time."""
    if len(poly) == 3:
        yield [tuple(poly)]
        return
    a, b = poly[0], poly[1]
    for k in range(2, len(poly)):              # the triangle on side a-b has its third corner at poly[k]
        left, right = poly[1:k + 1], [poly[0]] + poly[k:]
        for tl in (_polygon_triangulations(left) if len(left) >= 3 else [[]]):
            for tr in (_polygon_triangulations(right) if len(right) >= 3 else [[]]):
                yield [(a, b, poly[k])] + tl + tr


def triangulate(cfg):
    """Triangles for a configuration: each loop triangulated so that no diagonal lies on a cube face."""
    lps = loops(cfg)
    tris = [t for lp in lps for t in fan(lp)]
    for i, lp in enumerate(lps):
        others = [t for j, o in enumerate(lps) if j != i for t in fan(o)]
        for cand in _polygon_triangulations(lp):
            if check_triangles(cfg, others + cand)[0]:
                tris = [t for t in tris if t not in fan(lp)] + cand
                break
    return tris


def given_table():
    """Triangles for the given classes (from their loops)."""
    return {cfg: triangulate(cfg) for cfg in range(256) if CLASS_OF[cfg] in GIVEN_CLASSES or CLASS_OF[cfg] == 0}


class Grid:
    """The SDF sampled at the nodes of a regular grid over a box (in the mug's own frame)."""

    def __init__(self, sdf_local, lo, hi, h):
        self.lo, self.h = np.asarray(lo, float), float(h)
        self.n = np.maximum(np.ceil((np.asarray(hi, float) - self.lo) / h).astype(int), 1)   # cells per axis
        ax = [self.lo[k] + h * np.arange(self.n[k] + 1) for k in range(3)]
        X = np.stack(np.meshgrid(*ax, indexing='ij'), -1)
        with torch.no_grad():
            self.F = sdf_local(torch.as_tensor(X.reshape(-1, 3), dtype=torch.float32)).numpy().reshape(X.shape[:3])
        self.X = X
        cfg = np.zeros(tuple(self.n), int)
        nx, ny, nz = self.n
        for i, (dx, dy, dz) in enumerate(CORNERS):
            cfg |= (self.F[dx:dx + nx, dy:dy + ny, dz:dz + nz] < 0).astype(int) << i
        self.cfg = cfg
        self.cls = np.array(CLASS_OF)[cfg]

    def class_counts(self):
        return {k: int((self.cls == k).sum()) for k in range(15)}


class March:
    """Marching cubes, one z slab of cells at a time, stopping at configurations it has no triangles for."""

    def __init__(self, grid, table, interpolate=True):
        self.g, self.table, self.interpolate = grid, table, interpolate
        self.k = 0                         # next slab
        self.done = np.zeros(grid.cfg.shape, bool)
        self.tris = []                     # (T, 3) global edge ids, per emitted chunk
        self.stuck = None                  # (i, j, k) of the cell the march stopped at

    def known(self, cfg):
        return cfg in self.table

    def step(self):
        """Process the next slab; returns 'done', 'stuck' or 'ok'."""
        g = self.g
        if self.k >= g.n[2]:
            self.stuck = None
            return 'done'
        cf = g.cfg[:, :, self.k]
        known = np.vectorize(self.known, otypes=[bool])(cf) if cf.size else cf
        todo = known & ~self.done[:, :, self.k]
        self._emit(np.argwhere(todo), self.k)
        self.done[:, :, self.k] |= known
        miss = np.argwhere(~known)
        if len(miss):
            i, j = miss[0]
            self.stuck = (int(i), int(j), self.k)
            return 'stuck'
        self.stuck = None
        self.k += 1
        return 'done' if self.k >= g.n[2] else 'ok'

    def _emit(self, ij, k):
        out = []
        n = self.g.n + 1
        for i, j in ij:
            cfg = int(self.g.cfg[i, j, k])
            for t in self.table[cfg]:
                ids = []
                for e in t:
                    a = EDGES[e][0]
                    dx, dy, dz = CORNERS[a]
                    node = ((i + dx) * n[1] + (j + dy)) * n[2] + (k + dz)
                    ids.append(node * 3 + EDGE_AXIS[e])
                out.append(ids)
        if out:
            self.tris.append(np.array(out, np.int64))

    def holes(self, cls):
        """Cells of a class that the march has not filled yet."""
        return np.argwhere((self.g.cls == cls) & ~self.done)

    def mesh(self):
        """Vertices (V, 3) in the grid's frame and faces (T, 3), outward winding."""
        if not self.tris:
            return np.zeros((0, 3)), np.zeros((0, 3), int)
        T = np.concatenate(self.tris)
        ids, inv = np.unique(T, return_inverse=True)
        F = inv.reshape(-1, 3)
        node, axis = ids // 3, ids % 3
        n = self.g.n + 1
        i0 = np.stack([node // (n[1] * n[2]), node // n[2] % n[1], node % n[2]], 1)
        i1 = i0 + np.eye(3, dtype=int)[axis]
        Fv = self.g.F
        f0, f1 = Fv[tuple(i0.T)], Fv[tuple(i1.T)]
        x0, x1 = self.g.X[tuple(i0.T)], self.g.X[tuple(i1.T)]
        if self.interpolate:
            lam = np.clip(f0 / np.where(f0 - f1 == 0, 1e-12, f0 - f1), 0, 1)       # slide 189
        else:
            lam = np.full(len(f0), 0.5)
        V = x0 + lam[:, None] * (x1 - x0)
        # outward winding: the normal should point the way the SDF grows
        gx, gy, gz = np.gradient(Fv)
        c = V[F].mean(1)
        idx = np.clip(np.round((c - self.g.lo) / self.g.h).astype(int), 0, self.g.n)
        grad = np.stack([gx[tuple(idx.T)], gy[tuple(idx.T)], gz[tuple(idx.T)]], 1)
        nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        flip = (nrm * grad).sum(1) < 0
        F[flip] = F[flip][:, ::-1]
        return V, F


def watertight(F):
    """True when every edge of the mesh is shared by exactly two triangles."""
    if len(F) == 0:
        return False
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    _, c = np.unique(E, axis=0, return_counts=True)
    return bool((c == 2).all())


# ---------------------------------------------------------------- self-tests (python sdf_student.py)

def _test_torus(fn):
    p = torch.tensor([[5.0, 0, 0], [0, 0, 0], [3, 0, 0], [0, -3, 1], [0, 0, 2], [2, 2, 0.5]])
    want = reference_torus(p, 3.0, 1.0)
    got = fn(p, 3.0, 1.0)
    if not isinstance(got, torch.Tensor):
        return f'should return a torch tensor, got {type(got).__name__}'
    if got.shape != want.shape:
        return f'shape {tuple(got.shape)}, expected {tuple(want.shape)} (one distance per point)'
    names = ['(5, 0, 0) is 1 cm outside', '(0, 0, 0), the centre of the hole, is R - r = 2 away',
             '(3, 0, 0) is the tube centre: -r = -1', '(0, -3, 1) is on the surface: 0',
             '(0, 0, 2)', '(2, 2, 0.5)']
    for k in range(len(p)):
        if abs(float(got[k]) - float(want[k])) > 1e-4:
            return f'{names[k]}: got {float(got[k]):.3f}, expected {float(want[k]):.3f}'
    X = torch.as_tensor(np.random.default_rng(1).normal(0, 3, (500, 3)), dtype=torch.float32)
    if not torch.allclose(fn(X, torch.tensor(2.5), torch.tensor(0.7)), reference_torus(X, 2.5, 0.7), atol=1e-4):
        return 'right for R = 3, r = 1 but not for other R, r'
    return None


def _test_fit(fn):
    model = Placed(lambda p: p.norm(dim=-1) - 1.0, t=(0.6, -0.3, 0.2), scale=(1.3, 1.3, 1.3))
    d = torch.as_tensor(np.random.default_rng(2).normal(size=(400, 3)), dtype=torch.float32)
    pts = d / d.norm(dim=-1, keepdim=True)                  # the unit sphere at the origin
    calls = []
    fn(model, pts, steps=150, lr=0.05, report=lambda i, loss: calls.append((i, loss)))
    if len(calls) != 150:
        return f'report was called {len(calls)} times for 150 steps (once per step)'
    with torch.no_grad():
        err = float(model.t.norm()), float((model.log_s.exp() - 1).abs().max())
    if err[0] > 0.05 or err[1] > 0.05:
        return (f'a sphere 6 mm off should move onto the points: centre still {err[0] * 10:.1f} mm off, '
                f'loss {float(calls[0][1]):.3g} -> {float(calls[-1][1]):.3g}')
    return None


SELF_TESTS = [('T1', 'torus_sdf', _test_torus), ('T2', 'fit', _test_fit)]


def self_test(mod):
    """[(tag, name, status, detail)] with status ok / answer / todo / fail."""
    out = []
    for tag, name, test in SELF_TESTS:
        answers_used.discard(name)
        try:
            msg = test(getattr(mod, name))
        except NotImplementedError:
            out.append((tag, name, 'todo', 'still a TODO'))
            continue
        except Exception as e:     # noqa: BLE001 - shown to the student
            out.append((tag, name, 'fail', f'{type(e).__name__}: {e}'))
            continue
        status = 'answer' if name in answers_used else 'ok'
        out.append((tag, name, 'fail' if msg else status, msg or ('released answer' if status == 'answer' else 'passes')))
    return out


def print_self_test(mod):
    ok = True
    for tag, name, status, detail in self_test(mod):
        mark = {'ok': 'ok  ', 'answer': 'ans ', 'todo': 'TODO', 'fail': 'FAIL'}[status]
        print(f'{mark} {tag} {name}(): {detail}')
        ok &= status in ('ok', 'answer')
    return ok


def released_triangles(data, k, cfg):
    """The released triangles of class k (answers/s5_mc_cases.json), moved onto configuration cfg."""
    k = MIRROR_OF.get(k, k)
    entry = data.get(str(k)) if data else None
    if entry is None:
        return None
    src = sum(1 << c for c in entry['corners_inside'])
    tris = [tuple(edge_of(*pair) for pair in t) for t in entry['triangles']]
    return map_triangles(tris, src, cfg)
