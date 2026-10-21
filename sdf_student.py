"""Tutorial 7 - your code: a signed distance function and the fit that moves it.

Fill in T1 and T2. Each stub's last line, `return answer(...)(...)`, calls the
released answer once your instructor pushes it (git pull), and raises
NotImplementedError until then. Replace that line with your own code.

    python sdf_student.py            runs the self-tests (T1, T2)
    python 2026-10-22-sdf_demo.py    the demo; it reloads this file when you save

Conventions (same as sdf_lib.py): lengths in cm, points are torch tensors of
shape (..., 3), an SDF returns shape (...): negative inside, 0 on the surface.
"""
import torch  # noqa: F401  (you will need it)

from sdf_lib import answer


def torus_sdf(p, R, r):
    """T1. Signed distance from points p (..., 3) to a torus around the z axis.

    The torus is a circle of radius R in the xy plane, thickened into a tube
    of radius r. Hint: first find the distance from p to the circle (in the
    plane through p and the z axis it is a 2D distance to the point
    (R, 0)), then subtract r.
    Returns (...). R and r may be floats or 0-d tensors.
    """
    # TODO T1
    return answer('torus_sdf')(p, R, r)


def fit(model, points, steps=300, lr=0.05, report=print):
    """T2. Fit model (an nn.Module SDF) to surface points (N, 3) with Adam.

    Every point was seen on the surface, so the model should be 0 there:
    minimise the mean of model(points) squared. After each optimiser step
    call report(i, loss_value) with the step number and the loss as a float
    (the demo draws the model at that moment).
    """
    # TODO T2
    return answer('fit')(model, points, steps, lr, report)


if __name__ == '__main__':
    import sys

    import sdf_lib
    ok = sdf_lib.print_self_test(sys.modules[__name__])
    sys.exit(0 if ok else 1)
