"""Answer to T1 (Tutorial 7)."""
import torch


def torus_sdf(p, R, r):
    q = torch.stack([p[..., :2].norm(dim=-1) - R, p[..., 2]], dim=-1)
    return q.norm(dim=-1) - r
