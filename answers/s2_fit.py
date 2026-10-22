"""Answer to T2 (Tutorial 7)."""
import torch


def fit(model, points, steps=300, lr=0.05, report=print):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(steps):
        loss = model(points).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        report(i, loss.item())
