"""Test LR scheduler factory."""

import torch

from src.training.schedulers import create_scheduler


def test_create_step_scheduler():
    model = torch.nn.Linear(10, 2)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = create_scheduler(optim, "step", step_size=1, gamma=0.5)
    lrs = []
    for _ in range(3):
        lrs.append(optim.param_groups[0]["lr"])
        sched.step()
    assert lrs == [0.1, 0.05, 0.025]
