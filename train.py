"""train.py -- deep-supervision training for Task M / Task S, plus eval.

Each layer emits an English word; we supervise that word (process supervision).
The inter-layer message is committed-to-words + paraphrased, so the only thing a
layer inherits from the one below is the noised English message.
"""
import torch
import torch.nn.functional as F

from tasks import make_batch_M, make_batch_S, RPOS
from model import _tok_to_class_logits


def _tau(step, total, hi=1.0, lo=0.3):
    return hi + (lo - hi) * (step / max(1, total))


def train_M(model, device, steps=1500, B=256, n_layers=4, lr=3e-3, log=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        ids, lt, _ = make_batch_M(B, n_layers, device)
        outs = model.forward_layers(ids, tau=_tau(s, steps))   # list of (B,T,V)
        loss = 0.0
        for l, o in enumerate(outs):
            cl = _tok_to_class_logits(o[:, RPOS, :])
            loss = loss + F.cross_entropy(cl, lt[:, l])
        loss = loss / len(outs)
        opt.zero_grad(); loss.backward(); opt.step()
        if log and s % max(1, steps // 6) == 0:
            log(f"    M step {s:4d}  loss {loss.item():.3f}")
    return eval_M(model, device, n_layers)


@torch.no_grad()
def eval_M(model, device, n_layers=4, B=2000, reps=3):
    model.eval()
    acc = 0.0
    for _ in range(reps):
        ids, _, final = make_batch_M(B, n_layers, device)
        pred = _tok_to_class_logits(model.forward(ids, tau=0.3)[:, RPOS, :]).argmax(-1)
        acc += (pred == final).float().mean().item()
    return acc / reps


def train_S(model, device, steps=1500, B=256, n_layers=4, lr=3e-3, log=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        ids, lt, _ = make_batch_S(B, n_layers, device)
        outs = model.forward_layers(ids, tau=_tau(s, steps))
        loss = sum(F.cross_entropy(o[:, RPOS, :], lt[:, l])
                   for l, o in enumerate(outs)) / len(outs)
        opt.zero_grad(); loss.backward(); opt.step()
        if log and s % max(1, steps // 6) == 0:
            log(f"    S step {s:4d}  loss {loss.item():.3f}")
    return eval_S(model, device, n_layers)


@torch.no_grad()
def eval_S(model, device, n_layers=4, B=2000, reps=3):
    model.eval()
    acc = 0.0
    for _ in range(reps):
        ids, _, w0 = make_batch_S(B, n_layers, device)
        pred = model.forward(ids, tau=0.3)[:, RPOS, :].argmax(-1)
        acc += (pred == w0).float().mean().item()
    return acc / reps
