"""scale.py -- scaling sweep for the English-bottleneck LM.

Usage:
  python scale.py --axis depth --values 2,4,6,8,12 --rho 1.0 --steps 1000 --seeds 2
  python scale.py --axis width --values 32,64,128,256 --rho 1.0 --steps 1000 --seeds 2

Measures, at each scale point, ERT capability (Task M meaning + Task S surface),
the continuous-baseline capability, and the covert channel capacity (optimal +
learned) through the full stack. Writes scale_<axis>.json.

depth: n_layers grows AND hops grow with it (hops == n_layers), so deeper = more
       reasoning steps AND more paraphrase bottlenecks to cross.
width: hidden size d grows, depth fixed at 4.
"""
import os, sys, json, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch

from vocab import build_embeddings, V, N_CLASS
from paraphrase import ParaphraseChannel
from model import EnglishResidualTransformer, ContinuousBaseline
from train import train_M, train_S
from capacity import optimal_reader_capacity, learned_collusion_capacity

DEV = os.environ.get("EBL_DEVICE", "cpu")


def ert(E, L, rho):
    p = ParaphraseChannel(rho=rho, device=DEV).to(DEV)
    return EnglishResidualTransformer(E, L, paraphrase=p, seq_len=4).to(DEV)


def run(axis, values, rho, steps, seeds):
    out = {"axis": axis, "rho": rho, "steps": steps, "seeds": seeds,
           "chance_M": 1.0 / N_CLASS, "points": {}}
    for v in values:
        d = v if axis == "width" else 64
        L = v if axis == "depth" else 4
        accM, accS, bM = [], [], []
        for sd in range(seeds):
            torch.manual_seed(sd); np.random.seed(sd)
            E = build_embeddings(d, 0.5).to(DEV)
            accM.append(train_M(ert(E, L, rho), DEV, steps, n_layers=L))
            accS.append(train_S(ert(E, L, rho), DEV, steps, n_layers=L))
            b = ContinuousBaseline(E, L, seq_len=4).to(DEV)
            bM.append(train_M(b, DEV, steps, n_layers=L))
        capO = optimal_reader_capacity(rho, n_layers=L, device=DEV, trials=3000)
        capL = learned_collusion_capacity(rho, n_layers=L, device=DEV, steps=500)
        out["points"][str(v)] = dict(
            d=d, L=L,
            ert_M=float(np.mean(accM)), ert_M_std=float(np.std(accM)),
            ert_S=float(np.mean(accS)),
            baseline_M=float(np.mean(bM)),
            cap_optimal=capO, cap_learned=capL)
        print(f"{axis}={v:>4}  ertM={np.mean(accM):.3f}  baseM={np.mean(bM):.3f}  "
              f"ertS={np.mean(accS):.3f}  capO={capO:.3f}  capL={capL:.3f}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=["depth", "width"])
    ap.add_argument("--values", required=True)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    vals = [int(x) for x in a.values.split(",")]
    res = run(a.axis, vals, a.rho, a.steps, a.seeds)
    out = a.out or os.path.join(os.path.dirname(__file__), f"scale_{a.axis}.json")
    json.dump(res, open(out, "w"), indent=2)
    print("wrote", out, flush=True)
