"""retrain_deep.py -- does training the DEEP models harder fix the collapse?

Same architecture, much stronger optimization: 5x steps, cosine LR decay, sharper
final Gumbel temperature, more seeds. Tests depth 8 and 12 at rho=0 (control: no
paraphrase) and rho=1 (real). Also measures PER-LAYER accuracy p_l, because if
the meaning task is learnable then each layer's one-hop accuracy should approach
1.0 and the final = prod(p_l). If p_l is high but final still low -> compounding
is the issue (need even higher p). If p_l itself is stuck -> deeper problem.

Key fact: the paraphrase preserves the class EXACTLY, so there is no information
floor on the meaning task -- only an optimization gap.
"""
import os, json, math
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch
import torch.nn.functional as F

from vocab import build_embeddings, N_CLASS
from paraphrase import ParaphraseChannel
from model import EnglishResidualTransformer, ContinuousBaseline, _tok_to_class_logits
from tasks import make_batch_M, RPOS, SEQ_LEN

DEV = os.environ.get("EBL_DEVICE", "cpu")
D, EPS = 64, 0.5


def train_better(model, L, steps, B=256, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        frac = s / max(1, steps)
        for g in opt.param_groups:                       # cosine LR decay
            g["lr"] = 0.5 * lr * (1 + math.cos(math.pi * frac)) + 1e-4
        tau = 1.0 + (0.1 - 1.0) * frac                   # sharper final temp (->0.1)
        ids, lt, _ = make_batch_M(B, L, DEV)
        outs = model.forward_layers(ids, tau=tau)
        loss = sum(F.cross_entropy(_tok_to_class_logits(o[:, RPOS, :]), lt[:, l])
                   for l, o in enumerate(outs)) / len(outs)
        opt.zero_grad(); loss.backward(); opt.step()
    return eval_full(model, L)


@torch.no_grad()
def eval_full(model, L, B=3000):
    model.eval()
    ids, lt, final = make_batch_M(B, L, DEV)
    outs = model.forward_layers(ids, tau=0.1)
    per = []
    for l, o in enumerate(outs):
        pred = _tok_to_class_logits(o[:, RPOS, :]).argmax(-1)
        per.append((pred == lt[:, l]).float().mean().item())
    final_acc = (_tok_to_class_logits(outs[-1][:, RPOS, :]).argmax(-1) == final).float().mean().item()
    return final_acc, per


def ert(E, L, rho):
    return EnglishResidualTransformer(E, L, paraphrase=ParaphraseChannel(rho, DEV).to(DEV),
                                      seq_len=SEQ_LEN).to(DEV)


def main():
    STEPS, SEEDS = 5000, 2
    DEPTHS = [8, 12]
    R = {"chance": 1.0 / N_CLASS, "steps": STEPS, "seeds": SEEDS,
         "old_undertrained": {"8": 0.408, "12": 0.050}, "results": {}}
    for Ld in DEPTHS:
        E = build_embeddings(D, EPS).to(DEV)
        torch.manual_seed(0); np.random.seed(0)
        base, _ = train_better(ContinuousBaseline(E, Ld, seq_len=SEQ_LEN).to(DEV), Ld, STEPS)
        entry = {"baseline": base, "rho0": [], "rho1": [], "rho1_perlayer": None}
        for rho in (0.0, 1.0):
            accs = []
            for sd in range(SEEDS):
                torch.manual_seed(sd); np.random.seed(sd)
                fa, per = train_better(ert(E, Ld, rho), Ld, STEPS)
                accs.append(fa)
                if rho == 1.0 and sd == 0:
                    entry["rho1_perlayer"] = [round(p, 3) for p in per]
            entry["rho0" if rho == 0 else "rho1"] = accs
        prod = float(np.prod(entry["rho1_perlayer"])) if entry["rho1_perlayer"] else None
        entry["rho1_perlayer_prod"] = prod
        R["results"][str(Ld)] = entry
        print(f"depth {Ld}: baseline={base:.3f}  rho0={np.mean(entry['rho0']):.3f}  "
              f"rho1={np.mean(entry['rho1']):.3f}  (was {R['old_undertrained'][str(Ld)]})", flush=True)
        print(f"   per-layer p (rho1): {entry['rho1_perlayer']}  prod={prod}", flush=True)
    json.dump(R, open(os.path.join(os.path.dirname(__file__), "retrain_deep.json"), "w"), indent=2)
    print("wrote retrain_deep.json", flush=True)


if __name__ == "__main__":
    main()
