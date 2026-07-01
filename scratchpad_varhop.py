"""scratchpad_varhop.py -- confirm WHY ERT fails varhop, and that a legible
scratchpad fixes it.

varhop needs the hop-count k carried across layers. ERT has no continuous
residual, so unsupervised state dies at the word bottleneck -> it fails (~0.44).
Fix: supervise ERT to PERSIST the instruction token <hopk> at position 1 at
every layer (a legible scratchpad). If ERT then solves varhop, the tax was
"must externalize working memory", not "can't reason".

Compares: continuous baseline | ERT no-scratchpad | ERT WITH legible scratchpad,
at rho=0 and rho=1.
"""
import os, json
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch
import torch.nn.functional as F

from vocab import build_embeddings, N_CLASS, SYN_PER, MEMBERS, PAD, STOI
from paraphrase import ParaphraseChannel
from model import EnglishResidualTransformer, ContinuousBaseline, _tok_to_class_logits
from tasks import g_pow, RPOS, SEQ_LEN
from train import _tau

DEV = os.environ.get("EBL_DEVICE", "cpu")
D, EPS, L = 64, 0.5, 4
CPOS = 1  # counter / instruction position
_rng = np.random.default_rng(99)


def gen(B, dev):
    ids = np.full((B, SEQ_LEN), PAD, np.int64)
    pos0 = np.empty((B, L), np.int64)   # class target per layer (current item)
    kcount = np.empty(B, np.int64)
    fin = np.empty(B, np.int64)
    for i in range(B):
        c0 = int(_rng.integers(N_CLASS)); ids[i, 0] = MEMBERS[c0][_rng.integers(SYN_PER)]
        k = int(_rng.integers(1, L + 1)); ids[i, CPOS] = STOI[f"<hop{k}>"]; kcount[i] = k
        for l in range(L):
            pos0[i, l] = g_pow(c0, min(l + 1, k))
        fin[i] = g_pow(c0, k)
    return (torch.tensor(ids, device=dev), torch.tensor(pos0, device=dev),
            torch.tensor(kcount, device=dev), torch.tensor(fin, device=dev))


def train(model, scratchpad, rho_tag, steps=1200, B=256, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr); model.train()
    for s in range(steps):
        ids, pos0, kc, _ = gen(B, DEV)
        outs = model.forward_layers(ids, tau=_tau(s, steps))
        loss = 0.0
        for l, o in enumerate(outs):
            loss = loss + F.cross_entropy(_tok_to_class_logits(o[:, RPOS, :]), pos0[:, l])
            if scratchpad:  # persist the instruction token <hopk> at CPOS, every layer
                ktok = torch.tensor([STOI[f"<hop{int(k)}>"] for k in kc.tolist()], device=DEV)
                loss = loss + F.cross_entropy(o[:, CPOS, :], ktok)
        loss = loss / len(outs)
        opt.zero_grad(); loss.backward(); opt.step()
    return evaluate(model)


@torch.no_grad()
def evaluate(model, B=2000, reps=3):
    model.eval(); acc = 0.0
    for _ in range(reps):
        ids, _, _, fin = gen(B, DEV)
        pred = _tok_to_class_logits(model.forward(ids, tau=0.3)[:, RPOS, :]).argmax(-1)
        acc += (pred == fin).float().mean().item()
    return acc / reps


def ert(E, rho):
    return EnglishResidualTransformer(E, L, paraphrase=ParaphraseChannel(rho, DEV).to(DEV),
                                      seq_len=SEQ_LEN).to(DEV)


def main():
    torch.manual_seed(0); np.random.seed(0)
    E = build_embeddings(D, EPS).to(DEV)
    R = {"chance": 1.0 / N_CLASS}
    R["baseline"] = train(ContinuousBaseline(E, L, seq_len=SEQ_LEN).to(DEV), False, "base")
    R["ert_noscratch_rho0"] = train(ert(E, 0.0), False, "r0")
    R["ert_noscratch_rho1"] = train(ert(E, 1.0), False, "r1")
    R["ert_scratch_rho0"] = train(ert(E, 0.0), True, "r0s")
    R["ert_scratch_rho1"] = train(ert(E, 1.0), True, "r1s")
    for k, v in R.items():
        if k != "chance":
            print(f"{k:>22}: {v:.3f}", flush=True)
    json.dump(R, open(os.path.join(os.path.dirname(__file__), "scratchpad_varhop.json"), "w"), indent=2)
    print("wrote scratchpad_varhop.json", flush=True)


if __name__ == "__main__":
    main()
