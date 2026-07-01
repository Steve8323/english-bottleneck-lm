"""decrement_varhop.py -- the RIGHT scratchpad test.

My first scratchpad (persist fixed <hopk>) failed to fix varhop. That version
forces the model to compare its layer-identity l against k -- nonlocal. The
clean test is a DECREMENTING counter carried legibly: at layer l, position 1
holds <hop_remaining> with remaining=max(k-(l+1),0) (0 -> <pad>). Then each
layer does a purely LOCAL rule: "if remaining>0: advance the item and decrement;
else hold." If ERT solves varhop with this, the varhop tax was 'must externalize
working memory as legible tokens'. If it still fails, the limitation is deeper:
data-dependent conditional computation is hard through the discrete bottleneck.
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
CPOS = 1
_rng = np.random.default_rng(99)


def rem_tok(r):
    return STOI[f"<hop{r}>"] if r >= 1 else PAD


def gen(B, dev):
    ids = np.full((B, SEQ_LEN), PAD, np.int64)
    pos0 = np.empty((B, L), np.int64); cpos = np.empty((B, L), np.int64); fin = np.empty(B, np.int64)
    for i in range(B):
        c0 = int(_rng.integers(N_CLASS)); ids[i, 0] = MEMBERS[c0][_rng.integers(SYN_PER)]
        k = int(_rng.integers(1, L + 1)); ids[i, CPOS] = STOI[f"<hop{k}>"]
        for l in range(L):
            pos0[i, l] = g_pow(c0, min(l + 1, k))
            cpos[i, l] = rem_tok(max(k - (l + 1), 0))
        fin[i] = g_pow(c0, k)
    return (torch.tensor(ids, device=dev), torch.tensor(pos0, device=dev),
            torch.tensor(cpos, device=dev), torch.tensor(fin, device=dev))


def train(model, scratch, steps=1500, B=256, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr); model.train()
    for s in range(steps):
        ids, pos0, cpos, _ = gen(B, DEV)
        outs = model.forward_layers(ids, tau=_tau(s, steps))
        loss = 0.0
        for l, o in enumerate(outs):
            loss = loss + F.cross_entropy(_tok_to_class_logits(o[:, RPOS, :]), pos0[:, l])
            if scratch:
                loss = loss + F.cross_entropy(o[:, CPOS, :], cpos[:, l])
        loss = loss / len(outs)
        opt.zero_grad(); loss.backward(); opt.step()
    return ev(model)


@torch.no_grad()
def ev(model, B=2000, reps=3):
    model.eval(); a = 0.0
    for _ in range(reps):
        ids, _, _, fin = gen(B, DEV)
        a += (_tok_to_class_logits(model.forward(ids, tau=0.3)[:, RPOS, :]).argmax(-1) == fin).float().mean().item()
    return a / reps


def ert(E, rho):
    return EnglishResidualTransformer(E, L, paraphrase=ParaphraseChannel(rho, DEV).to(DEV), seq_len=SEQ_LEN).to(DEV)


def main():
    torch.manual_seed(0); np.random.seed(0)
    E = build_embeddings(D, EPS).to(DEV)
    R = {"chance": 1.0 / N_CLASS,
         "baseline": train(ContinuousBaseline(E, L, seq_len=SEQ_LEN).to(DEV), False),
         "ert_decrement_rho0": train(ert(E, 0.0), True),
         "ert_decrement_rho1": train(ert(E, 1.0), True)}
    for k, v in R.items():
        if k != "chance":
            print(f"{k:>22}: {v:.3f}", flush=True)
    json.dump(R, open(os.path.join(os.path.dirname(__file__), "decrement_varhop.json"), "w"), indent=2)
    print("wrote decrement_varhop.json", flush=True)


if __name__ == "__main__":
    main()
