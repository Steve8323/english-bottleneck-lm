"""benchmarks.py -- a small reasoning battery to measure how much worse the
English-bottleneck model reasons vs a normal transformer.

Three meaning-based reasoning tasks (the thesis predicts a SMALL tax, because
the paraphrase preserves meaning):

  T1 chain      : apply a fixed class rule g once per layer (the original Task M).
  T2 varhop     : variable #hops k (given by a token); answer g^k(c0). The model
                  must advance exactly k times then hold -> dynamic-depth reasoning.
  T3 ruleselect : an op token chooses rule g or rule h; apply it every layer.
                  -> reading an instruction and routing on it.

For each we report continuous baseline, ERT(rho=0) and ERT(rho=1) final accuracy
(the capability tax = baseline - ERT). Writes benchmarks.json.
"""
import os, json
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch
import torch.nn.functional as F

from vocab import build_embeddings, N_CLASS, SYN_PER, MEMBERS, PAD, STOI
from paraphrase import ParaphraseChannel
from model import EnglishResidualTransformer, ContinuousBaseline, _tok_to_class_logits
from tasks import SUCC, g_pow, RPOS, SEQ_LEN
from train import _tau

DEV = os.environ.get("EBL_DEVICE", "cpu")
D, EPS = 64, 0.5
_rng = np.random.default_rng(99)
SUCC2 = np.random.default_rng(777).permutation(N_CLASS)  # second rule h


def h_pow(c, k):
    for _ in range(k):
        c = int(SUCC2[c])
    return c


def gen_chain(B, L, dev):
    ids = np.full((B, SEQ_LEN), PAD, np.int64); lt = np.empty((B, L), np.int64)
    for i in range(B):
        c0 = int(_rng.integers(N_CLASS)); ids[i, 0] = MEMBERS[c0][_rng.integers(SYN_PER)]
        c = c0
        for l in range(L):
            c = int(SUCC[c]); lt[i, l] = c
    return torch.tensor(ids, device=dev), torch.tensor(lt, device=dev), torch.tensor(lt[:, -1], device=dev)


def gen_varhop(B, L, dev):
    ids = np.full((B, SEQ_LEN), PAD, np.int64); lt = np.empty((B, L), np.int64); fin = np.empty(B, np.int64)
    for i in range(B):
        c0 = int(_rng.integers(N_CLASS)); ids[i, 0] = MEMBERS[c0][_rng.integers(SYN_PER)]
        k = int(_rng.integers(1, L + 1)); ids[i, 1] = STOI[f"<hop{k}>"]
        for l in range(L):
            lt[i, l] = g_pow(c0, min(l + 1, k))  # advance to k then hold
        fin[i] = g_pow(c0, k)
    return torch.tensor(ids, device=dev), torch.tensor(lt, device=dev), torch.tensor(fin, device=dev)


def gen_ruleselect(B, L, dev):
    ids = np.full((B, SEQ_LEN), PAD, np.int64); lt = np.empty((B, L), np.int64); fin = np.empty(B, np.int64)
    for i in range(B):
        c0 = int(_rng.integers(N_CLASS)); ids[i, 0] = MEMBERS[c0][_rng.integers(SYN_PER)]
        op = int(_rng.integers(2)); ids[i, 1] = STOI["<hop1>"] if op == 0 else STOI["<hop2>"]
        pw = g_pow if op == 0 else h_pow
        for l in range(L):
            lt[i, l] = pw(c0, l + 1)
        fin[i] = pw(c0, L)
    return torch.tensor(ids, device=dev), torch.tensor(lt, device=dev), torch.tensor(fin, device=dev)


TASKS = {"chain": gen_chain, "varhop": gen_varhop, "ruleselect": gen_ruleselect}


def train(model, gen, L, steps=900, B=256, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr); model.train()
    for s in range(steps):
        ids, lt, _ = gen(B, L, DEV)
        outs = model.forward_layers(ids, tau=_tau(s, steps))
        loss = sum(F.cross_entropy(_tok_to_class_logits(o[:, RPOS, :]), lt[:, l])
                   for l, o in enumerate(outs)) / len(outs)
        opt.zero_grad(); loss.backward(); opt.step()
    return evaluate(model, gen, L)


@torch.no_grad()
def evaluate(model, gen, L, B=2000, reps=3):
    model.eval(); acc = 0.0
    for _ in range(reps):
        ids, _, fin = gen(B, L, DEV)
        pred = _tok_to_class_logits(model.forward(ids, tau=0.3)[:, RPOS, :]).argmax(-1)
        acc += (pred == fin).float().mean().item()
    return acc / reps


def main():
    L = 4; R = {"chance": 1.0 / N_CLASS, "n_layers": L, "results": {}}
    for name, gen in TASKS.items():
        torch.manual_seed(0); np.random.seed(0)
        E = build_embeddings(D, EPS).to(DEV)
        base = ContinuousBaseline(E, L, seq_len=SEQ_LEN).to(DEV)
        b = train(base, gen, L)
        e0 = train(EnglishResidualTransformer(E, L, paraphrase=ParaphraseChannel(0.0, DEV).to(DEV), seq_len=SEQ_LEN).to(DEV), gen, L)
        e1 = train(EnglishResidualTransformer(E, L, paraphrase=ParaphraseChannel(1.0, DEV).to(DEV), seq_len=SEQ_LEN).to(DEV), gen, L)
        R["results"][name] = {"baseline": b, "ert_rho0": e0, "ert_rho1": e1,
                              "tax_rho1": b - e1}
        print(f"{name:>11}: baseline={b:.3f}  ERT(rho=0)={e0:.3f}  ERT(rho=1)={e1:.3f}"
              f"  tax={b - e1:+.3f}", flush=True)
    json.dump(R, open(os.path.join(os.path.dirname(__file__), "benchmarks.json"), "w"), indent=2)
    print("wrote benchmarks.json", flush=True)


if __name__ == "__main__":
    main()
