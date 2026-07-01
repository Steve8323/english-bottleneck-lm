"""tasks.py -- two synthetic tasks giving a DOUBLE DISSOCIATION, trained with
process (deep) supervision: each layer emits an English word and we supervise
that word. This is the honest way to train a "think in words at every layer"
model and mirrors process-supervision (2506.01926).

Task M (MEANING): one hop of a fixed class->class rule g per layer. Input is a
start word (a synonym of class c0). After layer L the answer is g^L(c0).
Per-layer target for layer l is g^(l+1)(c0). Depends ONLY on meaning (class), so
a meaning-preserving paraphrase between layers must not hurt it -- the load-
bearing thing crossing each boundary is "the current class", which P preserves.

Task S (SURFACE): copy the EXACT start synonym at every layer. Per-layer target
is the exact word w0. Meaning is insufficient; you must carry surface across
boundaries, which P destroys -- so accuracy collapses as rho -> 1.
"""
import numpy as np
import torch

from vocab import N_CLASS, SYN_PER, MEMBERS, PAD, V

SEQ_LEN = 4
RPOS = 0  # the "current item" lives at position 0 and is read out there

_rng = np.random.default_rng(1234)
SUCC = _rng.permutation(N_CLASS)  # g(c)


def g_pow(c, k):
    for _ in range(k):
        c = int(SUCC[c])
    return c


def make_batch_M(B, n_layers, device):
    """ids (B,SEQ_LEN); layer_targets (B,n_layers) class per layer;
    final (B,) = g^n_layers(c0)."""
    ids = np.full((B, SEQ_LEN), PAD, dtype=np.int64)
    c0 = _rng.integers(0, N_CLASS, B)
    syn = _rng.integers(0, SYN_PER, B)
    lt = np.empty((B, n_layers), dtype=np.int64)
    for i in range(B):
        ids[i, RPOS] = MEMBERS[c0[i]][syn[i]]
        c = int(c0[i])
        for l in range(n_layers):
            c = int(SUCC[c]); lt[i, l] = c
    final = lt[:, -1].copy()
    return (torch.tensor(ids, device=device),
            torch.tensor(lt, device=device),
            torch.tensor(final, device=device))


def make_batch_S(B, n_layers, device):
    """ids (B,SEQ_LEN); layer_targets (B,n_layers) = exact w0 each layer;
    final (B,) = w0 token."""
    ids = np.full((B, SEQ_LEN), PAD, dtype=np.int64)
    w0 = np.empty(B, dtype=np.int64)
    for i in range(B):
        c0 = _rng.integers(0, N_CLASS); s0 = _rng.integers(0, SYN_PER)
        w0[i] = MEMBERS[c0][s0]
        ids[i, RPOS] = w0[i]
        # fill other positions with distractors so copy isn't trivial position id
        for p in range(1, SEQ_LEN):
            ids[i, p] = MEMBERS[_rng.integers(0, N_CLASS)][_rng.integers(0, SYN_PER)]
    lt = np.repeat(w0[:, None], n_layers, axis=1)
    return (torch.tensor(ids, device=device),
            torch.tensor(lt, device=device),
            torch.tensor(w0, device=device))
