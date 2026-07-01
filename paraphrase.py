"""paraphrase.py -- the cheap, fast, in-model paraphrase channel P.

P acts on a sequence of words (one-hot or straight-through one-hot) at every
layer boundary. It is MEANING-PRESERVING and SURFACE-RANDOMIZING:

    with prob rho, replace each content word by a uniformly-random *other*
    synonym in the SAME meaning class. Specials pass through unchanged.

Key properties, which are the whole point of the "tiny paraphraser":
  * No model, no forward pass: it is a fixed, class-structured stochastic map
    -- O(1) table lookups. This is the analog of robust_paraphraser's
    "model-free canonical pass": the un-foolable floor that closes the lexical
    channel without anything to optimize against.
  * rho is the paraphrase strength. rho=0 -> identity (surface passes freely).
    rho=1 -> surface fully randomized within meaning ("perfect paraphraser").

Straight-through: forward pass does the hard discrete swap (so the channel is
genuinely word-space and really destroys surface bits); backward pass treats the
swap as identity-in-expectation so gradients flow to train the stack.

We also provide a tiny LEARNED paraphraser (TinyNoiser) for the efficiency
study, to show the fixed table closes the channel as well as a trained noiser at
a fraction of the cost.
"""
import torch
import torch.nn as nn

from vocab import V, MEMBERS, CLASS_OF, SYN_PER


def _build_swap_table(device):
    """For each token, the list of *same-class other* synonym ids. Content words
    map to their (SYN_PER-1) class-mates; specials map to themselves only."""
    others = torch.arange(V, device=device).unsqueeze(1).repeat(1, max(1, SYN_PER - 1))
    for members in MEMBERS:
        for tok in members:
            mates = [m for m in members if m != tok]
            others[tok] = torch.tensor(mates, device=device)
    return others  # (V, SYN_PER-1)


class ParaphraseChannel(nn.Module):
    """Fixed meaning-preserving surface-randomizing channel."""

    def __init__(self, rho=1.0, device="cpu"):
        super().__init__()
        self.rho = rho
        self.register_buffer("swap", _build_swap_table(device))  # (V, k)
        self.k = self.swap.shape[1]

    def forward(self, y):
        """y: (B, T, V) straight-through one-hot (hard fwd, soft grad).
        Returns paraphrased (B, T, V) with the same STE convention."""
        if self.rho <= 0:
            return y
        B, T, _ = y.shape
        tok = y.argmax(-1)  # (B,T) current hard word
        # pick a random class-mate for every position
        ridx = torch.randint(0, self.k, (B, T), device=y.device)
        mate = self.swap[tok].gather(-1, ridx.unsqueeze(-1)).squeeze(-1)  # (B,T)
        do = (torch.rand(B, T, device=y.device) < self.rho)
        new_tok = torch.where(do, mate, tok)
        hard = torch.zeros_like(y)
        hard.scatter_(-1, new_tok.unsqueeze(-1), 1.0)
        # straight-through: forward = hard swapped word; grad flows via y
        # (swap is a permutation-in-expectation over class-mates -> identity on
        # the meaning subspace, which is what we let gradients see).
        return hard + (y - y.detach())


class TinyNoiser(nn.Module):
    """A *learned* paraphraser baseline for the efficiency study: one linear
    layer V->V (a learned V x V word-to-word stochastic-ish matrix) plus
    softmax. Trained to preserve class while maximizing surface entropy. Used
    only to show the fixed ParaphraseChannel matches it at ~zero cost."""

    def __init__(self, hidden=0):
        super().__init__()
        self.lin = nn.Linear(V, V, bias=False)

    def forward(self, y, tau=0.5):
        logits = self.lin(y)
        soft = torch.softmax(logits / tau, -1)
        hard = torch.zeros_like(soft)
        hard.scatter_(-1, soft.argmax(-1, keepdim=True), 1.0)
        return hard + (soft - soft.detach())
