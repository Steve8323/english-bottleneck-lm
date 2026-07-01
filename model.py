"""model.py -- English-Residual Transformer (ERT) and a continuous baseline.

ERT: there is NO continuous residual stream across layers. After every layer the
state is turned into *words* (Gumbel-softmax straight-through over the real
vocabulary), pushed through the paraphrase channel, and only those words are
read by the next layer. Inside a layer the compute is an ordinary transformer
block; the constraint is purely on what crosses a layer boundary.

Baseline: an ordinary continuous-residual transformer of matched depth/width --
the "normal model" reference (capability ceiling).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import V, N_CLASS, CLASS_OF


class Block(nn.Module):
    """Standard pre-norm transformer block (bidirectional attention)."""

    def __init__(self, d, heads=4, mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(),
                                 nn.Linear(mult * d, d))

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


def gumbel_st(logits, tau):
    """Gumbel-softmax straight-through: hard one-hot fwd, soft grad."""
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
    soft = torch.softmax((logits + g) / tau, -1)
    hard = torch.zeros_like(soft)
    hard.scatter_(-1, soft.argmax(-1, keepdim=True), 1.0)
    return hard + (soft - soft.detach())


class EnglishResidualTransformer(nn.Module):
    def __init__(self, E, n_layers=4, heads=4, paraphrase=None, seq_len=7):
        super().__init__()
        self.register_buffer("E", E)          # frozen public embedding (V,d)
        self.d = E.shape[1]
        self.pos = nn.Parameter(0.02 * torch.randn(seq_len, self.d))
        self.blocks = nn.ModuleList(Block(self.d, heads) for _ in range(n_layers))
        # per-layer unembedding (d -> V word logits): the learnable word-to-word map
        self.unemb = nn.ModuleList(nn.Linear(self.d, V) for _ in range(n_layers))
        self.lnf = nn.LayerNorm(self.d)
        self.para = paraphrase  # ParaphraseChannel or None
        self.n_layers = n_layers

    def forward_layers(self, ids, tau=0.5):
        """Returns list of per-layer word logits (B,T,V). Between layers the
        emitted words are committed (Gumbel-ST) and paraphrased -- that noised,
        discrete message is the ONLY thing the next layer sees."""
        y = F.one_hot(ids, V).float()
        outs = []
        for li, (blk, un) in enumerate(zip(self.blocks, self.unemb)):
            x = y @ self.E + self.pos.unsqueeze(0)
            x = blk(x)
            logits = un(self.lnf(x))               # (B,T,V) word logits
            outs.append(logits)
            if li < self.n_layers - 1:
                y = gumbel_st(logits, tau)         # commit to words
                if self.para is not None:
                    y = self.para(y)               # paraphrase between layers
        return outs

    def forward(self, ids, tau=0.5, return_trace=False):
        outs = self.forward_layers(ids, tau)
        if return_trace:
            return outs[-1], [o.argmax(-1).detach().cpu() for o in outs]
        return outs[-1]

    def readout_token(self, ids, qpos, tau=0.5):
        return self.forward(ids, tau)[:, qpos, :]

    def readout_class(self, ids, qpos, tau=0.5):
        return _tok_to_class_logits(self.readout_token(ids, qpos, tau))


class ContinuousBaseline(nn.Module):
    """Ordinary continuous-residual transformer encoder. Embeds once, keeps a
    real-valued residual stream throughout. No word bottleneck, no paraphrase."""

    def __init__(self, E, n_layers=4, heads=4, seq_len=7, trainable_emb=False):
        super().__init__()
        if trainable_emb:
            self.emb = nn.Parameter(E.clone())
        else:
            self.register_buffer("emb", E)
        self.trainable_emb = trainable_emb
        self.d = E.shape[1]
        self.pos = nn.Parameter(0.02 * torch.randn(seq_len, self.d))
        self.blocks = nn.ModuleList(Block(self.d, heads) for _ in range(n_layers))
        self.lnf = nn.LayerNorm(self.d)
        # per-layer heads so the baseline gets the SAME deep supervision signal
        self.heads = nn.ModuleList(nn.Linear(self.d, V) for _ in range(n_layers))
        self.head = self.heads[-1]

    def forward_layers(self, ids, **kw):
        x = F.one_hot(ids, V).float() @ self.emb + self.pos.unsqueeze(0)
        outs = []
        for blk, hd in zip(self.blocks, self.heads):
            x = blk(x)
            outs.append(hd(self.lnf(x)))
        return outs

    def forward(self, ids, **kw):
        return self.forward_layers(ids)[-1]

    def readout_token(self, ids, qpos, **kw):
        return self.forward(ids)[:, qpos, :]

    def readout_class(self, ids, qpos, **kw):
        return _tok_to_class_logits(self.forward(ids)[:, qpos, :])


def _tok_to_class_logits(tok_logits):
    """Sum token log-probs into class buckets -> (B, N_CLASS)."""
    logp = F.log_softmax(tok_logits, -1)
    B = logp.shape[0]
    out = torch.full((B, N_CLASS), -1e9, device=logp.device)
    co = CLASS_OF.to(logp.device)
    for c in range(N_CLASS):
        mask = (co == c)
        out[:, c] = torch.logsumexp(logp[:, mask], -1)
    return out
