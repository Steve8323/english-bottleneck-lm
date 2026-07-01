"""cost.py -- EXACT cost of adding the paraphraser, and how it scales.

Two regimes:
  * deployment (inference): the paraphrase is a token-level op -- argmax to a
    word id, then a table swap to a random class-mate. O(B*T). This is the real
    cost you'd pay at serving time.
  * training: the differentiable straight-through version materializes a V-wide
    one-hot, O(B*T*V) -- reported separately.

We compare against the cost of ONE transformer layer at several widths d and
sequence lengths T, because the paraphrase overhead (as a % of a layer) SHRINKS
as the model gets bigger: a layer is O(B*T*d^2 + B*T^2*d) while the paraphrase
is O(B*T), independent of d.
"""
import os, time, json
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch

from model import Block
from paraphrase import ParaphraseChannel
from vocab import V

DEV = "cpu"
B = 256


def timeit(fn, iters=80, warm=8):
    for _ in range(warm):
        fn()
    t = time.time()
    for _ in range(iters):
        fn()
    return (time.time() - t) / iters * 1e3  # ms/call


def main():
    res = {"batch": B, "vocab": V, "device": DEV, "configs": []}
    P = ParaphraseChannel(1.0, DEV).to(DEV)
    print(f"{'d':>5} {'T':>4} | {'1 layer(ms)':>12} {'para tok(ms)':>13} "
          f"{'para 1hot(ms)':>14} | {'+%/layer (tok)':>15}")
    for d, T in [(64, 4), (128, 16), (256, 64), (512, 128), (1024, 256)]:
        blk = Block(d).to(DEV)
        x = torch.randn(B, T, d)
        t_layer = timeit(lambda: blk(x))

        toks = torch.randint(0, V, (B, T))

        def para_tok():  # deployment: token-id swap, no V-onehot
            ridx = torch.randint(0, P.k, (B, T))
            mate = P.swap[toks].gather(-1, ridx.unsqueeze(-1)).squeeze(-1)
            do = torch.rand(B, T) < P.rho
            return torch.where(do, mate, toks)
        t_para_tok = timeit(para_tok)

        y = torch.zeros(B, T, V); y[..., 0] = 1.0

        def para_1hot():  # training: differentiable STE on V-wide one-hot
            return P(y)
        t_para_1hot = timeit(para_1hot)

        pct = t_para_tok / t_layer * 100
        res["configs"].append(dict(d=d, T=T, layer_ms=t_layer,
                                   para_tok_ms=t_para_tok, para_1hot_ms=t_para_1hot,
                                   para_pct_of_layer_tok=pct))
        print(f"{d:>5} {T:>4} | {t_layer:>12.3f} {t_para_tok:>13.4f} "
              f"{t_para_1hot:>14.3f} | {pct:>14.2f}%")
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "cost.json"), "w"), indent=2)
    print("\nwrote cost.json")


if __name__ == "__main__":
    main()
