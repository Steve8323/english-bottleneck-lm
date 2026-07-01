"""run.py -- orchestrate the whole study and dump results.json.

Experiments
-----------
E1 Capability double-dissociation: baseline vs ERT(rho) on Task M (meaning) and
   Task S (surface), across paraphrase strengths rho.
E2 Covert side-channel capacity through the L-layer stack vs rho (optimal reader
   + learned colluding sender/receiver).
E3 Legibility: print the residual stream (the words each layer emits) for a
   Task M example -- you can literally read the model thinking in English.
E4 Paraphraser cost: wall-clock of the fixed channel vs a model forward pass.
"""
import os, sys, json, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np
import torch

from vocab import build_embeddings, decode_ids, V, N_CLASS
from paraphrase import ParaphraseChannel
from model import EnglishResidualTransformer, ContinuousBaseline
from train import train_M, train_S, eval_M, eval_S
from capacity import optimal_reader_capacity, learned_collusion_capacity
from tasks import make_batch_M, RPOS, SEQ_LEN

QUICK = "--quick" in sys.argv
# CPU is faster here: the models are tiny and dominated by many small ops, where
# MPS kernel-launch overhead hurts. Override with EBL_DEVICE=mps if desired.
DEV = os.environ.get("EBL_DEVICE", "cpu")
torch.manual_seed(0); np.random.seed(0)

D = 64
LAYERS = 4
EPS = 0.5
STEPS = 600 if QUICK else 1800
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]


def log(m): print(m, flush=True)


def banner(m): log("\n" + "=" * 64 + f"\n{m}\n" + "=" * 64)


def new_ert(rho):
    E = build_embeddings(D, EPS).to(DEV)
    para = ParaphraseChannel(rho=rho, device=DEV).to(DEV) if rho > 0 else \
        ParaphraseChannel(rho=0.0, device=DEV).to(DEV)
    return EnglishResidualTransformer(E, LAYERS, paraphrase=para, seq_len=SEQ_LEN).to(DEV)


def main():
    t0 = time.time()
    R = {"config": dict(device=DEV, d=D, layers=LAYERS, hops=LAYERS,
                        eps=EPS, steps=STEPS, vocab=V, n_class=N_CLASS,
                        rhos=RHOS, quick=QUICK)}
    log(f"device={DEV}  steps={STEPS}  quick={QUICK}")

    # ---- chance levels ----
    R["chance"] = {"M": 1.0 / N_CLASS, "S": 1.0 / V}

    # ---- E1: capability ----
    banner("E1  Capability: baseline vs ERT(rho)  on Task M and Task S")
    E = build_embeddings(D, EPS).to(DEV)
    base = ContinuousBaseline(E, LAYERS, seq_len=SEQ_LEN).to(DEV)
    log("[baseline] Task M ..."); base_M = train_M(base, DEV, STEPS, n_layers=LAYERS, log=log)
    base2 = ContinuousBaseline(E, LAYERS, seq_len=SEQ_LEN).to(DEV)
    log("[baseline] Task S ..."); base_S = train_S(base2, DEV, STEPS, n_layers=LAYERS, log=log)
    log(f"  baseline  M={base_M:.3f}  S={base_S:.3f}")

    R["E1"] = {"baseline": {"M": base_M, "S": base_S}, "ert": {}}
    for rho in RHOS:
        m = new_ert(rho); log(f"[ERT rho={rho}] Task M ...")
        mM = train_M(m, DEV, STEPS, n_layers=LAYERS, log=log)
        s = new_ert(rho); log(f"[ERT rho={rho}] Task S ...")
        mS = train_S(s, DEV, STEPS, n_layers=LAYERS, log=log)
        R["E1"]["ert"][str(rho)] = {"M": mM, "S": mS}
        log(f"  ERT rho={rho}  M={mM:.3f}  S={mS:.3f}")

    # ---- E2: covert capacity ----
    banner("E2  Covert side-channel capacity through the stack vs rho")
    R["E2"] = {"optimal": {}, "learned": {}}
    for rho in RHOS:
        opt = optimal_reader_capacity(rho, n_layers=LAYERS, device=DEV,
                                      trials=2000 if QUICK else 4000)
        lrn = learned_collusion_capacity(rho, n_layers=LAYERS, device=DEV,
                                         steps=300 if QUICK else 700)
        R["E2"]["optimal"][str(rho)] = opt
        R["E2"]["learned"][str(rho)] = lrn
        log(f"  rho={rho}  optimal-reader={opt:.3f}  learned-collusion={lrn:.3f}")

    # ---- E3: legibility trace ----
    banner("E3  Legibility: the residual stream as English (ERT rho=1.0)")
    mleg = new_ert(1.0); train_M(mleg, DEV, STEPS, n_layers=LAYERS)
    ids, _, final = make_batch_M(4, LAYERS, DEV)
    with torch.no_grad():
        _, trace = mleg.forward(ids, tau=0.3, return_trace=True)
    legib = []
    for b in range(ids.shape[0]):
        ex = {"input": decode_ids(ids[b].cpu().tolist()),
              "target_class": int(final[b].item()),
              "layers": [decode_ids(t[b].tolist()) for t in trace]}
        legib.append(ex)
        log(f"  in: {ex['input']}")
        for li, ly in enumerate(ex["layers"]):
            log(f"     L{li}: {ly}")
    R["E3"] = legib

    # ---- E4: paraphraser cost ----
    banner("E4  Paraphraser cost: fixed channel vs a model forward")
    P = ParaphraseChannel(rho=1.0, device=DEV).to(DEV)
    y = torch.zeros(256, SEQ_LEN, V, device=DEV); y[..., 0] = 1.0
    for _ in range(3): P(y)  # warmup
    t = time.time()
    for _ in range(200): P(y)
    para_ms = (time.time() - t) / 200 * 1e3
    mdl = new_ert(1.0); ids, _, _ = make_batch_M(256, LAYERS, DEV)
    for _ in range(3): mdl.forward(ids)
    t = time.time()
    for _ in range(50): mdl.forward(ids)
    mdl_ms = (time.time() - t) / 50 * 1e3
    R["E4"] = {"paraphrase_ms_per_batch": para_ms, "model_fwd_ms_per_batch": mdl_ms,
               "ratio": mdl_ms / max(1e-6, para_ms)}
    log(f"  fixed paraphrase: {para_ms:.3f} ms/batch   model fwd: {mdl_ms:.2f} ms/batch"
        f"   (model is {R['E4']['ratio']:.0f}x costlier)")

    R["wall_time_s"] = time.time() - t0
    json.dump(R, open(os.path.join(os.path.dirname(__file__), "results.json"), "w"), indent=2)
    banner(f"DONE in {R['wall_time_s']:.0f}s  ->  results.json")
    return R


if __name__ == "__main__":
    main()
