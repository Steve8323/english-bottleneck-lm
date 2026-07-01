"""e1_seeds.py -- multi-seed capability curves (mean+/-std) for E1, merged into
results.json under key 'E1_seeds'. Keeps E2/E3/E4 from the full run."""
import os, sys, json
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch

from vocab import build_embeddings, V, N_CLASS
from paraphrase import ParaphraseChannel
from model import EnglishResidualTransformer, ContinuousBaseline
from train import train_M, train_S

DEV = os.environ.get("EBL_DEVICE", "cpu")
D, LAYERS, EPS = 64, 4, 0.5
STEPS = 1200
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]
HERE = os.path.dirname(os.path.abspath(__file__))


def ert(E, rho):
    p = ParaphraseChannel(rho=rho, device=DEV).to(DEV)
    return EnglishResidualTransformer(E, LAYERS, paraphrase=p, seq_len=4).to(DEV)


def main():
    acc = {"seeds": SEEDS, "rhos": RHOS,
           "baseline": {"M": [], "S": []},
           "ert": {str(r): {"M": [], "S": []} for r in RHOS}}
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        E = build_embeddings(D, EPS).to(DEV)
        b1 = ContinuousBaseline(E, LAYERS, seq_len=4).to(DEV)
        acc["baseline"]["M"].append(train_M(b1, DEV, STEPS, n_layers=LAYERS))
        b2 = ContinuousBaseline(E, LAYERS, seq_len=4).to(DEV)
        acc["baseline"]["S"].append(train_S(b2, DEV, STEPS, n_layers=LAYERS))
        for r in RHOS:
            m = ert(E, r); acc["ert"][str(r)]["M"].append(train_M(m, DEV, STEPS, n_layers=LAYERS))
            s = ert(E, r); acc["ert"][str(r)]["S"].append(train_S(s, DEV, STEPS, n_layers=LAYERS))
        print(f"seed {sd} done  ertM={[round(acc['ert'][str(r)]['M'][-1],3) for r in RHOS]}"
              f"  ertS={[round(acc['ert'][str(r)]['S'][-1],3) for r in RHOS]}", flush=True)

    def ms(xs): return [float(np.mean(xs)), float(np.std(xs))]
    summary = {"baseline": {"M": ms(acc["baseline"]["M"]), "S": ms(acc["baseline"]["S"])},
               "ert": {r: {"M": ms(acc["ert"][r]["M"]), "S": ms(acc["ert"][r]["S"])}
                       for r in acc["ert"]},
               "raw": acc}
    path = os.path.join(HERE, "results.json")
    R = json.load(open(path)); R["E1_seeds"] = summary
    json.dump(R, open(path, "w"), indent=2)
    print("merged E1_seeds into results.json", flush=True)


if __name__ == "__main__":
    main()
