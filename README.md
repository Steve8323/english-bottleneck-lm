# English-Bottleneck LM — an LLM that can only reason in English between layers

A proof-of-concept transformer, the **English-Residual Transformer (ERT)**, in
which there is **no continuous residual stream**. After every layer the hidden
state is turned into actual *words* (Gumbel-softmax straight-through over the
real vocabulary), pushed through a fixed paraphrase channel, and only those
words are read by the next layer. Compute *inside* a layer is an ordinary
transformer block; the constraint is purely on what crosses a layer boundary —
so the model's inter-layer "working memory" is legible English by construction.

The question: **if you force a model to think in English, what does it cost, and
what does it buy for oversight?**

## Results (toy scale: d=64, 4 layers/4 hops, 24-class synthetic meaning task)

| # | Experiment | Finding |
|---|---|---|
| **E1** | **Capability double-dissociation** | At full paraphrase strength (ρ=1) the ERT keeps the **meaning** task at **M=0.996** (chance 0.042) while the **surface** task collapses to **S=0.247**. The bottleneck preserves semantics and destroys surface form — exactly the intended dissociation. Continuous baseline: M=S=1.0. |
| **E2** | **Covert side-channel capacity** | A colluding sender/receiver trying to pass a secret bit through the surface channel drops to **chance (~0.50)** at ρ=1 — both for an optimal reader (0.496) and a learned collusion net (0.518). The bottleneck closes the covert channel. |
| **E3** | **Legibility** | You can literally print the words each layer emits and read the model "thinking" (see `report_latest.html`). |
| **E4** | **Paraphraser cost** | The fixed channel is **~72× cheaper** than a model forward pass (0.067 ms vs 4.8 ms/batch) — the "tiny paraphraser" is essentially free. |

The alignment pitch: a residual stream that is legible-by-construction and
provably low-capacity for covert coordination, at near-zero added cost. This is
a **toy testbed**, not a scaled model — see Limitations.

![results](report_latest.html) → open `report_latest.html` for the plots.

## The paraphrase channel

`P` acts on the sequence of words at each layer boundary. It is
**meaning-preserving, surface-randomizing**: with probability ρ it replaces each
content word with a uniformly random *other* synonym in the same meaning class.
It is **model-free** — a fixed, class-structured stochastic map (O(1) table
lookups), the "un-foolable floor" that closes the lexical channel with nothing
to optimize against. ρ=0 is the identity (surface passes freely); ρ=1 fully
randomizes surface while preserving meaning.

## Run

```bash
pip install torch numpy matplotlib
python run.py            # E1–E4 -> results.json + report_latest.html
python scale.py --axis depth --values 2,4,6,8,12 --rho 1.0 --steps 1000 --seeds 2
python scale.py --axis width --values 32,64,128,256 --rho 1.0 --steps 1000 --seeds 2
python report.py         # rebuild report from results.json
```

## Layout

```
model.py        ERT (English-residual) + continuous baseline transformer
paraphrase.py   the fixed meaning-preserving / surface-randomizing channel P
capacity.py     covert side-channel capacity (optimal reader + learned collusion)
run.py          orchestrates E1–E4 -> results.json
scale.py        depth/width scaling sweeps -> scale_*.json
report.py / scaling_report.py   HTML reports
results.json    headline results; report_latest.html   browsable report
```

## Limitations

Toy scale (d=64, 4 layers, a 24-class synthetic meaning/surface task, CPU). The
"meaning classes" are defined, not learned; capacity is measured on this
structured task, not natural language; deeper/wider behavior is only probed by
the scaling sweeps. The claims are existence/mechanism demonstrations, not
scaled results.

## Context

Companion to a line of work on *paraphrase-as-bottleneck* for oversight (closing
covert LLM-to-LLM channels by forcing legible intermediates). Related:
router-level legibility for Mixture-of-Experts models.

Released under the MIT License.
