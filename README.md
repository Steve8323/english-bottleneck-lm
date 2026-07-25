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
| **E1** | **Capability double-dissociation** | At full paraphrase strength (ρ=1) the ERT keeps the **meaning** task at **M=0.996** (chance 0.042) while the **surface** task collapses to **S=0.247**. The bottleneck preserves semantics and destroys surface form — exactly the intended dissociation. Continuous baseline: M=S=1.0. The dissociation also survives with *learned* (distilled) meaning classes (see 2026-07-24 follow-up), though weakly. |
| **E2** | **Covert side-channel capacity** | Capacity is **not** monotone in ρ. It falls 2 → 0.166 → 0.003 bits/token as ρ goes 0 → 0.25 → 0.5, hits **exactly zero at ρ=0.75** (emitted distribution is uniform), then **reopens** at ρ=1 to ~0.003 bits/token via forced-swap anti-correlation. **ρ=0.75, not ρ=1, is the true zero-capacity operating point.** See the 2026-07-24 follow-up — the earlier "closed at ρ=1" reading was a heuristic-reader artifact. |
| **E3** | **Legibility** | You can literally print the words each layer emits and read the model "thinking" (see `report_latest.html`). |
| **E4** | **Paraphraser cost** | The fixed channel is **~72× cheaper** than a model forward pass (0.067 ms vs 4.8 ms/batch) — the "tiny paraphraser" is essentially free. |

The alignment pitch: a residual stream that is legible-by-construction and
provably low-capacity for covert coordination, at near-zero added cost. This is
a **toy testbed**, not a scaled model — see Limitations.

![results](report_latest.html) → open `report_latest.html` for the plots.

## Follow-up experiments (2026-07-24)

A 30-agent follow-up sprint (results in a private research archive) sharpened
the covert-channel analysis. Four updates, one of which contradicts the headline
E2 claim in detail.

**1. E2 was wrong: capacity is non-monotone, and ρ=1 reopens the channel.**
Exact algebra plus a 600k-trial Monte Carlo per ρ gives:

| ρ | covert capacity (bits/token) | note |
|-----|------------------------------|------|
| 0.00 | 2.0 | identity, surface passes freely |
| 0.25 | 0.166 | |
| 0.50 | 0.003 | |
| 0.75 | **0** | emitted distribution is *exactly* uniform — provably zero |
| 1.00 | ~0.003 | **reopens** (~0.012 bits per 4-slot message) |

At ρ=1 every content word *must* change (forced swap), and that anti-correlation
constraint itself carries information. A Bayes-optimal reader gets **51.9%** at
ρ=1 (permutation p=0.001); the repo's published *heuristic* reader was actually
**below chance (48.1%)** and missed the leak entirely. Re-analysing the saved E2
learned-reader logs also puts it above chance (**0.5177**, p=4.2e-4). There is
additionally a small "never-output-self" parity leak in the fixed ρ=1 channel
(exact Bayes **0.5185**). **Upshot:** the true zero-capacity operating point is
**ρ=0.75**, not ρ=1.

**2. Input-independence, not fixedness, is what makes the channel safe (SUGGESTIVE).**
A *learned* paraphraser trained to the same surface-randomizing objective did
**not** reopen the channel — all tested arms at chance (p=0.27). But the weakness
modes tested were the easy, input-independent ones; correlated *input-dependent*
deviations and co-training the paraphraser inside the task loop remain untested.

**3. The M/S double dissociation survives with learned meaning classes — weakly
(WEAKLY CONFIRMED).** With distilled meaning classes the dissociation holds
descriptively (classes recovered at ARI=1.000, near-trivially; M in 0.68–1.00
with S≈0.25 across runs). A preregistered per-seed criterion failed on one
undertrained seed. k-means on embedding geometry does **not** recover the classes
(ARI 0.03–0.06) — the dissociation is not just geometry.

**4. Concealment variant: no overt form to leak (NULL).** An ERT trained to
*hide* synonym identity while a colluding decoder reads it at 0.93–0.97 was
probed with activation noise at layer boundaries. Across **14,400 forwards, 0
reveal-capable emissions**: token-level supervision collapses emissions onto
canonical forms, so the hidden variable never surfaces in the emitted words —
there is no overt form to leak.

## The paraphrase channel

`P` acts on the sequence of words at each layer boundary. It is
**meaning-preserving, surface-randomizing**: with probability ρ it replaces each
content word with a uniformly random *other* synonym in the same meaning class.
It is **model-free** — a fixed, class-structured stochastic map (O(1) table
lookups), the "un-foolable floor" that closes the lexical channel with nothing
to optimize against. ρ=0 is the identity (surface passes freely); ρ=1 fully
randomizes surface while preserving meaning. The property that makes the channel
safe is the **input-independence** of the paraphrase conditional, not the fact
that it is fixed: a *learned* paraphraser trained to the same surface-randomizing
objective did not reopen the channel (see 2026-07-24 follow-up).

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
"meaning classes" are mainly defined, not learned (the 2026-07-24 follow-up adds
a weak learned-class result); capacity is measured on this structured task, not
natural language; deeper/wider behavior is only probed by the scaling sweeps. The
claims are existence/mechanism demonstrations, not scaled results.

## Context

Companion to a line of work on *paraphrase-as-bottleneck* for oversight (closing
covert LLM-to-LLM channels by forcing legible intermediates). Related:
router-level legibility for Mixture-of-Experts models.

Released under the MIT License.
