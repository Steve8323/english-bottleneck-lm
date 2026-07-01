"""scaling_report.py -- figures + HTML for the scaling/cost/benchmark study.
Honest synthesis, including the depth-scaling limitation."""
import os, json, time, base64, io
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
def L(f): return json.load(open(os.path.join(H, f)))
dep, wid = L("scale_depth.json"), L("scale_width.json")
bench, cost = L("benchmarks.json"), L("cost.json")
scr, dec = L("scratchpad_varhop.json"), L("decrement_varhop.json")
rt = L("retrain_deep.json")
chance = dep["chance_M"]

def b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()

# Fig A: depth
dx = [int(k) for k in sorted(dep["points"], key=int)]
dM = [dep["points"][str(k)]["ert_M"] for k in dx]
dB = [dep["points"][str(k)]["baseline_M"] for k in dx]
dC = [dep["points"][str(k)]["cap_learned"] for k in dx]
fA, a = plt.subplots(figsize=(6.6, 4.4))
a.plot(dx, dB, "s--", color="#555", lw=2, ms=7, label="normal model — capability")
a.plot(dx, dM, "o-", color="#1b7837", lw=2.6, ms=9, label="ERT — capability (meaning)")
a.plot(dx, dC, "^-", color="#d95f02", lw=2.4, ms=8, label="covert channel capacity")
a.axhline(chance, ls=":", color="#b00", lw=1, label=f"chance ({chance:.2f})")
a.axhline(0.5, ls=":", color="#999", lw=1)
a.set_xlabel("depth = #layers = #reasoning hops"); a.set_ylabel("accuracy")
a.set_title("Depth scaling: capability collapses, channel stays closed")
a.set_xticks(dx); a.set_ylim(0, 1.05); a.grid(alpha=.3); a.legend(fontsize=8.5)
figA = b64(fA)

# Fig B: width
wx = [int(k) for k in sorted(wid["points"], key=int)]
wM = [wid["points"][str(k)]["ert_M"] for k in wx]
wMe = [wid["points"][str(k)]["ert_M_std"] for k in wx]
wC = [wid["points"][str(k)]["cap_learned"] for k in wx]
fB, a = plt.subplots(figsize=(6.6, 4.4))
a.axhline(1.0, ls="--", color="#555", lw=2, label="normal model — capability")
a.errorbar(wx, wM, yerr=wMe, fmt="o-", color="#1b7837", lw=2.6, ms=9, capsize=4, label="ERT — capability (meaning)")
a.plot(wx, wC, "^-", color="#d95f02", lw=2.4, ms=8, label="covert channel capacity")
a.axhline(0.5, ls=":", color="#999", lw=1); a.axhline(chance, ls=":", color="#b00", lw=1)
a.set_xlabel("hidden width d (depth fixed = 4)"); a.set_ylabel("accuracy")
a.set_title("Width scaling: channel stays closed; capability noisy but holds")
a.set_xscale("log", base=2); a.set_xticks(wx); a.get_xaxis().set_major_formatter(plt.ScalarFormatter())
a.set_ylim(0, 1.05); a.grid(alpha=.3); a.legend(fontsize=8.5)
figB = b64(fB)

# Fig C: benchmark tax
names = list(bench["results"].keys())
import numpy as np
xb = np.arange(len(names)); w = 0.26
fC, a = plt.subplots(figsize=(6.8, 4.4))
a.bar(xb - w, [bench["results"][n]["baseline"] for n in names], w, color="#555", label="normal model")
a.bar(xb, [bench["results"][n]["ert_rho0"] for n in names], w, color="#1b7837", label="ERT ρ=0")
a.bar(xb + w, [bench["results"][n]["ert_rho1"] for n in names], w, color="#9970ab", label="ERT ρ=1")
a.axhline(bench["chance"], ls=":", color="#b00", label=f"chance ({bench['chance']:.2f})")
a.set_xticks(xb); a.set_xticklabels(names); a.set_ylabel("accuracy"); a.set_ylim(0, 1.05)
a.set_title("Reasoning battery: tax falls on conditional/dynamic computation")
a.grid(alpha=.3, axis="y"); a.legend(fontsize=8.5)
figC = b64(fC)

# Fig D: cost
cd = [c["d"] for c in cost["configs"]]
cp = [c["para_pct_of_layer_tok"] for c in cost["configs"]]
fD, a = plt.subplots(figsize=(6.4, 4.2))
a.plot(cd, cp, "o-", color="#1f78b4", lw=2.6, ms=9)
for c in cost["configs"]:
    a.annotate(f"{c['para_pct_of_layer_tok']:.2f}%", (c["d"], c["para_pct_of_layer_tok"]),
               textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
a.set_xlabel("hidden width d"); a.set_ylabel("paraphrase overhead (% of one layer)")
a.set_title("Cost of the fixed paraphraser shrinks as the model grows")
a.set_xscale("log", base=2); a.set_xticks(cd); a.get_xaxis().set_major_formatter(plt.ScalarFormatter())
a.grid(alpha=.3)
figD = b64(fD)

# Fig E: retrain per-layer decay
fE, a = plt.subplots(figsize=(6.6, 4.4))
for Ld, col in [("8", "#1b7837"), ("12", "#c0392b")]:
    per = rt["results"][Ld]["rho1_perlayer"]
    a.plot(range(1, len(per) + 1), per, "o-", color=col, lw=2.4, ms=7,
           label=f"depth {Ld} (ρ=1), final={np.mean(rt['results'][Ld]['rho1']):.2f}")
a.axhline(1.0, ls="--", color="#555", lw=1.5, label="ρ=0 / baseline ≈ 1.0")
a.set_xlabel("layer index"); a.set_ylabel("per-layer one-hop accuracy")
a.set_title("After 5× training: per-layer fidelity decays through depth (ρ=1)")
a.set_ylim(0, 1.05); a.grid(alpha=.3); a.legend(fontsize=8.5)
figE = b64(fE)


def trow(cells, hdr=False):
    t = "th" if hdr else "td"
    return "<tr>" + "".join(f"<{t}>{c}</{t}>" for c in cells) + "</tr>"

deprows = "".join(trow([k, f"{dep['points'][str(k)]['ert_M']:.3f}",
                        f"{dep['points'][str(k)]['baseline_M']:.2f}",
                        f"{dep['points'][str(k)]['cap_learned']:.3f}"]) for k in dx)
ts = time.strftime("%Y%m%d-%H%M%S")
html = f"""<!doctype html><html><head><meta charset=utf-8><title>Scaling & cost — English-bottleneck LM</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:28px auto;color:#16181d;padding:0 18px;line-height:1.55}}
h1{{font-size:24px}} h2{{margin-top:32px;border-bottom:2px solid #eee;padding-bottom:5px}}
img{{max-width:100%;border:1px solid #eee;border-radius:8px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.k{{background:#f6f8fa;border-radius:8px;padding:13px 17px;font-size:14px;margin:10px 0}}
.bad{{background:#fdf1f1;border-left:4px solid #c0392b}}
.good{{background:#f1f8f3;border-left:4px solid #1b7837}}
table{{border-collapse:collapse;font-size:13.5px;margin:8px 0}} td,th{{border:1px solid #ddd;padding:6px 11px;text-align:center}}
code{{background:#eef;padding:1px 5px;border-radius:4px}}</style></head><body>
<h1>English-bottleneck LM — scaling, cost & reasoning battery</h1>
<p class=k>Follow-up to the capability/capacity study. Question: does the paraphrase-bottleneck idea
<b>scale</b>, what does it <b>cost</b>, and how much does <b>reasoning</b> degrade? Small-scale
demonstrator, real trained runs; depth/width sweeps run by two sub-agents (Sonnet, Haiku).</p>

<h2>1 · Depth — the load-bearing result</h2>
<div class=grid><img src="data:image/png;base64,{figA}">
<div><table>{trow(['depth','ERT M','normal','covert cap'],True)}{deprows}</table>
<div class="k bad"><b>Capability does not survive depth.</b> Meaning accuracy falls 1.00→0.96→0.41→0.05
(≈chance) as layers/hops grow 2→4→8→12, while the normal model stays 1.00. The covert channel,
though, stays <b>closed</b> (~0.5) at every depth ≥4. Past depth 4 it is no longer a double
dissociation — <b>both</b> covert capacity and deep capability are gone.</div></div></div>
<div class=k>Mechanism: each layer re-quantizes to a word and adds paraphrase noise, so per-crossing
fidelity is &lt;1. A k-step dependent computation degrades like fidelity<sup>k</sup>. The very thing that
scrubs covert channels also erodes legitimate multi-hop signal. Depth-2 is the only leaky point
(too few crossings to randomize surface).</div>

<h2>2 · Width</h2>
<div class=grid><img src="data:image/png;base64,{figB}">
<div class="k good">At fixed depth 4, growing width 32→256 keeps the covert channel <b>closed</b> (~0.5)
— it lives at the vocabulary level, not in residual dimensionality. Meaning capability stays well
above chance but is <b>noisy across seeds</b> (e.g. d=128: 0.69±0.23): the discrete-bottleneck model
trains less stably than the continuous baseline.</div></div>

<h2>3 · Reasoning battery — where the tax lands</h2>
<div class=grid><img src="data:image/png;base64,{figC}">
<div><div class=k><b>chain</b> (pure meaning): tax ≈ 0. <b>varhop</b> (dynamic depth) and <b>ruleselect</b>
(instruction routing): large tax, present <i>even at ρ=0</i> — so it is the architecture, not the paraphrase.</div>
<div class="k bad"><b>Conditional/stateful computation is hard through the bottleneck.</b> I hypothesized a
legible scratchpad would fix <code>varhop</code>; it did not — persisting the instruction
(0.42→0.36) and a decrementing counter (→0.10) both <i>failed</i>. Anything needing state to survive
many sequential crossings degrades, same mechanism as depth.</div></div></div>

<h2>4 · Cost — the paraphraser is ~free and free-er at scale</h2>
<div class=grid><img src="data:image/png;base64,{figD}">
<div class="k good">The fixed paraphraser (a class-structured permutation, <b>no model</b>) costs
<b>1.4% of one layer at toy size, 0.02% at d=1024</b> — a layer is O(d²), the paraphrase is O(1)/token.
Adding it to a normal net costs ~that per layer: negligible.<br><br>
Caveat: a <i>model-based</i> paraphraser (an LLM rewriting text each layer) would be a whole extra
forward (autoregressive) per layer — 1×–100×+ the model, ×depth. The result that the surface channel
closes with the <b>fixed</b> table is what lets you avoid that cost.</div></div>

<h2>5 · "Just retrain" — does training harder fix the depth collapse?</h2>
<div class=grid><img src="data:image/png;base64,{figE}">
<div><table>{trow(['depth','ρ=0 (no para)','ρ=1','ρ=1 was'],True)}
{trow(['8', f"{np.mean(rt['results']['8']['rho0']):.3f}", f"{np.mean(rt['results']['8']['rho1']):.3f}", '0.408'])}
{trow(['12', f"{np.mean(rt['results']['12']['rho0']):.3f}", f"{np.mean(rt['results']['12']['rho1']):.3f}", '0.050'])}</table>
<div class="k good"><b>Two corrections from 5× training (cosine LR, sharper temp).</b>
(1) The discrete English bottleneck does <b>not</b> break depth: at ρ=0, depth-8→0.99 and
<b>depth-12→1.00</b> — the earlier collapse was under-training. (2) Under paraphrase, depth-8 ρ=1
recovered <b>0.41→0.89</b>.</div>
<div class="k bad">But depth-12 ρ=1 stayed low (0.18), and per-layer fidelity <b>decays through the
stack</b> (0.84→0.24). The real wall is <b>paraphrase-noise × depth</b>, not depth or discretization:
re-randomizing within-class every layer accumulates the model's residual per-layer error across
stochastic channels.</div></div></div>

<h2>Bottom line (revised)</h2>
<div class="k good"><b>The discrete legible-English bottleneck IS depth-scalable</b> on its own (ρ=0:
perfect to depth 12) and trains fine with enough steps — the first sweep was under-trained.</div>
<div class="k bad"><b>The open problem is noise×depth at high ρ.</b> Full-strength paraphrase between
every one of many layers accumulates error (depth-12 ρ=1 ≈ 0.18 even after 5× training). So there is a
real <b>ρ ⟷ trainable-depth tradeoff</b>: max channel-closure (ρ=1) caps how deep you can stack;
moderate ρ or paraphrasing only some boundaries trains deeper while still suppressing the channel.
Whether better methods (denoising-aware training, word residual connections) beat the depth-12 ρ=1 wall
is the next experiment — it's plausibly improvable, not obviously fundamental.</div>
<p style="color:#888;font-size:12px">Generated {ts}. Every number from a real trained run on CPU.</p>
</body></html>"""
out = os.path.join(H, f"scaling_report_{ts}.html"); open(out, "w").write(html)
open(os.path.join(H, "scaling_report_latest.html"), "w").write(html)
print(out)
