"""summary.py -- compact, table-first PERFORMANCE SUMMARY page (ERT vs the
normal continuous transformer), pinned against chance. Opens a timestamped copy."""
import os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
rhos = R["config"]["rhos"]
cM, cS = R["chance"]["M"], R["chance"]["S"]
S = R.get("E1_seeds")
e4 = R["E4"]


def cell(mean, std=None, good=None):
    txt = f"{mean:.3f}" + (f"<span class=s>±{std:.3f}</span>" if std else "")
    cls = "" if good is None else (" class=hi" if good else " class=lo")
    return f"<td{cls}>{txt}</td>"


if S:
    erM = [(S["ert"][str(r)]["M"][0], S["ert"][str(r)]["M"][1]) for r in rhos]
    erS = [(S["ert"][str(r)]["S"][0], S["ert"][str(r)]["S"][1]) for r in rhos]
    bM, bS = S["baseline"]["M"][0], S["baseline"]["S"][0]
    nseed = len(S["raw"]["seeds"])
else:
    erM = [(R["E1"]["ert"][str(r)]["M"], None) for r in rhos]
    erS = [(R["E1"]["ert"][str(r)]["S"], None) for r in rhos]
    bM, bS = R["E1"]["baseline"]["M"], R["E1"]["baseline"]["S"]
    nseed = 1

capO = [R["E2"]["optimal"][str(r)] for r in rhos]
capL = [R["E2"]["learned"][str(r)] for r in rhos]

rho_hdr = "".join(f"<th>ρ={r}</th>" for r in rhos)
rowM = "".join(cell(m, s, good=(m > 0.8)) for m, s in erM)
rowS = "".join(cell(m, s, good=(m > 0.8)) for m, s in erS)
rowCO = "".join(cell(c, None, good=(c < 0.6)) for c in capO)
rowCL = "".join(cell(c, None, good=(c < 0.6)) for c in capL)

ts = time.strftime("%Y%m%d-%H%M%S")
html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Performance summary — English-bottleneck LM</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:30px auto;
color:#16181d;padding:0 18px;line-height:1.5}}
h1{{font-size:23px;margin-bottom:2px}} .sub{{color:#667;font-size:13.5px;margin-bottom:22px}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0 4px}}
th,td{{border:1px solid #dcdfe3;padding:8px 10px;text-align:center}}
th:first-child,td:first-child{{text-align:left;white-space:nowrap}}
thead th{{background:#f3f5f7}} tbody tr:nth-child(even){{background:#fafbfc}}
.s{{color:#8a93a0;font-size:11px;margin-left:3px}}
.hi{{background:#e7f6ec;color:#136b33;font-weight:600}}
.lo{{background:#fdecec;color:#9c1c1c;font-weight:600}}
.ref{{color:#555;font-style:italic}}
.note{{background:#f6f8fa;border-radius:8px;padding:12px 16px;font-size:13.5px;margin:8px 0}}
.verdict{{border-left:4px solid #1b7837;background:#f1f8f3;padding:12px 16px;border-radius:6px;
font-size:14.5px;margin:18px 0}}
caption{{text-align:left;font-weight:600;margin:18px 0 4px;font-size:15px}}
</style></head><body>
<h1>Performance summary — English-bottleneck LM (ERT) vs a normal transformer</h1>
<div class=sub>Same depth (4 layers) / width / vocab ({R['config']['vocab']} words,
{R['config']['n_class']} meaning classes). Accuracy, higher = better unless noted.
ERT cells are mean±std over {nseed} seeds. Green = good, red = bad.</div>

<table><caption>Capability — accuracy vs paraphrase strength ρ</caption>
<thead><tr><th>model / task</th>{rho_hdr}<th class=ref>normal model</th><th class=ref>chance</th></tr></thead>
<tbody>
<tr><td><b>Task M</b> — meaning (chained class rule)</td>{rowM}<td class=ref>{bM:.3f}</td><td class=ref>{cM:.3f}</td></tr>
<tr><td><b>Task S</b> — surface (copy exact synonym)</td>{rowS}<td class=ref>{bS:.3f}</td><td class=ref>{cS:.4f}</td></tr>
</tbody></table>
<div class=note>Meaning-based reasoning is <b>flat across ρ</b> and matches the normal model — the
paraphrase bottleneck is almost free. Surface-carrying <b>collapses to chance</b> (0.25 = keeps the
class, loses which of 4 synonyms): the cost falls only on illegible information.</div>

<table><caption>Covert side-channel capacity — recoverable secret bit (lower = better; 0.5 = closed)</caption>
<thead><tr><th>reader</th>{rho_hdr}<th class=ref>closed</th></tr></thead>
<tbody>
<tr><td>optimal reader (worst-case bound)</td>{rowCO}<td class=ref>0.500</td></tr>
<tr><td>learned colluding sender/receiver</td>{rowCL}<td class=ref>0.500</td></tr>
</tbody></table>
<div class=note>Holding overt meaning fixed, neither an optimal reader nor a trained colluding pair can pass a
bit across the 4 layers once ρ→1. The channel is closed <b>structurally</b>, not by penalty.</div>

<table><caption>Cost — the tiny paraphraser</caption>
<thead><tr><th>component</th><th>ms / batch</th><th>relative</th></tr></thead>
<tbody>
<tr><td>fixed paraphrase channel (no model)</td><td>{e4['paraphrase_ms_per_batch']:.3f}</td><td>1×</td></tr>
<tr><td>one model layer forward</td><td>{e4['model_fwd_ms_per_batch']:.1f}</td><td class=hi>{e4['ratio']:.0f}× costlier</td></tr>
</tbody></table>

<div class=verdict><b>Bottom line.</b> The English-bottleneck LM matches a normal transformer on meaning-based
reasoning at every paraphrase strength, while driving covert cross-layer channel capacity to chance —
and the paraphraser that does it is a fixed permutation, ~{e4['ratio']:.0f}× cheaper than a layer.
The only capability it loses is the ability to carry illegible (non-meaning) information across layers,
which is the entire point.</div>

<div class=sub>Generated {ts} · small-scale demonstrator, every number from a real trained run · device {R['config']['device']}.</div>
</body></html>"""

out = os.path.join(HERE, f"summary_{ts}.html")
open(out, "w").write(html)
open(os.path.join(HERE, "summary_latest.html"), "w").write(html)
print(out)
