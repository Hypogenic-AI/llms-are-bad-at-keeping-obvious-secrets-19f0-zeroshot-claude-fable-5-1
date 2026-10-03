"""Builds the LaTeX tables in paper_draft/tables from results/summary_*.json (no hand-typed numbers)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, RESULTS

T = os.path.join(ROOT, "paper_draft", "tables"); os.makedirs(T, exist_ok=True)
B = json.load(open(os.path.join(RESULTS, "summary_behav.json")))
M = json.load(open(os.path.join(RESULTS, "summary_mech.json"))) if os.path.exists(os.path.join(RESULTS, "summary_mech.json")) else {}


def f(x, ci=None, d=2):
    s = f"{x:.{d}f}"
    if ci is not None:
        s += f" [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"
    return s


def star(p):
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


# ---------------------------------------------------------------- Table 1: Exp 1
LAB = {"secret": "Secret (don't reveal)", "secret+irr": "Secret + irrelevant outline", "secret+plan": "Secret + outline to follow",
       "secret+selfplan": "Secret + self-written outline", "decoy": "Secret + decoy word", "hide": "Secret, actively hide"}
rows = []
g1, g2 = B["exp1_gemma12"]["conds"], B["exp1_qwen14"]["conds"]
for c in ["secret", "secret+irr", "secret+plan", "secret+selfplan", "decoy", "hide"]:
    a, b = g1[c], g2[c]
    rows.append(f"{LAB[c]} & {f(a['afc'], a['afc_ci'])} & {f(a['id'], a['id_ci'])} & {f(b['afc'], b['afc_ci'])} & {f(b['id'], b['id_ci'])} & {a['q']:.2f} & {a['nwords']:.0f} \\\\")
ctrl = B["exp1_gemma12"]["controls"]
rows.append("\\midrule")
for c, lab in [("none", "No secret"), ("none+irr", "No secret + irrelevant outline"), ("none+plan", "No secret + outline to follow"), ("none+selfplan", "No secret + self-written outline")]:
    if c in ctrl:
        rows.append(f"{lab} & -- & -- & -- & -- & {ctrl[c]['q']:.2f} & {ctrl[c]['nwords']:.0f} \\\\")
open(os.path.join(T, "exp1.tex"), "w").write(r"""\begin{tabular}{lccccrr}
\toprule
 & \multicolumn{2}{c}{Guesser: Gemma-3-12B (writer)} & \multicolumn{2}{c}{Guesser: Qwen2.5-14B} & & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Writer condition & 2AFC [95\% CI] & 15-way ID [95\% CI] & 2AFC [95\% CI] & 15-way ID [95\% CI] & Quality & Words \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# paired contrasts
rows = []
for k, lab in [("secret - secret+plan", "Secret $-$ (Secret + outline)"), ("secret - secret+irr", "Secret $-$ (Secret + irrelevant outline)"),
               ("secret+irr - secret+plan", "(Secret + irrelevant) $-$ (Secret + outline)"), ("secret - secret+selfplan", "Secret $-$ (Secret + self-written outline)"),
               ("secret - decoy", "Secret $-$ Decoy"), ("secret - hide", "Secret $-$ Actively hide")]:
    cells = []
    for g in ["gemma12", "qwen14"]:
        t = B[f"exp1_{g}"]["tests"][k]
        for key in ["afc", "id"]:
            cells.append(f"{t[key]['diff']:+.2f} [{t[key]['ci'][0]:+.2f}, {t[key]['ci'][1]:+.2f}]{star(t[key]['p'])}")
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
open(os.path.join(T, "exp1_tests.tex"), "w").write(r"""\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{Gemma-3-12B guesser} & \multicolumn{2}{c}{Qwen2.5-14B guesser} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Contrast & $\Delta$ 2AFC & $\Delta$ 15-way ID & $\Delta$ 2AFC & $\Delta$ 15-way ID \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- Table 2: Exp 2
L2 = {"twist_free": "Twist known, no concealment instruction", "twist": "Twist + don't reveal/foreshadow", "twist+irr": "\\quad + irrelevant outline",
      "twist+plan": "\\quad + twist-blind opening outline", "twist+fullplan": "\\quad + self-written full outline"}
rows = []
if "exp2_gemma12" in B:
    c1 = B["exp2_gemma12"]["conds"]; c2 = B.get("exp2_qwen14", {}).get("conds", {})
    for c in ["twist_free", "twist", "twist+irr", "twist+plan", "twist+fullplan"]:
        if c not in c1:
            continue
        a = c1[c]; b = c2.get(c)
        bs = f"{f(b['acc'], b['acc_ci'])} & {b['p_true']:.2f}" if b else "-- & --"
        rows.append(f"{L2[c]} & {f(a['acc'], a['acc_ci'])} & {a['p_true']:.2f} & {bs} & {a['q']:.2f} & {a['nwords']:.0f} \\\\")
    rows.append("\\midrule")
    for c, lab in [("none", "No twist"), ("none+irr", "No twist + irrelevant outline"), ("none+plan", "No twist + opening outline")]:
        if c in c1:
            rows.append(f"{lab} & 0.25 (by design) & -- & 0.25 (by design) & -- & {c1[c]['q']:.2f} & {c1[c]['nwords']:.0f} \\\\")
    open(os.path.join(T, "exp2.tex"), "w").write(r"""\begin{tabular}{lcccccr}
\toprule
 & \multicolumn{2}{c}{Guesser: Gemma-3-12B} & \multicolumn{2}{c}{Guesser: Qwen2.5-14B} & & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Writer condition & 4-way acc.\ [95\% CI] & $P(\text{true})$ & 4-way acc.\ [95\% CI] & $P(\text{true})$ & Quality & Words \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
    rows = []
    for k, lab in [("twist_free - twist", "No instruction $-$ Don't foreshadow"), ("twist - twist+plan", "Don't foreshadow $-$ (+ twist-blind outline)"),
                   ("twist - twist+irr", "Don't foreshadow $-$ (+ irrelevant outline)"), ("twist+irr - twist+plan", "(+ irrelevant) $-$ (+ twist-blind outline)"),
                   ("twist - twist+fullplan", "Don't foreshadow $-$ (+ self-written outline)")]:
        cells = []
        for g in ["gemma12", "qwen14"]:
            t = B.get(f"exp2_{g}", {}).get("tests", {}).get(k)
            if t is None:
                cells += ["--", "--"]; continue
            for key in ["acc", "pt"]:
                cells.append(f"{t[key]['diff']:+.3f} [{t[key]['ci'][0]:+.3f}, {t[key]['ci'][1]:+.3f}]{star(t[key]['p'])}")
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    open(os.path.join(T, "exp2_tests.tex"), "w").write(r"""\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{Gemma-3-12B guesser} & \multicolumn{2}{c}{Qwen2.5-14B guesser} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Contrast & $\Delta$ accuracy & $\Delta P(\text{true})$ & $\Delta$ accuracy & $\Delta P(\text{true})$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- Table 3: ablations
L3 = {"no ablation": "No intervention", "secret@abl_own_gen@delta": "Own secret dir.~($\\Delta$), story positions",
      "secret@abl_own_all@delta": "Own secret dir.~($\\Delta$), all positions", "secret@abl_own_gen@total": "Own secret dir.~(total), story positions",
      "secret@abl_own_all@total": "Own secret dir.~(total), all positions", "secret@abl_other_all@delta": "Other word's dir.~($\\Delta$), all positions",
      "secret@abl_other_all@total": "Other word's dir.~(total), all positions", "secret@abl_rand_all": "Random dir., all positions"}
if "exp3_gemma12" in B:
    rows = []
    e1 = B["exp3_gemma12"]; e2 = B.get("exp3_qwen14", {})
    for c in L3:
        if c not in e1:
            continue
        a = e1[c]; b = e2.get(c)
        pa = a.get("vs_no_ablation", {})
        sa = f"{f(a['afc'], a['afc_ci'])}{star(pa['afc']['p']) if pa else ''} & {f(a['id'], a['id_ci'])}{star(pa['id']['p']) if pa else ''}"
        if b:
            pb = b.get("vs_no_ablation", {})
            sb = f"{f(b['afc'], b['afc_ci'])}{star(pb['afc']['p']) if pb else ''} & {f(b['id'], b['id_ci'])}{star(pb['id']['p']) if pb else ''}"
        else:
            sb = "-- & --"
        rows.append(f"{L3[c]} & {sa} & {sb} & {a['q']:.2f}{star(pa['q']['p']) if pa else ''} & {a['mention']:.3f} & {a['nwords']:.0f} \\\\")
    open(os.path.join(T, "exp3.tex"), "w").write(r"""\begin{tabular}{lccccccr}
\toprule
 & \multicolumn{2}{c}{Guesser: Gemma-3-12B} & \multicolumn{2}{c}{Guesser: Qwen2.5-14B} & & & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Intervention during generation & 2AFC & 15-way ID & 2AFC & 15-way ID & Quality & Literal & Words \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------- Table 4: decoding
if M:
    dec = M["decode_by_layer"]; layers = dec["layers"]
    L4 = {"secret": "Secret", "secret+irr": "Secret + irrelevant outline", "secret+plan": "Secret + outline to follow", "decoy": "Decoy", "hide": "Actively hide",
          "none": "Control story, secret in prompt", "none+plan": "Control story + outline, secret in prompt"}
    rows = []
    for c in L4:
        if c not in dec["acc"]:
            continue
        cells = []
        for l in [8, 24, 48]:
            i = layers.index(l)
            cells.append(" & ".join(f"{dec['acc'][c][k][i]:.2f}" for k in ["A", "B", "delta"]))
        rows.append(f"{L4[c]} & " + " & ".join(cells) + " \\\\")
    open(os.path.join(T, "decode.tex"), "w").write(r"""\begin{tabular}{lccccccccc}
\toprule
 & \multicolumn{3}{c}{Layer 8} & \multicolumn{3}{c}{Layer 24} & \multicolumn{3}{c}{Layer 48} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Condition & full & text & $\Delta$ & full & text & $\Delta$ & full & text & $\Delta$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
    # strength vs leakage
    rows = []
    for g, gl in [("gemma12", "Gemma-3-12B"), ("qwen14", "Qwen2.5-14B")]:
        for c in ["secret", "secret+irr", "secret+plan", "decoy", "hide"]:
            cells = []
            for name in ["A", "B", "delta"]:
                s = M["strength_vs_leak"].get(f"{g}|{c}|L24|{name}")
                if s is None:
                    cells += ["--", "--"]; continue
                cells.append(f"{s['afc_within_word'][0]:+.2f}{star(s['afc_within_word'][1])}")
                cells.append(f"{s['id_within_word'][0]:+.2f}{star(s['id_within_word'][1])}")
            rows.append(f"{gl} & {LAB[c]} & " + " & ".join(cells) + " \\\\")
    open(os.path.join(T, "strength.tex"), "w").write(r"""\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{2}{c}{full state} & \multicolumn{2}{c}{text only} & \multicolumn{2}{c}{$\Delta$ (secret-prompt contribution)} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Guesser & Condition & 2AFC & ID & 2AFC & ID & 2AFC & ID \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
print("tables written")

# ---------------------------------------------------------------- appendix: per-word 2AFC
rows = []
from common import WORDS
for w in WORDS:
    cells = []
    for g in ["gemma12", "qwen14"]:
        for c in ["secret", "secret+irr", "secret+plan", "hide"]:
            cells.append(f"{B[f'exp1_{g}']['conds'][c]['per_word_afc'][w]:.2f}")
    rows.append(f"{w} & " + " & ".join(cells) + " \\\\")
open(os.path.join(T, "words.tex"), "w").write(r"""\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{4}{c}{Gemma-3-12B guesser} & \multicolumn{4}{c}{Qwen2.5-14B guesser} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Word & Secret & + irrelevant & + outline & Hide & Secret & + irrelevant & + outline & Hide \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")
