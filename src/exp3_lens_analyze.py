"""Refreshes only the logit-lens / output-probability section of results/summary_mech.json (after exp3_lens_fix.py)."""
"""Experiment 3a analysis: is the secret linearly present in the residual stream during story generation,
is that explained by the text itself, how do conditions differ, and does strength predict leakage?"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ACT = os.path.join(ROOT, "data", "acts"); D = os.path.join(RESULTS, "exp1")
FIG = os.path.join(ROOT, "paper_draft", "figures"); os.makedirs(FIG, exist_ok=True)
LAYERS = list(range(0, 49, 4)); LSUB = [8, 16, 24, 32, 40, 48]
S = {}


def load(name, kind="mean"):
    """float16 storage overflowed in Gemma's massive-activation dimension (2339) at layers 25-47: zero non-finite
    entries and drop that dimension everywhere."""
    X = np.load(os.path.join(ACT, f"{name}_{kind}.npy")).astype(np.float32)
    X[~np.isfinite(X)] = 0
    if X.shape[-1] == 3840:
        X[..., 2339] = 0
    return X


def meta(cond):
    rows = load_jsonl(os.path.join(D, cond + ".jsonl"))
    wi = np.array([r["wi"] if "wi" in r else (r["pi"] + r["j"]) % 15 for r in rows])
    return np.array([r["pi"] for r in rows]), wi


def fit(Xtr, ytr):
    """standardise -> PCA(128, fit on the training fold) -> multinomial logistic regression."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    p = PCA(128, random_state=0).fit((Xtr - mu) / sd)
    clf = LogisticRegression(C=0.1, max_iter=500).fit(p.transform((Xtr - mu) / sd), ytr)
    return lambda X: clf.predict_log_proba(p.transform((X - mu) / sd))


def cv_decode(X, y, pi, Xtest=None):
    """leave-2-premises-out CV. Returns log-prob matrix [n, 15] for held-out stories (probe applied to Xtest if given)."""
    Xt = X if Xtest is None else Xtest
    out = np.zeros((len(y), 15))
    for f in range(6):
        te = (pi % 6) == f
        out[te] = fit(X[~te], y[~te])(Xt[te])
    return out


CONDS = ["secret", "secret+irr", "secret+plan", "decoy", "hide"]
ACTS = {}
for c in CONDS + ["none", "none+plan"]:
    n = c.replace("+", "_")
    if os.path.exists(os.path.join(ACT, n + "_A_mean.npy")):
        ACTS[c] = (load(n + "_A"), load(n + "_B"))
META = {c: meta(c) for c in ACTS}

# ---------------------------------------------------------------- 4. logit lens / output probability of the secret word
lens = {}
for c in ACTS:
    n = c.replace("+", "_")
    LA, LB = load(n + "_A", "lens"), load(n + "_B", "lens"); pi, wi = META[c]
    # exp3_acts.py double-normalised the (already final-normed) last hidden state; replace the last two columns
    # (layer-48 lens, output log-prob) with the values recomputed by exp3_lens_fix.py (raw output logit, log-prob)
    fx = os.path.join(ACT, f"{n}_A_out.npy")
    if os.path.exists(fx):
        FA, FB = np.load(fx), np.load(os.path.join(ACT, f"{n}_B_out.npy"))
        LA[:, 5:7] = FA; LB[:, 5:7] = FB
    idx = np.arange(len(wi))
    own = lambda L: L[idx, :, wi]                                        # [n, 7]
    oth = lambda L: (L.sum(2) - L[idx, :, wi]) / 14
    dA = own(LA) - own(LB); dO = oth(LA) - oth(LB)                       # effect of the secret prompt on own / other words
    contrast = dA - dO
    # cluster over words for a simple CI
    wmeans = np.stack([contrast[wi == w].mean(0) for w in range(15)])
    lens[c] = dict(layers=LSUB + ["output_logprob"], own_minus_neutral=dA.mean(0).tolist(), other_minus_neutral=dO.mean(0).tolist(),
                   contrast=contrast.mean(0).tolist(), contrast_sem_over_words=(wmeans.std(0, ddof=1) / np.sqrt(15)).tolist(),
                   frac_words_positive=(wmeans > 0).mean(0).tolist(),
                   own_logprob_A=float(own(LA)[:, -1].mean()), own_logprob_B=float(own(LB)[:, -1].mean()))
    print("lens", c, "contrast", np.round(lens[c]["contrast"], 3), "sem", np.round(lens[c]["contrast_sem_over_words"], 3),
          "own lp A/B", round(lens[c]["own_logprob_A"], 2), round(lens[c]["own_logprob_B"], 2))
S["logit_lens"] = lens

fig, ax = plt.subplots(figsize=(5.2, 3.3))
keep = [0, 1, 2, 3, 4, 6]  # layer-48 lens equals the output distribution; show it once
x = np.arange(len(keep))
for c in lens:
    ax.errorbar(x, np.array(lens[c]["contrast"])[keep], yerr=np.array(lens[c]["contrast_sem_over_words"])[keep], marker="o", ms=3, capsize=2, label=c.replace("none", "control story"))
ax.axhline(0, color="k", lw=1, ls="--"); ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in LSUB[:-1]] + ["output\nlog-prob"], fontsize=8)
ax.set_ylabel("secret-word token: effect of secret prompt\n(own word minus other 14 words)", fontsize=8); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "logit_lens.pdf")); plt.close()


P = os.path.join(RESULTS, "summary_mech.json")
S0 = json.load(open(P)) if os.path.exists(P) else {}
S0["logit_lens"] = S["logit_lens"]
json.dump(S0, open(P, "w"), indent=1)
