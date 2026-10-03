"""Behavioural analysis for Exp. 1 (word secrets), Exp. 2 (twists) and Exp. 3b (ablations).
Writes results/summary_behav.json and figures."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from twists import TWISTS
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(ROOT, "paper_draft", "figures"); os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(0)
NB = 4000
S = {}


def boot2(val, a, b, na, nb, fn=np.mean):
    """two-way cluster ('pigeonhole') bootstrap over factors a (e.g. word) and b (premise)."""
    val, a, b = np.asarray(val, float), np.asarray(a), np.asarray(b)
    cell = np.full((na, nb), np.nan); cnt = np.zeros((na, nb))
    sums = np.zeros((na, nb))
    np.add.at(sums, (a, b), val); np.add.at(cnt, (a, b), 1)
    out = []
    for _ in range(NB):
        ia = rng.integers(0, na, na); ib = rng.integers(0, nb, nb)
        s = sums[np.ix_(ia, ib)].sum(); c = cnt[np.ix_(ia, ib)].sum()
        out.append(s / max(c, 1))
    return np.array(out)


def ci(bs):
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def word_cond(expdir, g, cond):
    p = os.path.join(RESULTS, expdir, f"guess_{g}_{cond}.jsonl")
    if not os.path.exists(p):
        return None
    R = load_jsonl(p)
    d = dict(pi=np.array([r["pi"] for r in R]), wi=np.array([r["wi"] for r in R]),
             lo=np.array([r["lo1"] + r["lo2"] for r in R]), lo1=np.array([r["lo1"] for r in R]), lo2=np.array([r["lo2"] for r in R]),
             idlp=np.array([r["id_lp"] for r in R]), q=np.array([r["quality"] for r in R]),
             mention=np.array([r["mention"] for r in R]), nwords=np.array([r["nwords"] for r in R]), raw=R)
    d["afc"] = (d["lo"] > 0).astype(float)                       # order-debiased pair-level 2AFC
    d["afc_raw"] = ((d["lo1"] > 0).astype(float) + (d["lo2"] > 0)) / 2   # single-judgment accuracy
    d["id"] = (d["idlp"].argmax(1) == d["wi"]).astype(float)
    return d


def summarize(d, npi_offset=0):
    pi = d["pi"] - d["pi"].min()
    npi = pi.max() + 1
    out = {"n": int(len(pi))}
    for k in ["afc", "afc_raw", "id", "q", "mention", "nwords"]:
        out[k] = float(d[k].mean()); out[k + "_ci"] = ci(boot2(d[k], d["wi"], pi, 15, npi))
    return out


def diff_test(d1, d2, key="afc"):
    """paired (same word x premise cells) bootstrap for the difference d1 - d2."""
    pi = d1["pi"] - d1["pi"].min(); npi = pi.max() + 1
    c1 = np.zeros((15, npi)); c2 = np.zeros((15, npi)); n = np.zeros((15, npi))
    np.add.at(c1, (d1["wi"], pi), d1[key]); np.add.at(c2, (d2["wi"], d2["pi"] - d2["pi"].min()), d2[key]); np.add.at(n, (d1["wi"], pi), 1)
    diffs = []
    for _ in range(NB):
        ia = rng.integers(0, 15, 15); ib = rng.integers(0, npi, npi)
        diffs.append((c1[np.ix_(ia, ib)].sum() - c2[np.ix_(ia, ib)].sum()) / n[np.ix_(ia, ib)].sum())
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return dict(diff=float(d1[key].mean() - d2[key].mean()), ci=ci(diffs), p=float(max(p, 1 / NB)))


# ------------------------------------------------------------------ Exp 1
E1 = ["secret", "secret+irr", "secret+plan", "secret+selfplan", "decoy", "hide"]
LAB = {"secret": "Secret\n(don't reveal)", "secret+irr": "Secret +\nirrelevant outline", "secret+plan": "Secret +\noutline to follow",
       "secret+selfplan": "Secret +\nself-written outline", "decoy": "Decoy word", "hide": "Actively hide"}
for g in GUESSERS:
    res = {}
    D = {c: word_cond("exp1", g, c) for c in E1}
    if D["secret"] is None:
        continue
    for c in E1:
        if D[c] is not None:
            res[c] = summarize(D[c])
            res[c]["per_word_afc"] = {WORDS[w]: float(D[c]["afc"][D[c]["wi"] == w].mean()) for w in range(15)}
    d = D["decoy"]
    if d is not None and "decoy_lo1" in d["raw"][0]:
        dl = np.array([r["decoy_lo1"] + r["decoy_lo2"] for r in d["raw"]])
        res["decoy"]["afc_for_decoy_word"] = float((dl > 0).mean())
        res["decoy"]["afc_for_decoy_word_ci"] = ci(boot2(dl > 0, d["wi"], d["pi"], 15, 12))
    d = D["secret+selfplan"]
    if d is not None:
        pl = np.array([r["plan_id_lp"] for r in d["raw"]])
        res["secret+selfplan"]["plan_id"] = float((pl.argmax(1) == d["wi"]).mean())
        res["secret+selfplan"]["plan_mention"] = float(np.mean([r["plan_mention"] for r in d["raw"]]))
        pid = (pl.argmax(1) == d["wi"])
        res["secret+selfplan"]["afc_given_plan_identified"] = float(d["afc"][pid].mean()) if pid.any() else None
        res["secret+selfplan"]["afc_given_plan_not_identified"] = float(d["afc"][~pid].mean())
    tests = {}
    for a, b in [("secret", "secret+plan"), ("secret", "secret+irr"), ("secret+irr", "secret+plan"), ("secret", "decoy"),
                 ("secret", "hide"), ("secret", "secret+selfplan"), ("secret+selfplan", "secret+plan")]:
        if D[a] is not None and D[b] is not None:
            tests[f"{a} - {b}"] = {k: diff_test(D[a], D[b], k) for k in ["afc", "id"]}
    # identification prior on control stories
    ctrl = {}
    for c in ["none", "none+plan", "none+irr", "none+selfplan"]:
        p = os.path.join(RESULTS, "exp1", f"guess_{g}_{c}.jsonl")
        if os.path.exists(p):
            R = load_jsonl(p)
            ctrl[c] = dict(q=float(np.mean([r["quality"] for r in R])), nwords=float(np.mean([r["nwords"] for r in R])))
    S[f"exp1_{g}"] = dict(conds=res, tests=tests, controls=ctrl)

    conds = [c for c in E1 if c in res]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, (key, chance, title) in enumerate([("afc", 0.5, "2AFC discrimination (secret vs. matched no-secret story)"),
                                              ("id", 1 / 15, "15-way identification of the secret word")]):
        v = [res[c][key] for c in conds]
        err = np.array([[res[c][key] - res[c][key + "_ci"][0], res[c][key + "_ci"][1] - res[c][key]] for c in conds]).T
        cols = ["#4c72b0", "#8c8c8c", "#dd8452", "#c44e52", "#55a868", "#937860"][:len(conds)]
        ax[k].bar(range(len(conds)), v, yerr=err, color=cols, capsize=3)
        ax[k].axhline(chance, color="k", ls="--", lw=1)
        ax[k].set_xticks(range(len(conds))); ax[k].set_xticklabels([LAB[c].replace("\n", " ") for c in conds], fontsize=7.5, rotation=20, ha="right")
        ax[k].set_title(title, fontsize=9); ax[k].set_ylabel("accuracy")
        for i, x in enumerate(v):
            ax[k].text(i, x + err[1][i] + 0.01, f"{x:.2f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"exp1_{g}.pdf")); plt.close()

# ------------------------------------------------------------------ Exp 3b (ablation), held-out premises only
for g in GUESSERS:
    base = word_cond("exp1", g, "secret")
    if base is None or not os.path.isdir(os.path.join(RESULTS, "exp3")):
        continue
    keep = base["pi"] >= 6
    b = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    res = {"no ablation": summarize(b)}
    D3 = {"no ablation": b}
    for f in sorted(os.listdir(os.path.join(RESULTS, "exp3"))):
        if f.startswith(f"guess_{g}_"):
            c = f[len(f"guess_{g}_"):-6]
            D3[c] = word_cond("exp3", g, c); res[c] = summarize(D3[c])
            res[c]["vs_no_ablation"] = {k: diff_test(b, D3[c], k) for k in ["afc", "id", "q"]}
    S[f"exp3_{g}"] = res

# ------------------------------------------------------------------ Exp 2 (twists)
E2 = ["none", "none+irr", "none+plan", "twist_free", "twist", "twist+irr", "twist+plan", "twist+fullplan"]
for g in GUESSERS:
    res = {}
    D2 = {}
    for c in E2:
        p = os.path.join(RESULTS, "exp2", f"guess_{g}_{c}.jsonl")
        if not os.path.exists(p):
            continue
        R = load_jsonl(p)
        lp = np.array([r["lp"] for r in R]); pi = np.array([r["pi"] for r in R])
        prob = np.exp(lp); prob /= prob.sum(1, keepdims=True)
        if R[0]["ti"] is None:
            # no-secret control: guessability from the premise alone. Score against every candidate twist (balanced => 25%)
            ent = float(-(prob * np.log(prob)).sum(1).mean())
            res[c] = dict(n=len(R), acc=0.25, mean_max_prob=float(prob.max(1).mean()), entropy=ent,
                          choice_dist=[float((lp.argmax(1) == t).mean()) for t in range(4)], q=float(np.mean([r["quality"] for r in R])),
                          nwords=float(np.mean([r["nwords"] for r in R])))
            D2[c] = dict(prob=prob, pi=pi, lp=lp)
        else:
            ti = np.array([r["ti"] for r in R])
            acc = (lp.argmax(1) == ti).astype(float); pt = prob[np.arange(len(R)), ti]
            res[c] = dict(n=len(R), acc=float(acc.mean()), acc_ci=ci(boot2(acc, pi, ti, 12, 4)), p_true=float(pt.mean()),
                          p_true_ci=ci(boot2(pt, pi, ti, 12, 4)), q=float(np.mean([r["quality"] for r in R])),
                          nwords=float(np.mean([r["nwords"] for r in R])))
            D2[c] = dict(acc=acc, pt=pt, pi=pi, ti=ti, lp=lp)
    if "twist" not in res:
        continue
    # leakage relative to the matched no-secret control: P(guess = t | twist t) - P(guess = t | no twist), averaged over (premise, t)
    for c, ctl in [("twist", "none"), ("twist_free", "none"), ("twist+irr", "none+irr"), ("twist+plan", "none+plan"), ("twist+fullplan", "none")]:
        if c in D2 and ctl in D2:
            base = np.zeros((12, 4))
            for p_ in range(12):
                m = D2[ctl]["pi"] == p_
                base[p_] = [(D2[ctl]["lp"][m].argmax(1) == t).mean() for t in range(4)]
            res[c]["ctrl_rate_for_true_twist"] = float(base[D2[c]["pi"], D2[c]["ti"]].mean())
    tests = {}
    for a, b in [("twist", "twist+plan"), ("twist", "twist+irr"), ("twist+irr", "twist+plan"), ("twist", "twist+fullplan"), ("twist_free", "twist")]:
        if a in D2 and b in D2:
            out = {}
            for key in ["acc", "pt"]:
                c1 = np.zeros((12, 4)); c2 = np.zeros((12, 4))
                np.add.at(c1, (D2[a]["pi"], D2[a]["ti"]), D2[a][key]); np.add.at(c2, (D2[b]["pi"], D2[b]["ti"]), D2[b][key])
                n = len(D2[a]["pi"]) / 48
                diffs = []
                for _ in range(NB):
                    ia = rng.integers(0, 12, 12)
                    diffs.append((c1[ia].sum() - c2[ia].sum()) / (48 * n))
                diffs = np.array(diffs)
                out[key] = dict(diff=float(D2[a][key].mean() - D2[b][key].mean()), ci=ci(diffs),
                                p=float(max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1 / NB)))
            tests[f"{a} - {b}"] = out
    S[f"exp2_{g}"] = dict(conds=res, tests=tests)
    conds = [c for c in ["twist_free", "twist", "twist+irr", "twist+plan", "twist+fullplan"] if c in res]
    L2 = {"twist_free": "Twist known,\nno concealment\ninstruction", "twist": "Twist +\ndon't foreshadow", "twist+irr": "+ irrelevant\noutline",
          "twist+plan": "+ twist-blind\nopening outline", "twist+fullplan": "+ self-written\nfull outline"}
    fig, ax = plt.subplots(figsize=(6, 3.4))
    v = [res[c]["acc"] for c in conds]
    err = np.array([[res[c]["acc"] - res[c]["acc_ci"][0], res[c]["acc_ci"][1] - res[c]["acc"]] for c in conds]).T
    ax.bar(range(len(conds)), v, yerr=err, color=["#937860", "#4c72b0", "#8c8c8c", "#dd8452", "#c44e52"][:len(conds)], capsize=3)
    ax.axhline(0.25, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([L2[c] for c in conds], fontsize=7.5)
    ax.set_ylabel("4-way twist identification accuracy")
    for i, x in enumerate(v):
        ax.text(i, x + err[1][i] + 0.01, f"{x:.2f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"exp2_{g}.pdf")); plt.close()

json.dump(S, open(os.path.join(RESULTS, "summary_behav.json"), "w"), indent=1)
for k, v in S.items():
    print("=====", k)
    if k.startswith("exp1"):
        for c, r in v["conds"].items():
            print(f"{c:18s} afc {r['afc']:.3f} {np.round(r['afc_ci'], 3)} raw {r['afc_raw']:.3f} id {r['id']:.3f} {np.round(r['id_ci'], 3)} q {r['q']:.2f} mention {r['mention']:.3f} nw {r['nwords']:.0f}")
        for t, r in v["tests"].items():
            print("  ", t, {k2: (round(x['diff'], 3), np.round(x['ci'], 3).tolist(), x['p']) for k2, x in r.items()})
        print("  ", {k2: x for k2, x in v["conds"].get("secret+selfplan", {}).items() if "plan" in k2}, v["conds"].get("decoy", {}).get("afc_for_decoy_word"))
        print("  ctrl", v["controls"])
    elif k.startswith("exp3"):
        for c, r in v.items():
            print(f"{c:28s} afc {r['afc']:.3f} {np.round(r['afc_ci'], 3)} id {r['id']:.3f} q {r['q']:.2f} mention {r['mention']:.3f} nw {r['nwords']:.0f}",
                  {k2: (round(x['diff'], 3), x['p']) for k2, x in r.get("vs_no_ablation", {}).items()})
    else:
        for c, r in v["conds"].items():
            print(f"{c:16s}", {k2: (np.round(x, 3).tolist() if not isinstance(x, int) else x) for k2, x in r.items()})
        for t, r in v["tests"].items():
            print("  ", t, {k2: (round(x['diff'], 3), np.round(x['ci'], 3).tolist(), x['p']) for k2, x in r.items()})
