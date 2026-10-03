"""Leakage measurement with local guesser models (first-token forced choice, order-debiased).
usage: python guess.py <guesser> [exp1|exp2|exp3] """
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from twists import TWISTS, TG_USER
import numpy as np

g = sys.argv[1]
which = sys.argv[2:] or ["exp1", "exp2"]
GID = GUESSERS[g]
QUAL = ("Rate the overall quality of the following short story (coherence, fluency, and craft) on a scale from 1 "
        "(incoherent or broken) to 9 (excellent). Answer with ONLY a single digit.\n\nSTORY:\n{t}")


def ctrl_name(cond):
    suf = cond.split("+")[1] if "+" in cond else None
    return "none" + ("+" + suf if suf else "")


def pair_foils(sec, ctr):
    """one-to-one pairing of secret stories with control stories of the same premise and plan index."""
    pool = {}
    for c in ctr:
        pool.setdefault((c["pi"], c["k"]), []).append(c)
    used = {}
    foils = []
    for r in sec:
        key = (r["pi"], r["k"])
        i = used.get(key, 0)
        used[key] = i + 1
        foils.append(pool[key][i % len(pool[key])])
    return foils


def afc(sec, foils, words):
    msgs = []
    for r, f, w in zip(sec, foils, words):
        for first in (True, False):
            msgs.append(afc_messages(r["text"], f["text"], w, first))
    lp = np.array(local_choice(msgs, ["1", "2"], GID, desc="2afc")).reshape(len(sec), 2, 2)
    # log-odds in favour of the secret text, per order
    lo1 = lp[:, 0, 0] - lp[:, 0, 1]
    lo2 = lp[:, 1, 1] - lp[:, 1, 0]
    return lo1, lo2


def ident(texts):
    shifts = [0, 5, 10]
    msgs = []
    for t in texts:
        for sh in shifts:
            order = [(i + sh) % 15 for i in range(15)]
            msgs.append(id_messages(t, order))
    lp = np.array(local_choice(msgs, list(LETTERS), GID, desc="id15")).reshape(len(texts), 3, 15)
    out = np.zeros((len(texts), 15))
    for si, sh in enumerate(shifts):
        for i in range(15):
            out[:, (i + sh) % 15] += lp[:, si, i] / 3
    return out  # [n, 15] mean log-prob per word index


def quality(texts):
    lp = np.array(local_choice([[{"role": "user", "content": QUAL.format(t=t)}] for t in texts], list("123456789"), GID, desc="qual"))
    return (np.exp(lp) * np.arange(1, 10)).sum(1)


def do_word_conditions(expdir, conds, ctrl_dir=None, tag=""):
    ctrl_dir = ctrl_dir or expdir
    for cond in conds:
        out = os.path.join(expdir, f"guess_{g}_{cond}.jsonl")
        if os.path.exists(out) or not os.path.exists(os.path.join(expdir, cond + ".jsonl")):
            continue
        sec = load_jsonl(os.path.join(expdir, cond + ".jsonl"))
        base = cond.split("@")[0]
        ctr = load_jsonl(os.path.join(ctrl_dir, ("none" if base in ("decoy", "hide", "secret") else ctrl_name(base)) + ".jsonl"))
        foils = pair_foils(sec, ctr)
        lo1, lo2 = afc(sec, foils, [WORDS[r["wi"]] for r in sec])
        idl = ident([r["text"] for r in sec])
        q = quality([r["text"] for r in sec])
        rows = []
        for i, r in enumerate(sec):
            rows.append(dict(pi=r["pi"], wi=r["wi"], s=r["s"], lo1=float(lo1[i]), lo2=float(lo2[i]), id_lp=idl[i].tolist(),
                             quality=float(q[i]), mention=mentions(r["text"], WORDS[r["wi"]]), nwords=len(r["text"].split())))
        if base == "decoy":
            d1, d2 = afc(sec, foils, [r["decoy"] for r in sec])
            for row, a, b in zip(rows, d1, d2):
                row["decoy_lo1"], row["decoy_lo2"] = float(a), float(b)
        if "plan" in sec[0]:  # self-written plans: can the secret be identified from the plan itself?
            pl = ident([r["plan"] for r in sec])
            for row, a, r in zip(rows, pl, sec):
                row["plan_id_lp"] = a.tolist()
                row["plan_mention"] = mentions(r["plan"], WORDS[r["wi"]])
        save_jsonl(rows, out)
        print(cond, "2afc debiased", np.mean((lo1 + lo2) > 0), "id acc", np.mean(idl.argmax(1) == np.array([r["wi"] for r in sec])), flush=True)


if "exp1" in which:
    D = os.path.join(RESULTS, "exp1")
    do_word_conditions(D, ["secret", "secret+plan", "secret+irr", "decoy", "hide", "secret+selfplan"])
    # controls: quality + identification "prior" (what the guesser says when there is no secret)
    for cond in ["none", "none+plan", "none+irr", "none+selfplan"]:
        out = os.path.join(D, f"guess_{g}_{cond}.jsonl")
        if os.path.exists(out) or not os.path.exists(os.path.join(D, cond + ".jsonl")):
            continue
        ctr = load_jsonl(os.path.join(D, cond + ".jsonl"))
        idl = ident([r["text"] for r in ctr]); q = quality([r["text"] for r in ctr])
        save_jsonl([dict(pi=r["pi"], j=r["j"], id_lp=idl[i].tolist(), quality=float(q[i]), nwords=len(r["text"].split()))
                    for i, r in enumerate(ctr)], out)

if "exp3" in which:
    D = os.path.join(RESULTS, "exp3")
    conds = sorted(f[:-6] for f in os.listdir(D) if f.endswith(".jsonl") and not f.startswith("guess_"))
    do_word_conditions(D, conds, ctrl_dir=os.path.join(RESULTS, "exp1"))

if "exp2" in which:
    D = os.path.join(RESULTS, "exp2")
    for cond in ["none", "twist", "none+plan", "twist+plan", "twist+irr", "none+irr", "twist_free", "twist+fullplan"]:
        out = os.path.join(D, f"guess_{g}_{cond}.jsonl")
        if os.path.exists(out) or not os.path.exists(os.path.join(D, cond + ".jsonl")):
            continue
        st = load_jsonl(os.path.join(D, cond + ".jsonl"))
        msgs = []
        for r in st:
            t = TWISTS[r["pi"]]
            for sh in range(4):
                opts = "\n".join(f"{LETTERS[i]}. {t['twists'][(i + sh) % 4]}" for i in range(4))
                msgs.append([{"role": "user", "content": TG_USER.format(premise=t["premise"], t=r["text"], opts=opts)}])
        lp = np.array(local_choice(msgs, list("ABCD"), GID, desc="twist4")).reshape(len(st), 4, 4)
        agg = np.zeros((len(st), 4))
        for sh in range(4):
            for i in range(4):
                agg[:, (i + sh) % 4] += lp[:, sh, i] / 4
        q = quality([r["text"] for r in st])
        save_jsonl([dict(pi=r["pi"], ti=r["ti"], s=r["s"], lp=agg[i].tolist(), raw=lp[i].tolist(), quality=float(q[i]),
                         nwords=len(r["text"].split())) for i, r in enumerate(st)], out)
        if st[0]["ti"] is not None:
            print(cond, "4-way acc", np.mean(agg.argmax(1) == np.array([r["ti"] for r in st])), flush=True)
