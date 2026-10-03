"""Experiment 2: plot-level secrets (twists). Gemma-3-12B writes only the opening scene."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from twists import *

OUT = os.path.join(RESULTS, "exp2")
os.makedirs(OUT, exist_ok=True)
NP, NS = 3, 3  # plans per premise, samples per (premise, twist)


def run(name, rows, prompts, **kw):
    path = os.path.join(OUT, name + ".jsonl")
    if os.path.exists(path):
        return load_jsonl(path)
    outs = generate(prompts, desc=name, **kw)
    for r, o in zip(rows, outs):
        r["text"] = o
    save_jsonl(rows, path)
    return rows


rows = [dict(pi=pi, k=k) for pi in range(len(TWISTS)) for k in range(NP)]
plans = run("plans", rows, [T_OUTLINE.format(premise=TWISTS[r["pi"]]["premise"]) for r in rows], max_new_tokens=350, seed=101)
PL = {(r["pi"], r["k"]): r["text"] for r in plans}


def mk(cond):
    if cond.startswith("none"):
        return [dict(cond=cond, pi=pi, ti=None, s=s, k=s % NP) for pi in range(len(TWISTS)) for s in range(4 * NS)]
    return [dict(cond=cond, pi=pi, ti=ti, s=s, k=s) for pi in range(len(TWISTS)) for ti in range(4) for s in range(NS)]


def prompt_for(r):
    t = TWISTS[r["pi"]]
    tw = t["twists"][r["ti"]] if r["ti"] is not None else None
    c = "twist" + r["cond"][len("twist_free"):] if r["cond"].startswith("twist_free") else r["cond"]
    p = twist_prompt(c, t["premise"], tw, PL[(r["pi"], r["k"])], PL[((r["pi"] + 5) % len(TWISTS), r["k"])])
    if r["cond"].startswith("twist_free"):  # twist known, no instruction to conceal it
        p = p.replace(T_SECRET.format(twist=tw),
                      f"\n\nTWIST (to be revealed in the final scene, which you are NOT writing now): {tw}.")
    return p


seed = 110
for cond in ["none", "twist", "none+plan", "twist+plan", "twist+irr", "none+irr", "twist_free"]:
    rows = mk(cond)
    run(cond, rows, [prompt_for(r) for r in rows], max_new_tokens=650, batch_size=24, seed=seed)
    seed += 1

# full self-written outline knowing the twist, then scene 1
rows = mk("twist+fullplan")
first = [T_FULLPLAN.format(premise=TWISTS[r["pi"]]["premise"], twist=TWISTS[r["pi"]]["twists"][r["ti"]]) for r in rows]
prows = run("twist+fullplan_plans", [dict(r) for r in rows], first, max_new_tokens=600, batch_size=24, seed=seed)
msgs = [[{"role": "user", "content": f}, {"role": "assistant", "content": p["text"]}, {"role": "user", "content": T_FULLPLAN_WRITE}]
        for f, p in zip(first, prows)]
for r, p in zip(rows, prows):
    r["plan"] = p["text"]
run("twist+fullplan", rows, msgs, max_new_tokens=650, batch_size=24, seed=seed + 1)
print("done")
