"""Experiment 1: word secrets. Generates plans and stories for all conditions with Gemma-3-12B."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *

OUT = os.path.join(RESULTS, "exp1")
os.makedirs(OUT, exist_ok=True)
N_PLANS, N_SAMP, N_CTRL = 3, 2, 30
DECOYS = ["river", "candle", "ladder", "orchard", "mirror", "anchor", "blanket", "harvest", "compass", "lantern",
          "pebble", "curtain", "saddle", "whistle", "marble"]
SELF_WRITE = ("Now write the short story (5-8 paragraphs, about 450 words) following your outline exactly, beat by beat. "
              "Do not add events, objects, settings or themes that are not in the outline. Output only the story.")


def run(name, rows, prompts, **kw):
    path = os.path.join(OUT, name + ".jsonl")
    if os.path.exists(path):
        return load_jsonl(path)
    outs = generate(prompts, desc=name, **kw)
    for r, o in zip(rows, outs):
        r["text"] = o
    save_jsonl(rows, path)
    return rows


# ---- plans (no secret)
rows = [dict(pi=pi, k=k) for pi in range(len(PREMISES)) for k in range(N_PLANS)]
plans = run("plans", rows, [OUTLINE_TASK.format(premise=PREMISES[r["pi"]]) for r in rows], max_new_tokens=400, seed=1)
PL = {(r["pi"], r["k"]): r["text"] for r in plans}


def secret_rows(cond):
    return [dict(cond=cond, pi=pi, wi=wi, s=s, k=(wi + s) % N_PLANS) for pi in range(len(PREMISES))
            for wi in range(len(WORDS)) for s in range(N_SAMP)]


def ctrl_rows(cond):
    return [dict(cond=cond, pi=pi, j=j, k=j % N_PLANS) for pi in range(len(PREMISES)) for j in range(N_CTRL)]


def prompt_for(r):
    pi, k = r["pi"], r["k"]
    irr_pi = (pi + 5) % len(PREMISES)
    w = WORDS[r["wi"]] if "wi" in r else None
    d = DECOYS[(r["wi"] + r["pi"]) % len(DECOYS)] if "wi" in r else None
    if d is not None and r["cond"] == "decoy":
        r["decoy"] = d
    return build_prompt(r["cond"], PREMISES[pi], w=w, d=d, plan=PL[(pi, k)], irr=PL[(irr_pi, k)])


seed = 10
for cond, maker in [("none", ctrl_rows), ("secret", secret_rows), ("none+plan", ctrl_rows), ("secret+plan", secret_rows),
                    ("none+irr", ctrl_rows), ("secret+irr", secret_rows), ("decoy", secret_rows), ("hide", secret_rows)]:
    rows = maker(cond)
    run(cond, rows, [prompt_for(r) for r in rows], max_new_tokens=1000, batch_size=36, seed=seed)
    seed += 1

# ---- self-plan: the model writes its own outline (with / without the secret in context), then the story
for cond, maker in [("none+selfplan", ctrl_rows), ("secret+selfplan", secret_rows)]:
    rows = maker(cond)
    base = cond.split("+")[0]
    first = [SYS[base].format(w=WORDS[r["wi"]] if "wi" in r else None) + "\n\n" + OUTLINE_TASK.format(premise=PREMISES[r["pi"]])
             for r in rows]
    prows = run(cond + "_plans", [dict(r) for r in rows], first, max_new_tokens=400, seed=seed)
    msgs = [[{"role": "user", "content": f}, {"role": "assistant", "content": p["text"]}, {"role": "user", "content": SELF_WRITE}]
            for f, p in zip(first, prows)]
    for r, p in zip(rows, prows):
        r["plan"] = p["text"]
    run(cond, rows, msgs, max_new_tokens=1000, batch_size=36, seed=seed + 1)
    seed += 2
print("done")
