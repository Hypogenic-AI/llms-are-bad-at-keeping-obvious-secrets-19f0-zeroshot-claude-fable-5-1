import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np
rows = load_jsonl(os.path.join(RESULTS, "pilot_stories.jsonl"))
sec = [r for r in rows if r["cond"] == "secret"]
prem = PREMISES[:4]
msgs, meta = [], []
for p in prem:
    ctr = [r for r in rows if r["cond"] == "none" and r["premise"] == p]
    for i, r in enumerate([r for r in sec if r["premise"] == p]):
        for first in (True, False):
            msgs.append(afc_messages(r["text"], ctr[i]["text"], r["w"], first)); meta.append((first, r["w"]))
for g in sys.argv[1:]:
    lp = np.array(local_choice(msgs, ["1", "2"], GUESSERS[g], desc=g))
    correct = np.array([(l[0] > l[1]) == f for l, (f, w) in zip(lp, meta)])
    print(g, "acc", correct.mean(), "P(choose 1)", (lp[:, 0] > lp[:, 1]).mean())
    lo = np.array([(l[0] - l[1]) * (1 if f else -1) for l, (f, w) in zip(lp, meta)]).reshape(-1, 2).sum(1)
    print(g, "debiased pair acc", (lo > 0).mean())
    for w in WORDS:
        print("  ", w, correct[[m[1] == w for m in meta]].mean())
