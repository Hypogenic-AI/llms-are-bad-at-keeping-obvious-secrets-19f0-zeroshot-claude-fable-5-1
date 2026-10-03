"""Pilot: does Gemma-3-12B leak with premise-based story prompts? Which guesser is sensitive?"""
import random, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import *

rng = random.Random(0)
prem = PREMISES[:4]
rows = []
for p in prem:
    for w in WORDS:
        rows.append(dict(cond="secret", premise=p, w=w, prompt=build_prompt("secret", p, w=w)))
    for k in range(15):
        rows.append(dict(cond="none", premise=p, w=None, prompt=build_prompt("none", p)))
outs = generate([r["prompt"] for r in rows], desc="pilot")
for r, o in zip(rows, outs):
    r["text"] = o
save_jsonl(rows, os.path.join(RESULTS, "pilot_stories.jsonl"))
sec = [r for r in rows if r["cond"] == "secret"]
print("literal mention rate", sum(mentions(r["text"], r["w"]) for r in sec) / len(sec))
print("mean words", sum(len(r["text"].split()) for r in rows) / len(rows))
for g in ["openai/gpt-4.1-mini", "anthropic/claude-haiku-4.5", "deepseek/deepseek-v3.2", "google/gemini-3-flash-preview"]:
    msgs, meta = [], []
    for p in prem:
        ctr = [r for r in rows if r["cond"] == "none" and r["premise"] == p]
        for i, r in enumerate([r for r in sec if r["premise"] == p]):
            for first in (True, False):
                msgs.append(afc_messages(r["text"], ctr[i]["text"], r["w"], first)); meta.append(first)
    res = api_map(g, msgs)
    ok = [(parse_choice(o) == (1 if f else 2)) for o, f in zip(res, meta) if parse_choice(o)]
    p1 = sum(parse_choice(o) == 1 for o in res if parse_choice(o)) / max(1, len(ok))
    print(g, "acc", sum(ok) / max(1, len(ok)), "n", len(ok), "P(choose 1)", p1)
