"""Recomputes the output-distribution statistics of exp3_acts.py correctly: HF's last hidden state is already
final-normed, so exp3_acts.py normalised it twice. Stores, per story, the mean over story positions of the raw
output logit and the output log-probability of the 15 secret-word tokens (pass A: secret prompt; pass B: neutral)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np, torch
torch.set_grad_enabled(False)

D = os.path.join(RESULTS, "exp1"); ACT = os.path.join(ROOT, "data", "acts")
model, tok = load_model()
tok.padding_side = "right"
WTOK = [tok.encode(" " + w, add_special_tokens=False)[0] for w in WORDS]
plans = {(r["pi"], r["k"]): r["text"] for r in load_jsonl(os.path.join(D, "plans.jsonl"))}
head = model.get_output_embeddings()


def prompt(r, cond):
    pi, k = r["pi"], r["k"]
    return build_prompt(cond, PREMISES[pi], w=WORDS[r["wi"]] if "wi" in r else None, d=r.get("decoy"),
                        plan=plans[(pi, k)], irr=plans[((pi + 5) % len(PREMISES), k)])


def collect(rows, conds, name, bs=12):
    path = os.path.join(ACT, name + "_out.npy")
    if os.path.exists(path):
        return
    n = len(rows); out = np.zeros((n, 2, 15), np.float32)
    order = sorted(range(n), key=lambda i: len(rows[i]["text"]))
    for b in range(0, n, bs):
        idx = order[b:b + bs]
        pre = [chat_text(tok, [{"role": "user", "content": prompt(rows[i], conds[i])}]) for i in idx]
        plen = [len(tok(p, add_special_tokens=False).input_ids) for p in pre]
        enc = tok([p + rows[i]["text"] for p, i in zip(pre, idx)], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        o = model(**enc, output_hidden_states=True, logits_to_keep=1)
        h = o.hidden_states[-1]  # already final-normed
        T = h.shape[1]; pos = torch.arange(T, device="cuda:0")[None]
        tlen = enc.attention_mask.sum(1); pl = torch.tensor(plen, device="cuda:0")
        m = ((pos >= (pl[:, None] - 1)) & (pos < (tlen[:, None] - 1))).float()
        lg_acc = torch.zeros(len(idx), 15, device="cuda:0"); lp_acc = torch.zeros(len(idx), 15, device="cuda:0")
        for t0 in range(0, T, 64):
            lg = head(h[:, t0:t0 + 64]).float()
            lp = torch.log_softmax(lg, -1)
            lg_acc += (lg[..., WTOK] * m[:, t0:t0 + 64, None]).sum(1); lp_acc += (lp[..., WTOK] * m[:, t0:t0 + 64, None]).sum(1)
        out[idx, 0] = (lg_acc / m.sum(1, keepdim=True)).cpu().numpy(); out[idx, 1] = (lp_acc / m.sum(1, keepdim=True)).cpu().numpy()
        del o
    np.save(path, out); print(name, "done", flush=True)


for cond in ["secret", "secret+plan", "secret+irr", "decoy", "hide"]:
    rows = load_jsonl(os.path.join(D, cond + ".jsonl"))
    collect(rows, [cond] * len(rows), cond.replace("+", "_") + "_A")
    neutral = "none" + ("+" + cond.split("+")[1] if "+" in cond else "")
    collect(rows, [neutral] * len(rows), cond.replace("+", "_") + "_B")
for cond in ["none", "none+plan"]:
    rows = load_jsonl(os.path.join(D, cond + ".jsonl"))
    collect(rows, [cond] * len(rows), cond.replace("+", "_") + "_B")
    for r in rows:
        r["wi"] = (r["pi"] + r["j"]) % 15
    collect(rows, ["secret" + cond[4:]] * len(rows), cond.replace("+", "_") + "_A")
print("done")
