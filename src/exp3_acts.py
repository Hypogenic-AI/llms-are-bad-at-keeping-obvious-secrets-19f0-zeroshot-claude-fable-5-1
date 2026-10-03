"""Experiment 3a: residual-stream activations while the model (teacher-forced) writes its own stories.
Pass A: the actual prompt (with the secret) + story. Pass B: the matched no-secret prompt + the *same* story text.
A - B isolates what holding the secret adds to the residual stream beyond what the story text itself carries."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np, torch
torch.set_grad_enabled(False)

D = os.path.join(RESULTS, "exp1")
ACT = os.path.join(ROOT, "data", "acts")
os.makedirs(ACT, exist_ok=True)
LSUB = [8, 16, 24, 32, 40, 48]
model, tok = load_model()
tok.padding_side = "right"
WTOK = [tok.encode(" " + w, add_special_tokens=False)[0] for w in WORDS]
print("first tokens:", [tok.decode([t]) for t in WTOK])
plans = {(r["pi"], r["k"]): r["text"] for r in load_jsonl(os.path.join(D, "plans.jsonl"))}
final_norm = model.model.language_model.norm if hasattr(model.model, "language_model") else model.model.norm
WU = model.get_output_embeddings().weight[WTOK].detach().float()  # [15, d]


def prompt(r, cond):
    pi, k = r["pi"], r["k"]
    return build_prompt(cond, PREMISES[pi], w=WORDS[r["wi"]] if "wi" in r else None, d=r.get("decoy"),
                        plan=plans[(pi, k)], irr=plans[((pi + 5) % len(PREMISES), k)])


def collect(rows, conds, name, bs=12):
    if os.path.exists(os.path.join(ACT, name + "_mean.npy")):
        return
    n = len(rows)
    mean = np.zeros((n, 49, 3840), np.float16); quart = np.zeros((n, len(LSUB), 4, 3840), np.float16)
    lens = np.zeros((n, len(LSUB) + 1, 15), np.float32)
    order = sorted(range(n), key=lambda i: len(rows[i]["text"]))
    for b in range(0, n, bs):
        idx = order[b:b + bs]
        pre = [chat_text(tok, [{"role": "user", "content": prompt(rows[i], conds[i])}]) for i in idx]
        plen = [len(tok(p, add_special_tokens=False).input_ids) for p in pre]
        enc = tok([p + rows[i]["text"] for p, i in zip(pre, idx)], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        with torch.no_grad():
            o = model(**enc, output_hidden_states=True, logits_to_keep=1)
        T = enc.input_ids.shape[1]
        pos = torch.arange(T, device="cuda:0")[None]
        tlen = enc.attention_mask.sum(1)
        pl = torch.tensor(plen, device="cuda:0")
        # positions whose residual predicts a story token: from last prompt token to the second-to-last story token
        m = ((pos >= (pl[:, None] - 1)) & (pos < (tlen[:, None] - 1))).float()
        frac = (pos - (pl[:, None] - 1)).float() / (tlen - pl)[:, None].float()
        for l in range(49):
            h = o.hidden_states[l].float()
            mean[idx, l] = ((h * m[..., None]).sum(1) / m.sum(1, keepdim=True)).cpu().numpy()
            if l in LSUB:
                li = LSUB.index(l)
                for q in range(4):
                    mq = m * ((frac >= q / 4) & (frac < (q + 1) / 4)).float()
                    quart[idx, li, q] = ((h * mq[..., None]).sum(1) / mq.sum(1, keepdim=True).clamp(min=1)).cpu().numpy()
                lg = final_norm(o.hidden_states[l]).float() @ WU.T  # logit lens, 15 secret-word tokens
                lens[idx, li] = ((lg * m[..., None]).sum(1) / m.sum(1, keepdim=True)).cpu().numpy()
        # true output log-probs of the 15 word tokens, averaged over story positions (chunked to save memory)
        hl = final_norm(o.hidden_states[48])
        acc = torch.zeros(len(idx), 15, device="cuda:0")
        for t0 in range(0, T, 64):
            lp = torch.log_softmax(model.get_output_embeddings()(hl[:, t0:t0 + 64]).float(), -1)[..., WTOK]
            acc += (lp * m[:, t0:t0 + 64, None]).sum(1)
        lens[idx, -1] = (acc / m.sum(1, keepdim=True)).cpu().numpy()
        del o
        if (b // bs) % 5 == 0:
            print(name, b, n, flush=True)
    np.save(os.path.join(ACT, name + "_mean.npy"), mean); np.save(os.path.join(ACT, name + "_quart.npy"), quart)
    np.save(os.path.join(ACT, name + "_lens.npy"), lens)


for cond in ["secret", "secret+plan", "secret+irr", "decoy", "hide"]:
    rows = load_jsonl(os.path.join(D, cond + ".jsonl"))
    collect(rows, [cond] * len(rows), cond.replace("+", "_") + "_A")
    neutral = "none" + ("+" + cond.split("+")[1] if "+" in cond else "")
    collect(rows, [neutral] * len(rows), cond.replace("+", "_") + "_B")
# control stories under their own prompt, and under a secret prompt for a (rotating) word that did not shape the text
for cond in ["none", "none+plan"]:
    rows = load_jsonl(os.path.join(D, cond + ".jsonl"))
    collect(rows, [cond] * len(rows), cond.replace("+", "_") + "_B")
    for r in rows:
        r["wi"] = (r["pi"] + r["j"]) % 15
    collect(rows, ["secret" + cond[4:]] * len(rows), cond.replace("+", "_") + "_A")
print("done")
