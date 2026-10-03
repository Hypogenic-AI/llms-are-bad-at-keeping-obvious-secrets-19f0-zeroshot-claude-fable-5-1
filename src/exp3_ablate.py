"""Experiment 3b: remove the secret-word direction from the residual stream during story generation.
Directions are estimated on training premises (pi < 6); stories are generated for held-out premises (pi >= 6).
usage: python exp3_ablate.py [variant ...]"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import numpy as np, torch

D = os.path.join(RESULTS, "exp1"); OUT = os.path.join(RESULTS, "exp3"); ACT = os.path.join(ROOT, "data", "acts")
os.makedirs(OUT, exist_ok=True)
NTRAIN = 6


def sanitize(X):
    """Activations were stored in float16; Gemma's single massive-activation dimension overflows at layers 25-47.
    Non-finite entries are zeroed and that dimension is dropped from every direction (see BAD_DIMS)."""
    X = X.astype(np.float32)
    bad = ~np.isfinite(X)
    X[bad] = 0
    return X


BAD_DIMS = [2339]


def directions(cond="secret"):
    """returns dict of per-word unit directions [15, 49, d] and baseline projections [15, 49]."""
    c = cond.replace("+", "_")
    rows = load_jsonl(os.path.join(D, cond + ".jsonl"))
    A = sanitize(np.load(os.path.join(ACT, c + "_A_mean.npy")))
    B = sanitize(np.load(os.path.join(ACT, c + "_B_mean.npy")))
    tr = np.array([r["pi"] < NTRAIN for r in rows]); wi = np.array([r["wi"] for r in rows])
    out = {}
    for name, X in [("delta", A - B), ("total", A)]:
        M = np.stack([X[tr & (wi == w)].mean(0) for w in range(15)])  # [15, 49, d]
        dirs = np.zeros_like(M); mu = np.zeros((15, 49), np.float32)
        for w in range(15):
            d = M[w] - M[np.arange(15) != w].mean(0)
            d[:, BAD_DIMS] = 0
            d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8
            dirs[w] = d
            # baseline: mean projection of stories written with the *other* secrets (pass A activations)
            oth = A[tr & (wi != w)]
            mu[w] = np.einsum("nld,ld->nl", oth, d).mean(0)
        out[name] = (dirs, mu)
    # random directions with matched baseline
    rng = np.random.default_rng(0)
    R = rng.standard_normal(out["delta"][0].shape).astype(np.float32)
    R[:, :, BAD_DIMS] = 0
    R /= np.linalg.norm(R, axis=-1, keepdims=True)
    muR = np.stack([np.einsum("nld,ld->nl", A[tr], R[w]).mean(0) for w in range(15)])
    out["random"] = (R, muR)
    for k, (d_, m_) in out.items():
        assert np.isfinite(d_).all() and np.isfinite(m_).all(), k
    return out


def make_hooks(dirs, mu, wis, positions):
    """dirs [15,49,d], mu [15,49]; wis: direction index per prompt (in original prompt order)."""
    Dt = torch.tensor(dirs, dtype=torch.bfloat16, device="cuda:0"); Mt = torch.tensor(mu, dtype=torch.bfloat16, device="cuda:0")

    def hooks_fn(model, idx, enc):
        w = torch.tensor([wis[i] for i in idx], device="cuda:0")
        tok = load_model()[1]
        keep = (enc.attention_mask.bool() & (enc.input_ids != tok.bos_token_id))  # [B, T] prompt positions to edit
        handles = []
        for li, layer in enumerate(get_layers(model)):
            d = Dt[w, li + 1]; m = Mt[w, li + 1]  # hidden_states[li+1] is the output of layer li

            def hook(mod, inp, out, d=d, m=m):
                x = out[0] if isinstance(out, tuple) else out
                first = x.shape[1] > 1
                if first and positions == "gen":
                    return out
                coef = torch.einsum("btd,bd->bt", x, d) - m[:, None]
                if first:
                    coef = coef * keep[:, :x.shape[1]].to(coef.dtype)
                x = x - coef[..., None] * d[:, None, :]
                return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
            handles.append(layer.register_forward_hook(hook))
        return handles
    return hooks_fn


if __name__ == "__main__":
    DIRS = directions()
    VARIANTS = {
        "own_gen@delta": ("delta", "own", "gen"), "own_all@delta": ("delta", "own", "all"),
        "other_all@delta": ("delta", "other", "all"), "rand_all": ("random", "own", "all"),
        "own_all@total": ("total", "own", "all"), "own_gen@total": ("total", "own", "gen"),
        "other_all@total": ("total", "other", "all"),
    }
    todo = sys.argv[1:] or list(VARIANTS)
    plans = {(r["pi"], r["k"]): r["text"] for r in load_jsonl(os.path.join(D, "plans.jsonl"))}
    base = [r for r in load_jsonl(os.path.join(D, "secret.jsonl")) if r["pi"] >= NTRAIN]
    for vi, v in enumerate(todo):
        path = os.path.join(OUT, f"secret@abl_{v}.jsonl")
        if os.path.exists(path):
            continue
        kind, which, positions = VARIANTS[v]
        rows = [dict(cond="secret", pi=r["pi"], wi=r["wi"], s=r["s"], k=r["k"], variant=v) for r in base]
        prompts = [build_prompt("secret", PREMISES[r["pi"]], w=WORDS[r["wi"]]) for r in rows]
        wis = [(r["wi"] + 7) % 15 if which == "other" else r["wi"] for r in rows]
        dirs, mu = DIRS[kind]
        outs = generate(prompts, max_new_tokens=1000, batch_size=36, seed=300 + list(VARIANTS).index(v), hooks_fn=make_hooks(dirs, mu, wis, positions), desc=v)
        for r, o in zip(rows, outs):
            r["text"] = o
        save_jsonl(rows, path)
    print("done")
