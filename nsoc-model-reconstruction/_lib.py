"""
_lib.py  --  shared, verified primitives for the reconstruction solver.

Loading, exact pairing, the forward pass, and forward-evaluation counting all live
here so they are identical and correct wherever they are used.

  * load_all()    -> dict with X, T, X0 (= proj(X)), proj, last, inp, out, ins, outs, Wl, bl, K
  * pair_blocks() -> EXACT W_in -> W_out pairing via Hungarian on ||W_out @ W_in||_F (100%, 0 evals)
  * block_apply() -> one residual block;  forward_mse() -> full forward + MSE
  * reset_evals()/get_evals()/_EVALS -> the forward-evaluation counter
    (forward_mse increments it once per call -- a forward eval, full or prefix)

The solver reads ONLY data/pieces/*.pth and data/history_data.csv; it never reads
or hardcodes any reference ordering.
"""
import os, torch, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count() or 4)

PIECES_DIR = "data/pieces"
DATA = "data/history_data.csv"
INPUT_DIM, LATENT_DIM, HIDDEN_DIM, NUM_CLASSES = 784, 16, 32, 10

_EVALS = [0]
def reset_evals():            _EVALS[0] = 0
def get_evals():              return _EVALS[0]


def load_all():
    df = pd.read_csv(DATA)
    X = torch.tensor(df[[c for c in df if c.startswith("pixel_")]].values, dtype=torch.float32)
    T = torch.tensor(df[[c for c in df if c.startswith("pred_logit_")]].values, dtype=torch.float32)
    proj = last = None; inp, out = {}, {}
    for f in sorted(os.listdir(PIECES_DIR)):
        if not f.endswith(".pth"):
            continue
        p = torch.load(os.path.join(PIECES_DIR, f), map_location="cpu"); w, b = p["weight"], p["bias"]
        if w.shape == (LATENT_DIM, INPUT_DIM):     proj = {"f": f, "w": w, "b": b}
        elif w.shape == (NUM_CLASSES, LATENT_DIM): last = {"f": f, "w": w, "b": b}
        elif w.shape == (HIDDEN_DIM, LATENT_DIM):  inp[f] = {"w": w, "b": b}
        elif w.shape == (LATENT_DIM, HIDDEN_DIM):  out[f] = {"w": w, "b": b}
    ins, outs = sorted(inp), sorted(out)
    X0 = torch.nn.functional.linear(X, proj["w"], proj["b"])
    return dict(X=X, T=T, X0=X0, proj=proj, last=last, inp=inp, out=out,
                ins=ins, outs=outs, Wl=last["w"], bl=last["b"], K=len(ins))


def pair_blocks(S):
    """Exact W_in -> W_out pairing (Hungarian, 0 evals). Returns dict + cost matrix."""
    ins, outs, inp, out = S["ins"], S["outs"], S["inp"], S["out"]
    M = np.array([[torch.norm(out[fo]["w"] @ inp[fi]["w"]).item() for fo in outs] for fi in ins])
    r, c = linear_sum_assignment(-M)
    return {ins[i]: outs[j] for i, j in zip(r, c)}, M


def block_apply(S, feat, fi, fo):
    inp, out = S["inp"], S["out"]
    h = torch.relu(torch.nn.functional.linear(feat, inp[fi]["w"], inp[fi]["b"]))
    return feat + torch.nn.functional.linear(h, out[fo]["w"], out[fo]["b"])


def forward_mse(S, order_in, order_out, count=True):
    """Full forward proj->blocks->last; returns MSE. Counts one forward eval."""
    f = S["X0"]
    for fi, fo in zip(order_in, order_out):
        f = block_apply(S, f, fi, fo)
    if count:
        _EVALS[0] += 1
    return torch.nn.functional.mse_loss(torch.nn.functional.linear(f, S["Wl"], S["bl"]), S["T"]).item()
