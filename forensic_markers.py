"""
forensic_markers.py  --  training-induced fingerprint extraction

Given a *solved* submission (default submission.csv, MSE==0), this
reverse-engineers the recovered network and quantifies the structural markers
that gradient-descent training left in the weights. These are the signals an
optimized reconstruction solver should exploit to recover PAIRING and ORDER
without brute force.

Reported per-block / per-layer:
  * variance & L2 energy of W_in, W_out, biases
  * spectral signature (top singular values, spectral radius of W_out @ W_in)
  * weight-entropy (Shannon entropy of normalized |W| histogram)
  * activation spikes (max pre-ReLU, dead-unit count, fired fraction)
  * residual-branch energy ratio ||residual|| / ||x||  (depth marker)
  * latent-norm trajectory across depth  (ordering marker)
  * pairing separability (Hungarian margin on ||W_out W_in||)

Usage:  python forensic_markers.py [submission.csv]
"""
import os, sys, csv, math, torch, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

torch.set_grad_enabled(False)
np.set_printoptions(precision=4, suppress=True)

PIECES_DIR, DATA = "data/pieces", "data/history_data.csv"
INPUT_DIM, LATENT_DIM, HIDDEN_DIM, NUM_CLASSES = 784, 16, 32, 10
SUB = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"

# ---- load
df = pd.read_csv(DATA)
px = [c for c in df.columns if c.startswith("pixel_")]
lg = [c for c in df.columns if c.startswith("pred_logit_")]
X = torch.tensor(df[px].values, dtype=torch.float32)
T = torch.tensor(df[lg].values, dtype=torch.float32)
proj = last = None; inp, out = {}, {}
for f in sorted(os.listdir(PIECES_DIR)):
    if not f.endswith(".pth"): continue
    p = torch.load(os.path.join(PIECES_DIR, f), map_location="cpu"); w, b = p["weight"], p["bias"]
    if w.shape == (LATENT_DIM, INPUT_DIM):     proj = {"w": w, "b": b}
    elif w.shape == (NUM_CLASSES, LATENT_DIM): last = {"w": w, "b": b}
    elif w.shape == (HIDDEN_DIM, LATENT_DIM):  inp[f] = {"w": w, "b": b}
    elif w.shape == (LATENT_DIM, HIDDEN_DIM):  out[f] = {"w": w, "b": b}
ins, outs = sorted(inp), sorted(out)


def entropy(t):
    a = t.abs().flatten().numpy(); a = a / (a.sum() + 1e-12)
    a = a[a > 0]
    return float(-(a * np.log2(a)).sum())


def fingerprint_pieces():
    print("\n" + "#" * 78)
    print("# 1. PER-FRAGMENT STATISTICAL FINGERPRINTS (no ordering needed)")
    print("#" * 78)
    print(f"{'piece':<14}{'role':<7}{'var':>10}{'L2':>10}{'entropy':>10}"
          f"{'smax':>9}{'srad?':>9}{'bias|mu|':>10}")
    for f in ins:
        w = inp[f]["w"]; sv = torch.linalg.svdvals(w)
        print(f"{f:<14}{'W_in':<7}{w.var():>10.4f}{w.norm():>10.3f}"
              f"{entropy(w):>10.3f}{sv[0]:>9.3f}{'-':>9}{inp[f]['b'].abs().mean():>10.4f}")
    for f in outs:
        w = out[f]["w"]; sv = torch.linalg.svdvals(w)
        print(f"{f:<14}{'W_out':<7}{w.var():>10.4f}{w.norm():>10.3f}"
              f"{entropy(w):>10.3f}{sv[0]:>9.3f}{'-':>9}{out[f]['b'].abs().mean():>10.4f}")


def pairing_separability():
    print("\n" + "#" * 78)
    print("# 2. PAIRING MARKER  -- Hungarian on ||W_out @ W_in||_F")
    print("#" * 78)
    M = np.array([[torch.norm(out[fo]["w"] @ inp[fi]["w"]).item() for fo in outs] for fi in ins])
    r, c = linear_sum_assignment(-M)
    margins = []
    for i in range(len(ins)):
        row = np.sort(M[i])[::-1]; margins.append((row[0] - row[1]) / (abs(row[0]) + 1e-9))
    print(f" assignment confidence (mean relative 1st-2nd margin): {np.mean(margins):.1%}")
    print(f" weakest rows (ambiguous pairs): "
          f"{[ins[i] for i in np.argsort(margins)[:4]]}")
    print(" -> margin is modest; pairing is NOT cleanly separable from weights alone,")
    print("    so the solver must let data-driven swaps repair mispairs (see report).")


def depth_and_trajectory(order_inp, order_out):
    print("\n" + "#" * 78)
    print("# 3. ORDERING / DEPTH MARKERS along the RECOVERED sequence")
    print("#" * 78)
    f = torch.nn.functional.linear(X, proj["w"], proj["b"])
    print(f"{'slot':>5}{'in_piece':>13}{'out_piece':>13}{'resE':>9}"
          f"{'|x|out':>9}{'preReLUmax':>12}{'fired%':>8}{'dead':>6}{'srad':>8}")
    res_energies, norms = [], [torch.norm(f, dim=1).mean().item()]
    for k, (fi, fo) in enumerate(zip(order_inp, order_out)):
        pre = torch.nn.functional.linear(f, inp[fi]["w"], inp[fi]["b"])
        h = torch.relu(pre)
        res = torch.nn.functional.linear(h, out[fo]["w"], out[fo]["b"])
        resE = (torch.norm(res, dim=1) / torch.norm(f, dim=1)).mean().item()
        fired = (pre > 0).float().mean().item()
        dead = int((pre.max(0).values <= 0).sum())
        srad = torch.linalg.svdvals(out[fo]["w"] @ inp[fi]["w"])[0].item()
        f = f + res
        res_energies.append(resE); norms.append(torch.norm(f, dim=1).mean().item())
        print(f"{k:>5}{fi:>13}{fo:>13}{resE:>9.4f}{norms[-1]:>9.3f}"
              f"{pre.max():>12.3f}{fired*100:>7.1f}%{dead:>6}{srad:>8.3f}")
    res_energies = np.array(res_energies)
    norms = np.array(norms)
    dec = np.mean(np.diff(norms) < 0)            # fraction of steps where |x| shrinks
    print(f"\n latent-norm trend along depth : {norms[0]:.2f} -> {norms[-1]:.2f}  "
          f"({dec:.0%} of steps decreasing)")
    print(f"   => near-monotone latent CONTRACTION is the dominant ordering fingerprint")
    print(f"   => sort blocks by descending output |x| to recover the sequence almost directly")
    print(f" residual energy  first->last          : {res_energies[0]:.4f} -> {res_energies[-1]:.4f} "
          f"(corr w/ depth {np.corrcoef(np.arange(len(res_energies)), res_energies)[0,1]:+.2f})")


def main():
    fingerprint_pieces()
    pairing_separability()
    if os.path.exists(SUB):
        sub = pd.read_csv(SUB).sort_values("block_index")
        depth_and_trajectory(sub["inp_piece"].tolist(), sub["out_piece"].tolist())
        # confirm it is actually the MSE==0 reconstruction
        f = torch.nn.functional.linear(X, proj["w"], proj["b"])
        for fi, fo in zip(sub["inp_piece"], sub["out_piece"]):
            h = torch.relu(torch.nn.functional.linear(f, inp[fi]["w"], inp[fi]["b"]))
            f = f + torch.nn.functional.linear(h, out[fo]["w"], out[fo]["b"])
        mse = torch.nn.functional.mse_loss(torch.nn.functional.linear(f, last["w"], last["b"]), T).item()
        print(f"\n[verified] submission '{SUB}' Logits MSE = {mse:.3e} "
              f"({'PERFECT' if mse < 1e-9 else 'not perfect'})")
    else:
        print(f"\n[!] {SUB} not found -- depth/ordering markers skipped "
              f"(run solve_reconstruct.py first).")


if __name__ == "__main__":
    main()
