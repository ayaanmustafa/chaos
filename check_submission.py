"""
check_submission.py  --  MSE verifier for OPERATION REBUILD FROM CHAOS

Loads a submission CSV (block_index, inp_piece, out_piece), reconstructs the
residual network from the scrambled .pth fragments, runs the forward pass over
the calibration telemetry and reports the Logits MSE against the original model.

Usage:
    python check_submission.py [submission.csv]      (default: submission_best.csv)
"""
import os, sys, csv, torch, pandas as pd

torch.set_grad_enabled(False)

PIECES_DIR = "data/pieces"
DATA = "data/history_data.csv"
INPUT_DIM, LATENT_DIM, HIDDEN_DIM, NUM_CLASSES = 784, 16, 32, 10


def load_components():
    df = pd.read_csv(DATA)
    px = [c for c in df.columns if c.startswith("pixel_")]
    lg = [c for c in df.columns if c.startswith("pred_logit_")]
    X = torch.tensor(df[px].values, dtype=torch.float32)
    T = torch.tensor(df[lg].values, dtype=torch.float32)
    proj = last = None
    inp, out = {}, {}
    for f in sorted(os.listdir(PIECES_DIR)):
        if not f.endswith(".pth"):
            continue
        p = torch.load(os.path.join(PIECES_DIR, f), map_location="cpu")
        w, b = p["weight"], p["bias"]
        if w.shape == (LATENT_DIM, INPUT_DIM):
            proj = {"f": f, "w": w, "b": b}
        elif w.shape == (NUM_CLASSES, LATENT_DIM):
            last = {"f": f, "w": w, "b": b}
        elif w.shape == (HIDDEN_DIM, LATENT_DIM):
            inp[f] = {"w": w, "b": b}
        elif w.shape == (LATENT_DIM, HIDDEN_DIM):
            out[f] = {"w": w, "b": b}
    return X, T, proj, last, inp, out


def forward_mse(order_inp, order_out, X, T, proj, last, inp, out):
    f = torch.nn.functional.linear(X, proj["w"], proj["b"])          # proj 784->16
    for fi, fo in zip(order_inp, order_out):
        h = torch.relu(torch.nn.functional.linear(f, inp[fi]["w"], inp[fi]["b"]))
        f = f + torch.nn.functional.linear(h, out[fo]["w"], out[fo]["b"])
    logits = torch.nn.functional.linear(f, last["w"], last["b"])     # 16->10
    return torch.nn.functional.mse_loss(logits, T).item()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "submission_best.csv"
    if not os.path.exists(path):
        print(f"[!] Submission file not found: {path}")
        sys.exit(1)
    X, T, proj, last, inp, out = load_components()
    sub = pd.read_csv(path).sort_values("block_index")
    oi = sub["inp_piece"].tolist()
    oo = sub["out_piece"].tolist()

    # integrity checks
    problems = []
    if sorted(oi) != sorted(inp):
        problems.append("inp_piece set does not match the 32 W_in fragments")
    if sorted(oo) != sorted(out):
        problems.append("out_piece set does not match the 32 W_out fragments")
    if len(oi) != len(set(oi)):
        problems.append("duplicate inp_piece entries")
    if len(oo) != len(set(oo)):
        problems.append("duplicate out_piece entries")

    mse = forward_mse(oi, oo, X, T, proj, last, inp, out)
    print("=" * 64)
    print(f" Submission : {path}")
    print(f" Blocks     : {len(oi)}")
    print(f" Logits MSE : {mse:.12e}")
    is_zero = mse < 1e-9
    print(f" MSE == 0 ? : {'YES  (perfect reconstruction)' if is_zero else 'NO'}")
    if problems:
        print(" Integrity  : " + "; ".join(problems))
    else:
        print(" Integrity  : OK (uses each fragment exactly once)")
    print("=" * 64)
    sys.exit(0 if is_zero and not problems else 2)


if __name__ == "__main__":
    main()
