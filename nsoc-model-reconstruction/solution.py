"""
solution.py -- competition entry point (automated reassembly script).

Runs the full self-contained reconstruction and writes the two deliverables:

    submission.csv     reassembled block mapping (block_index, inp_piece, out_piece)
    final_model.pth    state_dict of the reconstructed network (see model.py)

It reuses the solver in solve_best.py: exact Hungarian pairing (0 forward evals) +
an unsupervised ordering prior (0 evals) + a suspect-gated insertion-sort repair.
Reads ONLY data/pieces/*.pth and data/history_data.csv -- no hardcoded mapping, no
reference answer.

Usage:  python solution.py
"""
import torch
import pandas as pd

from _lib import load_all, pair_blocks, reset_evals, get_evals, forward_mse
from solve_best import build_prior, solve
from model import ReconstructedResNet

SUBMISSION_CSV = "submission.csv"
MODEL_PTH = "final_model.pth"


def main():
    S = load_all()
    pair, _ = pair_blocks(S)              # exact pairing, 0 forward evals
    order0 = build_prior(S, pair)         # unsupervised prior, 0 forward evals
    reset_evals()                         # count only the repair forward passes
    order, mse = solve(S, pair, order0)
    evals = get_evals()
    print(f"reconstruction: logits MSE = {mse:.3e}   forward_evals = {evals}")

    # ---- deliverable 1: submission.csv -------------------------------------------
    rows = [{"block_index": k, "inp_piece": fi, "out_piece": pair[fi]}
            for k, fi in enumerate(order)]
    pd.DataFrame(rows).to_csv(SUBMISSION_CSV, index=False)

    # ---- deliverable 2: final_model.pth ------------------------------------------
    net = ReconstructedResNet.from_reconstruction(S, order, pair).eval()
    torch.save(net.state_dict(), MODEL_PTH)

    # ---- self-check: the saved model reproduces MSE 0 (load fresh) ---------------
    fresh = ReconstructedResNet()
    fresh.load_state_dict(torch.load(MODEL_PTH, map_location="cpu"))
    fresh.eval()
    with torch.no_grad():
        model_mse = torch.nn.functional.mse_loss(fresh(S["X"]), S["T"]).item()
    sub_mse = forward_mse(S, order, [pair[f] for f in order], count=False)
    assert sub_mse < 1e-9 and model_mse < 1e-9, (sub_mse, model_mse)

    print(f"wrote {SUBMISSION_CSV} and {MODEL_PTH}")
    print(f"verified: submission MSE = {sub_mse:.3e}, reloaded-model MSE = {model_mse:.3e} (both exact 0)")
    return mse, evals


if __name__ == "__main__":
    main()
