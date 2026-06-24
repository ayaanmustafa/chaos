# Operation: Rebuild From Chaos — NSOC Model Reconstruction

Reconstructs a trained deep **residual MLP** that was shattered into 66 unlabeled
weight fragments, recovering the exact block **pairing** and **ordering** so the
rebuilt network reproduces the original model's logits with **MSE 0**
(9.17 × 10⁻¹², bit-exact up to float32 round-off) — in **60 forward evaluations**,
fully self-contained and leak-free.

| | |
|---|---|
| Logits MSE vs original | **9.169501 × 10⁻¹²** (exact 0) |
| Pairing accuracy | **100 %**, in **0 forward evaluations** |
| Total forward evaluations | **60** |
| Search space avoided | `(32!)² ≈ 7 × 10⁷⁰` |

## Where everything lives

The solution is in [`nsoc-model-reconstruction/`](nsoc-model-reconstruction/);
the technical report is [`report.pdf`](report.pdf).

```
chaos/
├── report.pdf                     # technical report
└── nsoc-model-reconstruction/     # the solution (run commands from here)
    ├── solution.py        # entry point — reassembles, writes submission.csv + final_model.pth
    ├── solve_best.py       # the solver (pairing + prior + repair)
    ├── model.py            # ReconstructedResNet (loads final_model.pth)
    ├── _lib.py             # shared primitives: load, exact pairing, forward + eval counter
    ├── check_submission.py # independent MSE + integrity verifier
    ├── forensic_markers.py # per-block training-marker report
    ├── submission.csv      # reassembled mapping (deliverable)
    ├── final_model.pth     # reconstructed model weights (deliverable)
    ├── requirements.txt    # pinned deps (torch, numpy, pandas, scipy)
    ├── data/  samples/
    └── README.md  APPROACH.md  EVOLUTION.md  SOLUTION.md  methodology.md  RULES.md
```

## Quick start

```bash
cd nsoc-model-reconstruction
pip install -r requirements.txt

python solution.py                          # -> writes submission.csv + final_model.pth
python check_submission.py submission.csv   # independent MSE + integrity check
python forensic_markers.py submission.csv   # training-fingerprint report
```

## Deliverables (RULES.md §10)

| File | Contents |
|---|---|
| `solution.py` | automated reassembly script |
| `submission.csv` | reassembled mapping (`block_index, inp_piece, out_piece`) |
| `final_model.pth` | `state_dict` of the reconstructed network |
| `report.pdf` | technical report |
| `requirements.txt` | pinned dependencies |

## How it works (one paragraph)

**Pairing is free:** each `W_in` is matched to its `W_out` by a Hungarian
assignment on `‖W_out · W_in‖_F` — 100 % correct from the weights alone, 0 forward
evaluations. **Ordering** is recovered by a strong unsupervised *prior* (a Borda
blend of per-block bias structure and ReLU firing-fraction-at-X0, oriented by the
network's latent-norm contraction) that places every block within ±4 of its true
slot, then a **suspect-gated insertion-sort repair** fixes the ~30 residual local
inversions in 60 forward evaluations — at the structural floor for sorting a
near-sorted permutation. Full details in
[`nsoc-model-reconstruction/APPROACH.md`](nsoc-model-reconstruction/APPROACH.md).

## Verification & integrity

- `python check_submission.py submission.csv` recomputes the logits MSE from a
  fresh forward pass and asserts each fragment is used exactly once and the pairing
  equals the Hungarian assignment.
- `solution.py` reloads `final_model.pth` from disk and confirms it reproduces MSE 0.
- The 60 forward evaluations were independently confirmed by intercepting every MSE
  computation (reported == intercepted).
- The solver reads only `data/pieces/*.pth` and `data/history_data.csv` — no
  hardcoded mapping, no reference answer.
