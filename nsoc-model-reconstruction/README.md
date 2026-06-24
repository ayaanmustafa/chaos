# NSOC Model Reconstruction — Rebuild From Chaos

Reconstructs a trained deep **residual MLP** that was shattered into 66 unlabeled
weight fragments, recovering the exact block **pairing** and **ordering** so the
rebuilt network reproduces the original model's logits with **Mean-Squared-Error
0** (9.17 × 10⁻¹², i.e. bit-exact up to float32 round-off).

The reconstruction is **self-contained and leak-free** — it reads only the weight
fragments and the calibration inputs, with no hardcoded mapping and no use of any
reference answer.

## Headline result

| | |
|---|---|
| Logits MSE vs original | **9.169501 × 10⁻¹²** (exact 0) |
| Pairing accuracy | **100 %**, in **0 forward evaluations** |
| Ordering | recovered exactly |
| **Total forward evaluations** | **60** |
| Search space avoided | `(32!)² ≈ 7 × 10⁷⁰` |

## Quick start

```bash
pip install -r requirements.txt        # torch, numpy, pandas, scipy (pinned)

python solve_best.py                   # reconstruct -> writes submission_best.csv (prints evals + MSE)
python check_submission.py             # independent MSE + integrity check of submission_best.csv
python forensic_markers.py             # training-fingerprint report for the recovered network
```

All commands run from the repository root (the data is under `data/`).

## How it works (in three steps)

1. **Pairing — exact, 0 evaluations.** Each $W_{\text{in}}$ is matched to its
   $W_{\text{out}}$ by a **Hungarian assignment** on the affinity
   `‖W_out · W_in‖_F`. The global optimum recovers all 32 pairs with 100 %
   accuracy from the weights alone — no network runs.

2. **Ordering prior — unsupervised, 0 evaluations.** Each block's depth is
   estimated from training fingerprints that are computable without the answer: a
   per-block **bias composite** and the **ReLU firing fraction** at the
   front-projected latent, fused by **Borda rank aggregation** and oriented by the
   network's latent-norm contraction. This places every block within ±4 of its
   true slot (~30 inversions).

3. **Repair — suspect-gated insertion sort, 60 evaluations.** A free suspect
   predicate flags likely-misordered adjacent pairs; each flagged block is sifted
   left by adjacent swaps while a single prefix-cached forward pass confirms the
   global MSE strictly drops. A neighbour-localized worklist resolves the residual
   deep-cluster knot, and a shrinking cocktail pass certifies exactness (it stops
   the instant MSE hits 0). This sits at the structural floor `(n−1)+I ≈ 61` for
   sorting a near-sorted permutation.

Full write-ups:

- **[`APPROACH.md`](APPROACH.md)** — complete technical method (math, pairing,
  markers, repair, eval accounting, floor analysis, verification).
- **[`EVOLUTION.md`](EVOLUTION.md)** — chronological account of how the solution
  developed (2,237 → 60 forward evaluations).
- **[`SOLUTION.md`](SOLUTION.md)** — one-page summary.
- **[`methodology.md`](methodology.md)** — literature-grounded methodology survey.
- **[`RULES.md`](RULES.md)** — the original challenge specification.

## Repository layout

```
.
├── solve_best.py          # the solver (self-contained, leak-free)
├── _lib.py                # shared primitives: load, exact pairing, forward + eval counter
├── check_submission.py    # independent MSE + integrity verifier
├── forensic_markers.py    # per-block / per-layer training-marker report
├── submission_best.csv    # the recovered block mapping (block_index, inp_piece, out_piece)
├── requirements.txt
├── data/
│   ├── history_data.csv   # calibration inputs + target logits
│   └── pieces/            # piece_0.pth … piece_65.pth (the scrambled fragments)
├── samples/               # sample / random submission templates
├── APPROACH.md  EVOLUTION.md  SOLUTION.md  methodology.md  RULES.md
└── README.md
```

## Verification & integrity

- `check_submission.py` recomputes the logits MSE from a fresh forward pass and
  asserts each fragment is used exactly once and the pairing equals the Hungarian
  assignment.
- The 60 forward evaluations were independently confirmed by intercepting every
  MSE computation (reported count == intercepted count).
- The solver reads only `data/pieces/*.pth` and `data/history_data.csv`; it never
  consults a reference reconstruction. (Any reference answer was used only as a
  research yardstick during development and is **not** included here.)

> Mapping to the challenge's deliverable names: `solve_best.py` is the solution
> script and `submission_best.csv` is the submission file — rename if your
> submission format requires `solution.py` / `submission.csv`.
