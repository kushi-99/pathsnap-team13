# PathSnap — MLP Baseline Evaluation Report (Person 4: Kruthik Reddy)

**Model Family:** Neural Networks — Multi-Layer Perceptron (`PathSnapMLP`, PyTorch)  
**Selected Baseline Configuration:** `mlp_128_64_32` (`28 -> 128 -> 64 -> 32 -> 1`, BatchNorm1d + ReLU + Dropout `0.20`)  
**Scope:** One baseline model inside the team's multi-model comparison experiment (Bayesian hyperparameter optimisation is deferred until the team compares at least 5 models across distinct algorithm families).  
**Data Integrity Statement:** All training, scaling, and architecture selection used **strictly the 80 `train` trips** under 5-fold trip-grouped cross-validation (`GroupKFold`). The **10 `student_test` trips** were used solely for held-out development evaluation. The **10 `evaluator_test` trips** were never accessed or used.

---

## 1. Problem Formulation & Input Representation

Raw workspace inspection revealed three structural properties of the PathSnap dataset that govern model formulation:

1. **No 1-to-1 `.track` ↔ `.route` row alignment:** `train` contains `208,417` GPS observations vs `31,336` reference route arcs (`student_test`: `19,160` vs `3,477`). Each `.route` file is a chronological sequence of unique traversed `arc_id`s (`99.96%` topologically continuous in `train`, `99.97%` in `student_test`).
2. **Per-trip local `arc_id` namespace:** Every trip has its own subgraph (`649`–`307,392` arcs) indexed `0 .. A-1`, so global multiclass classification over `arc_id` is invalid.
3. **Binary Candidate Scoring Formulation:** For each GPS observation $o_t$, spatial candidate generation (`ml/mlp/data_adapter.py`) builds a sub-segment sampled `cKDTree` in per-trip local tangent-plane meters (`search_radius = 75 m`, `fallback = 150 m`, `max_candidates = 12`). Ground-truth alignment for training/student-test labels uses exact $O(T \cdot R)$ monotonic dynamic programming along `.route` (never assuming row alignment). The MLP predicts $P(y_{t,c} = 1 \mid \mathbf{x}_{t,c})$ for each `(obs_idx, candidate arc_id)` pair:
   - **Observation-level Top-1 prediction:** $\hat{c}_t = \arg\max_{c \in \mathcal{C}(o_t)} P(y_{t,c} = 1 \mid \mathbf{x}_{t,c})$.
   - **HMM/Viterbi emission input:** Both raw sigmoid probabilities (`match_prob`) and per-observation normalized emission distributions (`emission_prob`, `log_emission_prob`) are preserved per `(obs_idx, arc_id)`.

### Candidate Feature Vector (28 Observable Features — Zero Leakage)

All 28 features (`ml/mlp/artifacts/feature_columns.json`) are computed strictly from `.track`, `.nodes`, and `.arcs`:
- **Spatial & projection geometry (8):** `perp_dist_m`, `log_perp_dist`, `proj_ratio`, `is_interior_proj`, `dist_to_start_m`, `dist_to_end_m`, `arc_length_m`, `log_arc_length`
- **Directional / heading alignment (4):** `cos_heading_diff`, `sin_heading_diff_abs`, `heading_diff_norm`, `heading_valid` (essential because `81.9%–99.7%` of road arcs are bidirectional)
- **Local vehicle kinematics (3):** `speed_mps`, `dt_prev_clipped`, `is_time_gap` (guards against GPS gaps up to `153,064 s` in `train`)
- **Local temporal context (5):** `prev_obs_dist_m`, `next_obs_dist_m`, `progress_along_arc`, `is_prev_connected`, `is_next_connected`
- **Road-network topology (3):** `has_reverse_arc`, `from_node_deg`, `to_node_deg`
- **Within-observation competitive context (5):** `dist_rank_norm`, `dist_diff_from_min_m`, `dist_ratio_to_min`, `cos_heading_diff_vs_best`, `num_candidates_obs`

### Dataset Extraction Summary

| Partition | Trips | GPS Observations | Candidate Rows | Positive Rate | Candidate Misses | Candidate Miss Rate | Median Trip Miss Rate | NaN/Inf |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `train` | 80 | 208,417 | 1,779,885 | 10.77% (`191,667`) | 16,750 | **8.04%** | **0.00%** | None |
| `student_test` | 10 | 19,160 | 172,897 | 11.06% (`19,126`) | 34 | **0.18%** | **0.00%** | None |

> **Critical Data-Quality Finding:** Two anomalous training trips — **`00000054`** (`19,313` obs, median GPS-to-route offset `138.5 m`, `58.9%` misses) and **`00000033`** (`5,844` obs, median offset `122.4 m`, `66.5%` misses) — account for `15,264 / 16,750` (**91.1%**) of all training candidate misses. Trip `00000002` also has drift spikes up to `569.5 m` (`8.9%` misses). All trips were retained unfiltered during CV evaluation to report honest, uninflated generalization metrics.

---

## 2. Why MLP Is Suitable for PathSnap (Deliverables Section 3.2A)

1. **Nonlinear cross-feature interactions:** Disambiguating parallel streets, highway on-ramps, and two-way roads requires combining spatial proximity (`perp_dist_m`, `dist_diff_from_min_m`), directional alignment (`cos_heading_diff`, `progress_along_arc`), and kinematics (`speed_mps`, `heading_valid`) nonlinearly — e.g., ignoring heading mismatch when the vehicle is stationary (`heading_valid == 0`) while heavily penalizing opposite-direction candidates at cruising speed.
2. **Smooth, calibrated probabilistic emissions for HMM/Viterbi:** Trained with Binary Cross-Entropy (`BCEWithLogitsLoss`), the MLP outputs continuous sigmoid probabilities (`student_test` Log Loss = **0.0779**), avoiding the hard step-function probability plateaus of shallow trees and feeding clean log-emission potentials directly into Viterbi decoding.
3. **Sub-millisecond vectorized inference:** With `14,529` parameters, CPU inference runs at **0.34–0.85 ms per 1,000 candidates** (`~13–15 ms` for a 2,500-observation trip), easily meeting real-time Laptop 1 FastAPI latency needs.

---

## 3. Architecture, Hyperparameters & Training Procedure (Deliverables Section 3.2B)

### Configurable Architecture (`ml/mlp/model.py`)

Each hidden block applies `Linear -> BatchNorm1d -> ReLU -> Dropout(0.20)`, followed by a linear 1-logit head and `BCEWithLogitsLoss` (binary cross-entropy). Weights use Kaiming-Normal initialization; all RNG seeds (`Python`, `NumPy`, `PyTorch`) are fixed (`seed = 42`, fold offset `+100 * fold`).

Two baseline architectures were compared strictly on **80-trip 5-fold GroupKFold CV** (`ml/mlp/results/cv_architecture_comparison.csv`):

| Experiment | Hidden Layers | Dropout | BatchNorm | Parameters | CV Macro F1 (± std) | CV Obs Top-1 (All) | CV Obs Top-1 (Covered) | CV Log Loss | CV Obs Continuity | CV Comp. Continuity | 5-Fold Train Time | Infer (ms/1k) | Mean Best Ep |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mlp_128_64` | `128-64` | 0.20 | Yes | 12,417 | 0.8984 (±0.0265) | 0.7548 | 0.8208 | **0.1088** | 0.9299 | **0.7775** | 578.38 s | 0.332 | 5 |
| **`mlp_128_64_32`** | **`128-64-32`** | **0.20** | **Yes** | **14,529** | **0.8981 (±0.0269)** | **0.7556** | **0.8216** | 0.1092 | **0.9300** | 0.7707 | 662.29 s | 0.340 | **6** |

- **Optimiser:** Adam (`lr = 1e-3`, `weight_decay = 1e-4`), mini-batch size `2048`, max `25` epochs per fold, early stopping patience `5` on validation BCE loss.
- **Loss / Objective:** `torch.nn.BCEWithLogitsLoss()` (unweighted to preserve calibrated posterior probabilities for Log Loss and HMM emission scoring).
- **Leakage Prevention:** `GroupKFold(n_splits=5)` grouped by `trip_id` with explicit runtime assertion `train_trips.isdisjoint(val_trips)`. `StandardScaler` is fitted strictly on training-fold features only.
- **Selection & Final Refit:** `mlp_128_64_32` edged out `mlp_128_64` on CV Observation-level Top-1 Accuracy (`0.7556` vs `0.7548`, covered `0.8216` vs `0.8208`) with virtually identical Macro-F1 and Log Loss, and was refitted on all 80 training trips for `6` epochs (the CV mean best epoch; final training time `95.20 s`, final training loss `0.0839`, final training Macro-F1 `0.9126`).

---

## 4. Actual Experimental Results (Deliverables Section 3.2C)

All numbers below come directly from executed scripts ([performance_summary_table.csv](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/results/performance_summary_table.csv), [cv_selected_summary.json](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/results/cv_selected_summary.json), [student_test_metrics.json](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/results/student_test_metrics.json)):

### 4.1 Consolidated Performance Table (CV vs Held-Out Student-Test)

| Metric | 5-Fold GroupKFold CV (80 Train Trips, OOF) | Student-Test (10 Held-Out Trips) |
|---|---:|---:|
| Candidate Accuracy | 0.9623 | **0.9688** |
| Macro Precision | 0.9144 | **0.9304** |
| Macro Recall | 0.8835 | **0.9076** |
| **Macro F1-Score** | **0.8981** (±0.0269 across folds) | **0.9186** |
| Positive-Class Precision | 0.8546 | **0.8819** |
| Positive-Class Recall | 0.7831 | **0.8290** |
| Positive-Class F1-Score | 0.8173 | **0.8546** |
| **Observation-Level Top-1 Accuracy (All Obs)** | **0.7556** (`157,472 / 208,417`) | **0.8641** (`16,556 / 19,160`) |
| **Observation-Level Top-1 Accuracy (Covered Obs)** | **0.8216** (`157,472 / 191,667`) | **0.8656** (`16,556 / 19,126`) |
| **Log Loss (Binary Cross-Entropy)** | **0.1092** | **0.0779** |
| Candidate Miss Rate | 8.04% (`16,750 / 208,417`) | **0.18%** (`34 / 19,160`) |
| Route Continuity (Obs-Step, Classifier-Only) | 0.9300 | **0.9417** |
| Route Continuity (Compressed, Classifier-Only) | 0.7707 | **0.7926** |
| Training Time | `662.29 s` (5-fold CV) / `95.20 s` (final 80-trip) | N/A |
| Inference Time | `0.340 ms / 1k candidates` | `0.847 ms / 1k candidates` (`14.6 ms / trip`) |

### 4.2 Fold-by-Fold 5-Fold GroupKFold CV Diagnostics (80 Training Trips)

| Fold | Val Trips | Best Epoch | Val BCE Loss | Macro F1 | Obs Top-1 (All) | Obs Top-1 (Covered) | Notable Trips in Fold |
|---:|---:|---:|---:|---:|---:|---:|---|
| Fold 1 | 16 | 1 | 0.1466 | 0.8673 | 0.4814 | 0.6893 | Contains `00000054` (138.5 m systematic offset, 11,375 misses) |
| Fold 2 | 16 | 4 | 0.1226 | 0.8778 | 0.8002 | 0.8036 | Standard urban/mixed trips |
| Fold 3 | 16 | 12 | 0.0740 | 0.9269 | 0.8750 | 0.8798 | Clean road-network trips |
| Fold 4 | 16 | 2 | 0.1066 | 0.8798 | 0.7144 | 0.8132 | Contains `00000033` (122.4 m offset) & `00000002` (569 m drift) |
| Fold 5 | 16 | 9 | 0.0959 | 0.9193 | 0.8647 | 0.8683 | Clean road-network trips |

### 4.3 Per-Trip Student-Test Breakdown (10 Held-Out Trips)

| Trip ID | GPS Obs | Obs Top-1 Accuracy | Obs-Step Continuity | Compressed Route Len | Compressed Continuity | Candidate Misses | Miss Rate | Avg Candidates / Obs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `00000000` | 2,503 | **0.9469** | 0.9732 | 271 | 0.7519 | 3 | 0.12% | 5.53 |
| `00000007` | 1,354 | **0.9594** | 0.9970 | 191 | 0.9789 | 0 | 0.00% | 5.47 |
| `00000009` | 1,049 | 0.8551 | 0.9122 | 427 | 0.7840 | 6 | 0.57% | 11.51 |
| `00000040` | 2,995 | **0.9255** | 0.9729 | 532 | 0.8475 | 0 | 0.00% | 6.01 |
| `00000052` | 262 | **0.9618** | 0.9540 | 86 | 0.8588 | 0 | 0.00% | 10.24 |
| `00000056` | 3,958 | 0.8828 | 0.9080 | 1,458 | 0.7502 | 22 | 0.56% | 10.38 |
| `00000065` | 768 | 0.8789 | 0.9413 | 212 | 0.7867 | 0 | 0.00% | 11.68 |
| `00000076` | 1,486 | 0.8782 | 0.9057 | 478 | 0.7065 | 3 | 0.20% | 11.59 |
| `00000077` | 3,911 | **0.6988** | 0.9409 | 1,463 | 0.8420 | 0 | 0.00% | 10.78 |
| `00000091` | 874 | 0.8684 | 0.9072 | 279 | 0.7086 | 0 | 0.00% | 10.84 |

### 4.4 Sample Actual Prediction vs Reference Comparison (`student_test/00000000`)

From [prediction_vs_reference_sample.csv](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/results/prediction_vs_reference_sample.csv):

| `obs_idx` | Predicted `arc_id` | Reference `arc_id` | Match Prob | Correct |
|---:|---:|---:|---:|:---:|
| 0 | `17630` | `17630` | 0.9706 | Yes |
| 1 | `17630` | `17630` | 0.9924 | Yes |
| 3 | `17633` | `17633` | 0.9784 | Yes |
| 5 | `211` | `211` | 0.9823 | Yes |
| 9 | `17638` | `17638` | 0.9755 | Yes |
| 20 | `17645` | `17645` | 0.9550 | Yes |

---

## 5. Overfitting / Underfitting Diagnosis (Deliverables Section 3.2C Q9)

**Evidence-based conclusion: No meaningful overfitting; well-generalised baseline with domain-driven variance from anomalous offset trips.**

1. **Train vs Validation curves on clean folds (Folds 3 & 5):** In [loss_curves.png](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/plots/loss_curves.png) and [performance_curves.png](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/plots/performance_curves.png), validation Macro-F1 on Fold 3 (`0.9269`) and Fold 5 (`0.9193`) actually **exceeds** training Macro-F1 (`~0.911–0.914`) because training folds include the two noisy offset trips (`00000054`, `00000033`) while Dropout (`0.20`) is active during training.
2. **Held-out generalization beats CV average:** The final 80-trip model achieved **0.9126** training Macro-F1 (epoch 6, loss `0.0839`) and generalized to **0.9186** Macro-F1 (**0.0779** Log Loss, **86.41%** Top-1) on the 10 completely unseen `student_test` trips.
3. **Why Fold 1's validation loss increased after Epoch 1:** Fold 1 holds Trip `00000054` (median GPS offset `138.5 m`, `11,375` candidate misses). As the network becomes more confident on clean road geometry after epoch 1, BCE penalizes confident false positives on `00000054`'s shifted GPS points — a **dataset-shift / label-noise effect**, not capacity memorization.
4. **Why `student_test` Trip `00000077` is lower (`69.88%` Top-1, `0.00%` misses):** High candidate density (`10.78` candidates/obs, `3,911` obs) creates dense parallel/overlapping streets where independent pointwise features cannot resolve corridor ambiguity without global sequence decoding.

---

## 6. Strengths, Limitations & Potential Improvements

### Strengths
- Strong held-out candidate discrimination (**0.9186 Macro-F1**, **0.9688 Candidate Accuracy**, **86.41% Obs Top-1 Accuracy** before HMM/Viterbi).
- Well-calibrated probabilities (**0.0779 Log Loss** on `student_test`), ideal as HMM emission potentials.
- Very fast CPU inference (**~14.6 ms per trip** on `student_test`), lightweight artifact (`<100 KB`).
- Zero leakage, 100% reproducible, modular feature contract swappable with any team-wide feature table.

### Limitations
- **No global route-topology constraint:** Classifier-only compressed route continuity is **79.26%** on `student_test` (vs `99.97%` in `.route`), leaving ~20.7% disconnected segment jumps — especially on dense parallel corridors like Trip `00000077`.
- **Sensitivity to systematic GPS offsets:** When raw GPS drifts >75 m (`00000054`, `00000033`), fixed-radius candidate generation misses the true arc (`candidate_miss_rate = 8.04%` in `train`).
- **Imbalanced positive recall at fixed 0.5 threshold:** Candidate-level positive recall is `0.8290` (`student_test`) vs precision `0.8819` (many ambiguous intersections split probability ~0.40/0.45 across two candidates, which Top-1 ranking handles better than a 0.50 cutoff).

### Potential Improvements (Next Steps with Team)
1. **HMM / Viterbi Sequential Decoding (Highest Impact):** Combine MLP `log_emission_prob` with road-network shortest-path transition probabilities to eliminate disconnected jumps and resolve parallel-street ambiguity on trips like `00000077`.
2. **Listwise / Group-Softmax Loss per Observation:** Training with cross-entropy normalized across the $K$ candidates of observation $o_t$ (instead of independent binary BCE) directly optimizes Observation-level Top-1 ranking and eliminates multi-candidate probability ambiguity.
3. **Training Outlier Down-weighting:** Down-weighting observations with `candidate_miss == 1` during training (never during validation/test!) prevents shifted trips like `00000054` from injecting false-negative gradients.
4. **Bayesian Hyperparameter Optimisation (Deferred):** Only if the team selects MLP after comparing all 5+ model families (e.g., tuning hidden widths, dropout, weight decay, learning rate, and candidate listwise loss via Optuna on 5-fold GroupKFold CV).

---

## 7. Generated Visualisations & Verifiable Artifacts

All plots and interactive Folium maps were generated directly from actual experimental outputs via [evaluate_mlp.py](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/evaluate_mlp.py):

- **Training & Validation Loss Curves:** [loss_curves.png](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/plots/loss_curves.png)
- **Training & Validation Performance Curves (Macro-F1 & Top-1):** [performance_curves.png](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/plots/performance_curves.png)
- **Candidate-Level Confusion Matrices (CV vs Student-Test):** [confusion_matrices.png](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/plots/confusion_matrices.png)
- **Student-Test Per-Trip Performance Breakdown:** [student_test_per_trip_comparison.png](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/plots/student_test_per_trip_comparison.png)
- **Baseline Architecture Comparison (`128-64` vs `128-64-32`):** [architecture_comparison.png](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/plots/architecture_comparison.png)
- **Interactive Folium Geographical Maps (Raw GPS + Road Network + Reference Route + MLP Top-1 Route):**
  - Best `student_test` trip (`00000052`, 96.2% Top-1): [folium_student_test_00000052.html](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/plots/folium_student_test_00000052.html)
  - Median `student_test` trip (`00000056`, 88.3% Top-1): [folium_student_test_00000056.html](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/plots/folium_student_test_00000056.html)
  - Hardest `student_test` trip (`00000077`, 69.9% Top-1): [folium_student_test_00000077.html](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/plots/folium_student_test_00000077.html)
- **Saved Model Artifacts:** [ml/mlp/artifacts/](file:///c:/Users/Admin/Downloads/ML%20HACKATHON/ml/mlp/artifacts/model_config.json) (`mlp_model.pt`, `scaler.joblib`, `feature_columns.json`, `label_mapping.json`, `model_config.json`, `training_metadata.json`)
