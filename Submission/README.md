# QYield — Computing guide

QYield is a local Python pipeline for customer probability estimation, six-customer selection and campaign risk simulation. **[Open report.html](report.html)** for the consolidated explanation, methods, charts and interpretation previously split between the HTML and PDF.

## Quick start

Run these commands from the project folder. The recorded environment uses Python 3.13.13 on macOS arm64; direct dependency versions are pinned in `requirements.txt`.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py --validate-only
python run_pipeline.py
python -m pytest -q
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. A full run overwrites the three result files and generated diagnostics. The HTML is a reviewed snapshot, not an automatically refreshed dashboard; changing model settings requires reviewing the report against the new results.

## Execution flow

```text
data/bank_campaign_participant_release.xlsx
  ├─ Stage1_Train (300 labelled rows) → fold-local validation → final model fit
  ├─ Stage1_Test  (100 unlabelled rows) → stage1_results.csv → top 10
  └─ Stage2_Data (filtered to test IDs) → 10-qubit XY-QAOA → selected six
                                                       ↓
                            6-qubit Bernoulli simulation + exact DP + Monte Carlo
```

All learned preprocessing and classifier fitting use only the supplied 300 training rows. Validation uses five stratified folds. `--validate-only` opens only `Stage1_Train`; it does not open the test, operational or constraint sheets. In a full run, the frozen model makes one test prediction call. IDs, target labels, `duration` and Stage 2 operational fields are excluded from prediction features. Test customers are ranked by descending probability, with ascending customer ID used to break ties.

## Code map

| File | Responsibility |
|---|---|
| `run_pipeline.py` | CLI, stage orchestration, result serialization, figures and execution manifest |
| `src/data_loader.py` | Workbook schema checks, train-only loading and ID-safe operational joins |
| `src/stage1_predict.py` | Numeric transforms, one-hot encoding, fold-local PCA/kernel features, logistic models and validation |
| `src/stage2_optimize.py` | Structured Dicke preparation, feasibility guard, XY-QAOA, sampling and exact enumeration benchmark |
| `src/stage3_simulate.py` | Ry circuit, exact distribution, shot simulation, Monte Carlo, noise and derivatives |
| `src/plotting.py` | Runtime diagnostic figures |
| `tests/` | Data-boundary, kernel, circuit, constraint, probability and input-hash checks |
| `notebooks/campaign_pipeline.ipynb` | Runnable walkthrough of the retained implementation and outputs |

## Frozen implementation

### Stage 1: hybrid probabilistic classifier

The 15 raw predictors comprise six numeric and nine categorical fields. Numeric transforms include signed-log balance, log contact counts, previous-contact status and inverse recency using a 30-day scale. The final full-data fit produces 50 encoded classical columns; category coverage can differ by validation fold.

PCA(4) and a second standardization produce four inputs to `0.30 * arctan(z)`. Qiskit's circular four-qubit `zz_feature_map`, with two repetitions, yields exact simulated statevectors. The model combines six centered fidelity-KernelPCA components and ten Z/ZZ expectations, scales the resulting quantum feature block by 0.35, and appends it to the classical representation. The logistic head uses `C=0.12`. The retained classical logistic reference uses `C=0.15`, so it is not an exactly matched ablation.

Statevectors are cached using angle vectors rounded to ten decimal places. Fidelity matrices and Pauli expectations use NumPy operations. The quantum circuit is fixed; the logistic head is learned classically. `frozen_config.json` records the selected configuration, but the runner does not load it as a runtime override: implementation parameters live in the source modules.

### Stage 2: simulated quantum optimization

The objective is `sum((p_i * deposit_value_i - contact_cost_i) * x_i)`, with exactly six selected, contact cost at most 25 and capacity at most 22. The register has ten qubits, one per shortlisted customer. Structured preparation creates the uniform weight-six Dicke state. Three alternating layers use cost phases and a number-conserving XY mixer over a ring plus five opposite-node chords.

For the supplied shortlist, the largest possible six-person cost and capacity totals are both 19. This certifies that every weight-six state is feasible and permits the cost layer `RZ(+gamma * net_value_i)`, up to global phase. Binding-constraint or noise-mode construction uses the exact diagonal positive-part penalty fallback. Inequalities are not replaced by equalities.

COBYLA uses eight restarts, at most 150 evaluations per restart and seed 42. The best feasible **observed** sample is retained by objective value. Enumeration of all 210 six-customer groups is an independent optimum check, not a substitute quantum answer.

### Stage 3: simulation and exact checks

Each selected customer receives one qubit with `Ry(2 * arcsin(sqrt(p_i)))`. A measured 1 represents a subscription; success means at least three ones. Qiskit bitstrings are little-endian: the rightmost character is q0. `qubit_to_customer_mapping` in `stage3_results.json` provides the authoritative order and angles.

The independent-Bernoulli distribution is checked with dynamic programming, 64-outcome enumeration, ideal statevectors, 100000 noiseless quantum shots and 100000 classical Monte Carlo trials. Three additional 100000-shot experiments isolate depolarizing, readout and combined noise. Total quantum shots across Stages 2 and 3 are 408192; Monte Carlo trials are separate.

## Required outputs and supporting evidence

| Output | Contents |
|---|---|
| `stage1_results.csv` | 100 unique IDs, estimated probabilities, ranks and top-ten flags |
| `stage2_results.json` | Top-ten inputs, selected six, bit ordering, objective, constraints, optimizer parameters, samples and resource audit |
| `stage3_results.json` | Selected IDs/probabilities, angles, exact PMF, raw measurement counts, Monte Carlo, noise and sensitivities |
| `results/stage1_validation_only_report.json` | Training-only validation metrics and environment |
| `results/stage1_detailed_metrics.json` | Generated by a full run: model metrics, kernel audit and provenance |
| `results/execution_manifest.json` | Generated by a full run: versions, seeds, input/output hashes and runtime |

JSON stores results and configuration records; Python performs the computation. The Stage 3 distribution is under `exact_classical_dp.pmf_by_hamming_weight`. Use these values rather than copying the inconsistent table from the superseded PDF.

## Circuit accounting

| Metric | Stage 2 | Stage 3 |
|---|---:|---:|
| Logical qubits | 10 | 6 |
| Physical hardware | None | None |
| Recorded original / transpiled depth | 166 / 741 | 2 / 2 |
| Original 1Q / 2Q / SWAP | 44 / 264 / 0 | 6 / 0 / 0 |
| Transpiled 1Q / 2Q / SWAP | 1150 / 426 / 0 | 6 / 0 / 0 |
| Main estimate shots | 8192 | 100000 |

Gate totals exclude measurements. Stage 2's recorded original depth includes measurements, while its transpiled depth excludes them. Compilation uses `[rz, sx, x, cx]`, optimization level 1 and seed 42, with no device coupling map. Physical-qubit counts are not applicable to this local simulation. The matched reference is 4259 CX gates versus 426 in production; state preparation is included. This comparison measures circuit resources, not hardware performance.

## Reproducibility and interpretation

Qiskit 2.6.2 and Qiskit Aer 0.18.2 provide `Statevector` and `AerSimulator`; scikit-learn 1.9.1 supplies preprocessing and estimators. The main model, split, optimizer, transpiler, simulator and Monte Carlo seeds are 42. Use `requirements.txt` for the complete direct dependency pins and generated manifests for the observed runtime environment. Numerical differences can occur across library versions and platforms.

The selected group is C0031, C0170, C0079, C0265, C0318 and C0256. The sampled quantum probability of at least three subscriptions is 0.14047; the exact model-conditional probability is 0.1412884529. The report distinguishes these estimates, synthetic noise effects and expected value from observed commercial outcomes. Validation results are selection-conditioned because model development used this small training dataset.
