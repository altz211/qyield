"""End-to-end reproducible runner for the Hybrid Quantum-Classical Term Deposit Campaign Pipeline.

Usage:
    # Training-only 5-fold CV on the 300 Stage1_Train rows (never touches Stage1_Test):
    python run_pipeline.py --validate-only

    # Full frozen pipeline execution (Stages 1, 2, and 3):
    python run_pipeline.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import (
    load_and_verify_stage1_train_only,
    load_and_verify_workbook,
    resolve_workbook_path,
)
from src.plotting import generate_all_figures
from src.stage1_predict import STAGE1_SEED, run_stage1_pipeline, validate_stage1_only
from src.stage2_optimize import (
    STAGE2_SEED,
    build_dicke_10_6_bartschi_eidenbenz_circuit,
    run_stage2_pipeline,
)
from src.stage3_simulate import (
    STAGE3_SEED,
    compute_exact_poisson_binomial_dp,
    compute_exact_stage3_analytical_derivatives,
    run_stage3_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_environment_manifest() -> dict[str, Any]:
    """Capture exact package versions, Python runtime, OS, process telemetry, and fixed seeds."""
    packages = [
        "numpy",
        "scipy",
        "pandas",
        "openpyxl",
        "scikit-learn",
        "qiskit",
        "qiskit-aer",
        "matplotlib",
        "pytest",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"

    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "package_versions": versions,
        "seeds": {
            "workbook_sampling_seed": 42,
            "workbook_train_test_split_seed": 42,
            "stage1_cv_and_model_seed": STAGE1_SEED,
            "stage2_qaoa_optimizer_and_simulator_seed": STAGE2_SEED,
            "stage3_simulator_and_mc_seed": STAGE3_SEED,
        },
    }


def compute_error_propagation_sensitivity(
    selected_df: pd.DataFrame,
    exact_stage3_prob: float,
) -> dict[str, Any]:
    """Quantify how Stage 1 probability uncertainty propagates into Stage 2 net value and Stage 3."""
    probs = selected_df["predicted_probability"].to_numpy(dtype=np.float64)
    dep_vals = selected_df["expected_deposit_value"].to_numpy(dtype=np.float64)
    costs = selected_df["contact_cost"].to_numpy(dtype=np.float64)
    cids = selected_df["customer_id"].astype(str).tolist()

    stage3_deriv_info = compute_exact_stage3_analytical_derivatives(probs, customer_ids=cids)
    jacobian_stage2 = {cids[k]: float(dep_vals[k]) for k in range(len(probs))}

    # Perturbation scenarios: uniform +/- 0.02 and +/- 0.05 probability calibration shifts
    scenarios: dict[str, dict[str, float]] = {}
    for delta in [-0.05, -0.02, 0.0, 0.02, 0.05]:
        p_shifted = np.clip(probs + delta, 0.0, 1.0)
        s2_obj = float(np.sum(p_shifted * dep_vals - costs))
        s3_prob = float(compute_exact_poisson_binomial_dp(p_shifted)["prob_at_least_3"])
        scenarios[f"delta_p_{delta:+.2f}"] = {
            "shifted_mean_prob": float(np.mean(p_shifted)),
            "stage2_expected_net_value": s2_obj,
            "stage3_prob_at_least_3": s3_prob,
        }

    return {
        "stage2_objective_gradient_wrt_p": jacobian_stage2,
        "stage3_analytical_derivative_identity": stage3_deriv_info["identity"],
        "stage3_conversion_gradient_wrt_p": stage3_deriv_info["analytical_gradient_wrt_p"],
        "stage3_central_diff_gradient_wrt_p": stage3_deriv_info["central_diff_gradient_wrt_p"],
        "stage3_min_derivative": stage3_deriv_info["min_derivative"],
        "stage3_max_derivative": stage3_deriv_info["max_derivative"],
        "stage3_sum_of_derivatives_uniform_shift_rate": stage3_deriv_info[
            "sum_of_derivatives_uniform_shift_rate"
        ],
        "stage3_max_abs_diff_analytical_vs_central_diff": stage3_deriv_info[
            "max_abs_diff_vs_central_difference"
        ],
        "calibration_shift_scenarios": scenarios,
        "interpretation": (
            "Stage 2 expected net value is linear in predicted probabilities "
            "(dZ/dp_i = expected_deposit_value_i), whereas by multilinear expansion of independent "
            "Bernoulli variables, Stage 3 target probability satisfies the exact identity "
            "dP(S >= 3)/dp_i = P(S_{-i} = 2)."
        ),
    }


def main(validate_only: bool = False) -> dict[str, Any]:
    """Execute either training-only validation (`--validate-only`) or the full frozen pipeline."""
    t_start = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    workbook_path = resolve_workbook_path()

    env_manifest = get_environment_manifest()

    if validate_only:
        train_only_df = load_and_verify_stage1_train_only(workbook_path)
        val_out = validate_stage1_only(train_df=train_only_df, seed=STAGE1_SEED)
        val_summary = {
            "mode": "validate_only_300_train_rows",
            "sheets_read": ["Stage1_Train"],
            "stage1_test_touched": False,
            "selected_model_name": val_out.selected_model_name,
            "selection_rationale": val_out.selection_rationale,
            "selection_optimism_note": val_out.selection_optimism_note,
            "cv_metrics_by_model": {
                m: {
                    "composite_train_score": val_out.cv_metrics_by_model[m]["composite_train_score"],
                    "qualifies": val_out.cv_metrics_by_model[m]["qualification_check"]["qualifies"],
                    "pooled_oof": val_out.cv_metrics_by_model[m]["pooled_oof"],
                    "per_fold_n60_pos7": val_out.cv_metrics_by_model[m]["per_fold_n60_pos7"],
                }
                for m in val_out.cv_metrics_by_model
            },
            "environment": env_manifest,
            "runtime_sec": float(time.perf_counter() - t_start),
        }
        (RESULTS_DIR / "stage1_validation_only_report.json").write_text(
            json.dumps(val_summary, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(val_summary, indent=2))
        return val_summary

    dataset = load_and_verify_workbook(workbook_path)

    # --- STAGE 1 ---
    stage1_out = run_stage1_pipeline(
        train_df=dataset.stage1_train,
        test_df=dataset.stage1_test,
        seed=STAGE1_SEED,
    )

    stage1_csv_path = PROJECT_ROOT / "stage1_results.csv"
    stage1_out.results_df.to_csv(stage1_csv_path, index=False, float_format="%.16g")

    stage1_reloaded = pd.read_csv(stage1_csv_path)
    max_reload_diff = float(
        np.max(
            np.abs(
                stage1_reloaded["predicted_probability"].to_numpy(dtype=np.float64)
                - stage1_out.results_df["predicted_probability"].to_numpy(dtype=np.float64)
            )
        )
    )
    if max_reload_diff > 1e-15:
        raise RuntimeError(f"Precision drift detected in stage1_results.csv: {max_reload_diff}")

    # --- STAGE 2 ---
    stage2_out = run_stage2_pipeline(
        stage1_results_df=stage1_reloaded,
        stage2_test_data_df=dataset.stage2_data_test_only,
        constraints=dataset.constraints,
        seed=STAGE2_SEED,
    )

    cap_top6 = stage2_out.hamiltonian_verification["feasibility_guard_proof_O_N_log_N"][
        "max_weight_k_capacity_sum"
    ]
    cost_top6 = stage2_out.hamiltonian_verification["feasibility_guard_proof_O_N_log_N"][
        "max_weight_k_cost_sum"
    ]

    stage2_json_payload = {
        "stage": 2,
        "task": "Constrained Customer Selection (Top 10 -> Exact 6)",
        "data_discrepancy_and_constraint_notes": {
            "stage2_data_sheet_discrepancy": dataset.audit_summary["discrepancy_note"],
            "budget_and_capacity_activity_analysis": (
                f"Across all 400 rows in Stage2_Data, contact_cost is in {{1, 2, 3, 4}}, so for any "
                f"exact K=6 selection, total contact_cost <= 6 * 4 = 24 < 25 = C (max top-6 sum on "
                f"this candidate pool is {cost_top6:.0f} <= 25). Capacity units are in {{1, 2, 3, 4}} "
                f"across Stage2_Data so capacity H <= 22 can bind in general (up to 6 * 4 = 24 > 22); "
                f"on our frozen Stage 1 top-10 candidate pool, the 6 largest candidate capacities sum "
                f"to {cap_top6:.0f} <= 22, so our O(N log N) feasibility guard proves all 210 weight-6 "
                f"subsets are feasible and dispatches the reduced 1-qubit RZ cost separator (0 CX per "
                f"layer). Whenever capacity or budget can bind on H_6 or a weight-leaking noise model "
                f"is active, the circuit automatically dispatches the exact 10-qubit positive-part "
                f"penalty DiagonalGate (verified on a binding capacity counterexample in tests)."
            ),
        },
        "constraints": {
            "N_candidates": dataset.constraints.candidate_count_n,
            "K_exact_selection": dataset.constraints.exact_selection_k,
            "budget_limit_C": dataset.constraints.budget_limit_c,
            "capacity_limit_H": dataset.constraints.capacity_limit_h,
        },
        "top_10_candidates": stage2_out.candidates_df.to_dict(orient="records"),
        "quantum_solution": {
            "solver": stage2_out.circuit_report["ansatz_type"],
            "selected_customer_ids": stage2_out.selected_customer_ids,
            "selected_bitstring_little_endian": stage2_out.selected_bitstring_little_endian,
            "qubit_to_customer_convention": (
                "Qubit q_i (i=0..9) corresponds to candidate rank i+1. In the 10-character "
                "little-endian bitstring 'b9 b8 ... b1 b0', the rightmost bit b0 is q_0."
            ),
            "objective_expected_net_value": stage2_out.quantum_objective,
            "total_contact_cost": stage2_out.quantum_total_cost,
            "total_capacity_units": stage2_out.quantum_total_capacity,
            "is_feasible": stage2_out.quantum_is_feasible,
            "selected_customers_detail": stage2_out.selected_df.to_dict(orient="records"),
        },
        "classical_exact_reference": {
            "method": "Exhaustive enumeration of all C(10,6)=210 weight-6 subsets and all 2^10=1024 states",
            "optimal_customer_ids": stage2_out.exact_reference.optimal_customer_ids,
            "optimal_bitstring_little_endian": stage2_out.exact_reference.optimal_bitstring_little_endian,
            "optimal_objective_expected_net_value": stage2_out.exact_reference.optimal_objective,
            "optimal_total_contact_cost": stage2_out.exact_reference.optimal_total_cost,
            "optimal_total_capacity_units": stage2_out.exact_reference.optimal_total_capacity,
            "total_weight6_subsets": stage2_out.exact_reference.total_weight6_subsets,
            "feasible_weight6_subsets": stage2_out.exact_reference.feasible_weight6_subsets,
            "capacity_violating_weight6_subsets": stage2_out.exact_reference.capacity_violating_weight6_subsets,
            "budget_violating_weight6_subsets": stage2_out.exact_reference.budget_violating_weight6_subsets,
            "top_5_feasible_subsets": stage2_out.exact_reference.top5_feasible_subsets,
        },
        "comparison": {
            "matches_exact_optimum": bool(stage2_out.optimality_gap_abs < 1e-12),
            "absolute_optimality_gap": stage2_out.optimality_gap_abs,
            "relative_optimality_gap": stage2_out.optimality_gap_rel,
        },
        "hamiltonian_verification": stage2_out.hamiltonian_verification,
        "mixer_verification": stage2_out.mixer_verification,
        "quantum_sampling_summary": stage2_out.sampling_summary,
        "circuit_resources": stage2_out.circuit_report,
        "runtime_sec": stage2_out.runtime_sec,
        "environment": env_manifest,
    }
    stage2_json_path = PROJECT_ROOT / "stage2_results.json"
    stage2_json_path.write_text(json.dumps(stage2_json_payload, indent=2), encoding="utf-8")

    # --- STAGE 3 ---
    stage3_out = run_stage3_pipeline(
        selected_df=stage2_out.selected_df,
        seed=STAGE3_SEED,
    )
    error_prop = compute_error_propagation_sensitivity(
        selected_df=stage2_out.selected_df,
        exact_stage3_prob=stage3_out.exact_dp_result["prob_at_least_3"],
    )

    stage3_json_payload = {
        "stage": 3,
        "task": "Campaign Outcome Simulation (P(at least 3 of 6 selected customers subscribe))",
        "model_conditional_disclaimer": (
            "All Stage 3 conversion probabilities are conditional on the frozen Stage 1 "
            "predicted probabilities for the 6 selected customers and assume independent "
            "customer responses; they are not observed ground-truth test outcomes."
        ),
        "selected_customer_ids": [r["customer_id"] for r in stage3_out.selected_customers_table],
        "qubit_to_customer_mapping": stage3_out.selected_customers_table,
        "bit_ordering_convention": stage3_out.circuit_report["bit_ordering_convention"],
        "summary_estimates_prob_at_least_3": {
            "exact_poisson_binomial_dp": stage3_out.exact_dp_result["prob_at_least_3"],
            "exact_64_outcome_enumeration": stage3_out.exact_64_result["prob_at_least_3"],
            "ideal_quantum_statevector": stage3_out.ideal_statevector_result["prob_at_least_3"],
            "noiseless_quantum_shots_100k": stage3_out.noiseless_shot_result["estimated_prob_at_least_3"],
            "classical_monte_carlo_100k": stage3_out.classical_mc_result["estimated_prob_at_least_3"],
            "noisy_gate_depolarizing_only_100k": (
                stage3_out.noise_experiments["gate_depolarizing_only"]["estimated_prob_at_least_3"]
            ),
            "noisy_readout_error_only_100k": (
                stage3_out.noise_experiments["readout_error_only"]["estimated_prob_at_least_3"]
            ),
            "noisy_combined_depol_and_readout_100k": (
                stage3_out.noise_experiments["combined_depolarizing_and_readout"]["estimated_prob_at_least_3"]
            ),
        },
        "exact_classical_dp": stage3_out.exact_dp_result,
        "ideal_quantum_statevector": stage3_out.ideal_statevector_result,
        "noiseless_quantum_shot_simulation": stage3_out.noiseless_shot_result,
        "classical_monte_carlo_simulation": stage3_out.classical_mc_result,
        "synthetic_noise_experiments": stage3_out.noise_experiments,
        "extreme_input_verification": stage3_out.extreme_input_checks,
        "analytical_sensitivity_derivatives": stage3_out.analytical_derivatives,
        "error_propagation_and_sensitivity": error_prop,
        "circuit_resources": stage3_out.circuit_report,
        "runtime_sec": stage3_out.runtime_sec,
        "environment": env_manifest,
    }
    stage3_json_path = PROJECT_ROOT / "stage3_results.json"
    stage3_json_path.write_text(json.dumps(stage3_json_payload, indent=2), encoding="utf-8")

    # Save supporting Stage 1 detailed report and circuit diagrams under results/
    stage1_detailed_payload = {
        "stage": 1,
        "hard_data_boundary_verification": dataset.audit_summary,
        "development_provenance_note": (
            "All feature transformations, PCA(4) bounded-angle encoding (angle_scale=0.30 adopted from "
            "C02_Bandwidth_030_Reps2 under the preregistered PLAN.md Section 2.2 two-seed mean + 7/10 "
            "paired fold-win rule), quantum feature map design, regularization parameters, and model "
            "selection were determined exclusively from StratifiedKFold out-of-fold metrics on the "
            "300 Stage1_Train rows (verified via `python run_pipeline.py --validate-only`). Note on "
            "session provenance and selection bias: initial integration runs were executed on the "
            "prior frozen baseline (C00, angle_scale=0.35), and final production inference was run "
            "once after freezing C02 (angle_scale=0.30). No test labels exist in the workbook and no "
            "Stage 1 design choices were tuned on test-set outputs. Because C02 was selected from 25 "
            "configurations evaluated on the 300 training rows, its OOF metrics carry finite-sample "
            "selection bias and do not guarantee out-of-sample generalization gain."
        ),
        "selected_model_name": stage1_out.selected_model_name,
        "selected_config_id": "C02_Bandwidth_030_Reps2",
        "selection_rationale": stage1_out.selection_rationale,
        "selection_optimism_note": stage1_out.selection_optimism_note,
        "quantum_gram_matrix_audit": stage1_out.gram_audit,
        "cv_metrics_by_model_5fold_oof": stage1_out.cv_metrics_by_model,
        "circuit_resources": stage1_out.circuit_report,
        "runtimes_sec": stage1_out.runtimes_sec,
        "top_10_test_candidates_preview": (
            stage1_out.results_df.iloc[:10].to_dict(orient="records")
        ),
        "environment": env_manifest,
    }
    (RESULTS_DIR / "stage1_detailed_metrics.json").write_text(
        json.dumps(stage1_detailed_payload, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / "stage1_circuit_diagram.txt").write_text(
        stage1_out.circuit_report["diagram_text"],
        encoding="utf-8",
    )
    (RESULTS_DIR / "stage2_circuit_diagram.txt").write_text(
        str(build_dicke_10_6_bartschi_eidenbenz_circuit().draw(output="text")),
        encoding="utf-8",
    )
    (RESULTS_DIR / "stage3_circuit_diagram.txt").write_text(
        stage3_out.circuit_report["diagram_text"],
        encoding="utf-8",
    )

    figure_paths = generate_all_figures(stage1_out, stage2_out, stage3_out, RESULTS_DIR)
    total_runtime = time.perf_counter() - t_start

    # Write local execution manifest with SHA-256 hashes of inputs and outputs
    pdf_path = workbook_path.parent / "PROBLEM_STATEMENT_THEATOMIQHACKTHON.pdf"
    file_hashes: dict[str, str] = {
        "bank_campaign_participant_release.xlsx": compute_sha256(workbook_path),
        "stage1_results.csv": compute_sha256(stage1_csv_path),
        "stage2_results.json": compute_sha256(stage2_json_path),
        "stage3_results.json": compute_sha256(stage3_json_path),
    }
    if pdf_path.exists():
        file_hashes["PROBLEM_STATEMENT_THEATOMIQHACKTHON.pdf"] = compute_sha256(pdf_path)

    exec_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": env_manifest,
        "sha256_hashes": file_hashes,
        "total_pipeline_runtime_sec": float(total_runtime),
    }
    (RESULTS_DIR / "execution_manifest.json").write_text(
        json.dumps(exec_manifest, indent=2),
        encoding="utf-8",
    )

    summary = {
        "total_pipeline_runtime_sec": float(total_runtime),
        "stage1_selected_model": stage1_out.selected_model_name,
        "stage1_selected_config_id": "C02_Bandwidth_030_Reps2",
        "stage1_oof_roc_auc": stage1_out.cv_metrics_by_model[stage1_out.selected_model_name]["pooled_oof"]["roc_auc"],
        "stage1_oof_pr_auc_ap": stage1_out.cv_metrics_by_model[stage1_out.selected_model_name]["pooled_oof"]["pr_auc_average_precision"],
        "stage1_oof_brier": stage1_out.cv_metrics_by_model[stage1_out.selected_model_name]["pooled_oof"]["brier_score"],
        "stage1_oof_log_loss": stage1_out.cv_metrics_by_model[stage1_out.selected_model_name]["pooled_oof"]["log_loss"],
        "stage2_selected_customer_ids": stage2_out.selected_customer_ids,
        "stage2_selected_bitstring_le": stage2_out.selected_bitstring_little_endian,
        "stage2_quantum_objective": stage2_out.quantum_objective,
        "stage2_exact_optimal_objective": stage2_out.exact_reference.optimal_objective,
        "stage2_optimality_gap_abs": stage2_out.optimality_gap_abs,
        "stage2_total_cost": stage2_out.quantum_total_cost,
        "stage2_total_capacity": stage2_out.quantum_total_capacity,
        "stage2_transpiled_cx": stage2_out.circuit_report["two_qubit_gate_count_transpiled"],
        "stage2_transpiled_depth": stage2_out.circuit_report["transpiled_1q2q_depth"],
        "stage3_exact_prob_at_least_3": stage3_out.exact_dp_result["prob_at_least_3"],
        "stage3_ideal_sv_prob_at_least_3": stage3_out.ideal_statevector_result["prob_at_least_3"],
        "stage3_noiseless_shots_prob_at_least_3": stage3_out.noiseless_shot_result["estimated_prob_at_least_3"],
        "stage3_classical_mc_prob_at_least_3": stage3_out.classical_mc_result["estimated_prob_at_least_3"],
        "stage3_noisy_combined_prob_at_least_3": (
            stage3_out.noise_experiments["combined_depolarizing_and_readout"]["estimated_prob_at_least_3"]
        ),
        "figures_saved": figure_paths,
        "sha256_hashes": file_hashes,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Hybrid Quantum-Classical Term Deposit Campaign Pipeline."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run 5-fold CV strictly on the 300 Stage1_Train rows without touching Stage1_Test.",
    )
    args = parser.parse_args()
    main(validate_only=args.validate_only)
