"""Comprehensive automated verification suite for the AtomiQ Hackathon Submission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit.quantum_info import Statevector

from run_pipeline import main as run_pipeline_main
from src.data_loader import (
    EXPECTED_FEATURE_COLUMNS,
    Stage2Constraints,
    load_and_verify_workbook,
    resolve_workbook_path,
)
from src.stage1_predict import (
    QUANTUM_ANGLE_SCALE,
    EndToEndQuantumClassifier,
    build_4qubit_feature_map_circuit,
    compute_quantum_fidelity_gram,
    evaluate_cached_statevectors,
    rank_stage1_predictions,
    validate_probabilities_array,
    validate_self_gram_matrix,
)
from src.stage2_optimize import (
    N_CANDIDATES,
    bitstring_le_to_bitvector,
    bitvector_to_bitstring_le,
    build_dicke_10_6_bartschi_eidenbenz_circuit,
    build_stage2_qaoa_circuit,
    build_uniform_dicke_statevector,
    compute_exact_reference_and_diagonal,
    index_to_bitvector_le,
    prepare_top10_candidates,
    prove_all_weight6_subsets_feasible,
    run_stage2_pipeline,
    verify_hamiltonian_and_mixer,
)
from src.stage3_simulate import (
    build_stage3_bernoulli_circuit,
    compute_exact_64_outcomes,
    compute_exact_poisson_binomial_dp,
    compute_exact_stage3_analytical_derivatives,
    verify_stage3_extreme_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_validate_only_reads_strictly_stage1_train_sheet(monkeypatch: Any) -> None:
    """Verify `--validate-only` reads ONLY the `Stage1_Train` sheet and never touches `Stage1_Test`."""
    real_read_excel = pd.read_excel
    sheets_accessed: list[str] = []

    def spied_read_excel(*args: Any, **kwargs: Any) -> Any:
        sheet = kwargs.get("sheet_name")
        if sheet is None and len(args) >= 2:
            sheet = args[1]
        sheets_accessed.append(str(sheet))
        return real_read_excel(*args, **kwargs)

    monkeypatch.setattr(pd, "read_excel", spied_read_excel)
    val_summary = run_pipeline_main(validate_only=True)

    assert sheets_accessed == ["Stage1_Train"]
    assert val_summary["stage1_test_touched"] is False
    assert val_summary["sheets_read"] == ["Stage1_Train"]
    assert QUANTUM_ANGLE_SCALE == 0.30


def test_stage1_fold_isolation_and_gram_psd() -> None:
    """Verify fold isolation, C02 angle_scale=0.30 default, and PSD/unit-diagonal Gram properties."""
    dataset = load_and_verify_workbook()
    X_train = dataset.stage1_train[EXPECTED_FEATURE_COLUMNS].iloc[:80].copy()
    y_train = (dataset.stage1_train["y"].iloc[:80] == "yes").astype(int).to_numpy()
    X_val = dataset.stage1_train[EXPECTED_FEATURE_COLUMNS].iloc[80:110].copy()

    clf = EndToEndQuantumClassifier(mode="hybrid_feature_aug", C=0.12, seed=42)
    assert clf.angle_scale == 0.30
    clf.fit(X_train, y_train)

    audit = validate_self_gram_matrix(
        compute_quantum_fidelity_gram(clf.sv_train_, clf.sv_train_)
    )
    assert audit["is_psd"] is True
    assert audit["max_asymmetry"] < 1e-12
    assert audit["max_diagonal_error_from_1"] < 1e-12

    probs_clean = clf.predict_proba(X_val)[:, 1]
    validate_probabilities_array(probs_clean)

    X_val_corrupted = X_val.copy()
    X_val_corrupted.iloc[5:, 0] = 999999
    probs_corrupted = clf.predict_proba(X_val_corrupted)[:, 1]
    np.testing.assert_allclose(probs_clean[:5], probs_corrupted[:5], atol=1e-13)

    angles = np.array([[0.15, -0.22, 0.31, -0.10], [-0.25, 0.18, -0.12, 0.29]])
    sv = evaluate_cached_statevectors(angles)
    for i in range(len(angles)):
        qc = build_4qubit_feature_map_circuit(angles[i])
        sv_ref = Statevector.from_instruction(qc).data
        np.testing.assert_allclose(sv[i], sv_ref, atol=1e-12)


def test_stage1_to_stage2_ranking_and_id_joins() -> None:
    """Verify Stage 1 ranking, tie-breaking, and Stage 2 top-10 candidate join integrity."""
    dataset = load_and_verify_workbook()
    s1_df = pd.read_csv(PROJECT_ROOT / "stage1_results.csv")

    assert s1_df.shape == (100, 4)
    assert list(s1_df.columns) == [
        "customer_id",
        "predicted_probability",
        "rank",
        "top_10_flag",
    ]
    assert set(s1_df["customer_id"]) == set(dataset.stage1_test["customer_id"])
    validate_probabilities_array(s1_df["predicted_probability"].to_numpy())
    assert list(s1_df["rank"]) == list(range(1, 101))
    assert int(s1_df["top_10_flag"].sum()) == 10
    assert list(s1_df["top_10_flag"].iloc[:10]) == [1] * 10
    assert list(s1_df["top_10_flag"].iloc[10:]) == [0] * 90

    tie_df = rank_stage1_predictions(
        ["CUST_0099", "CUST_0002", "CUST_0050"],
        np.array([0.4, 0.4, 0.7]),
    )
    assert list(tie_df["customer_id"]) == ["CUST_0050", "CUST_0002", "CUST_0099"]

    top10 = prepare_top10_candidates(s1_df, dataset.stage2_data_test_only, dataset.constraints)
    assert len(top10) == 10
    assert list(top10["customer_id"]) == list(s1_df["customer_id"].iloc[:10])


def test_stage2_bartschi_eidenbenz_dicke_and_guarded_dispatch_equivalence() -> None:
    """Verify Bärtschi–Eidenbenz Dicke state fidelity, CX reduction, and 210-subset guarded RZ equivalence."""
    dataset = load_and_verify_workbook()
    s1_df = pd.read_csv(PROJECT_ROOT / "stage1_results.csv")
    candidates_df = prepare_top10_candidates(
        s1_df, dataset.stage2_data_test_only, dataset.constraints
    )
    exact_ref = compute_exact_reference_and_diagonal(candidates_df, dataset.constraints)
    dicke_sv = build_uniform_dicke_statevector(N_CANDIDATES, 6)

    # 1. Verify Bärtschi–Eidenbenz Dicke state preparation fidelity and CX reduction
    qc_be = build_dicke_10_6_bartschi_eidenbenz_circuit()
    sv_be = Statevector.from_instruction(qc_be).data
    fidelity = float(np.abs(np.vdot(dicke_sv, sv_be)) ** 2)
    assert abs(1.0 - fidelity) < 1e-14

    t_be = transpile(qc_be, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=42)
    assert int(t_be.count_ops().get("cx", 0)) == 246

    # 2. Verify O(N log N) feasibility guard proves all 210 weight-6 subsets are feasible on this pool
    net_values = candidates_df["expected_net_value"].to_numpy(dtype=np.float64)
    costs = candidates_df["contact_cost"].to_numpy(dtype=np.float64)
    caps = candidates_df["capacity_units"].to_numpy(dtype=np.float64)
    guard = prove_all_weight6_subsets_feasible(costs, caps, k=6, budget_limit_c=25.0, capacity_limit_h=22.0)
    assert guard["all_weight_k_subsets_proved_feasible"] is True
    assert exact_ref.feasible_weight6_subsets == 210

    # 3. Verify across multiple parameter vectors that guarded RZ + Bärtschi–Eidenbenz Dicke prep
    # matches unreduced StatePreparation + DiagonalGate probabilities across all 1024 states
    param_sets = [
        (np.array([0.11, 0.23, 0.17]), np.array([0.19, 0.31, 0.14])),
        (np.array([0.29, 0.07, 0.34]), np.array([0.41, 0.12, 0.27])),
    ]
    for gammas, betas in param_sets:
        qc_guarded = build_stage2_qaoa_circuit(
            gammas,
            betas,
            exact_ref.all_1024_energies,
            dicke_sv,
            net_values=net_values,
            all_weight6_feasible=True,
            noise_model=None,
            use_guarded_rz_dispatch=True,
            use_bartschi_dicke_prep=True,
        )
        qc_unreduced = build_stage2_qaoa_circuit(
            gammas,
            betas,
            exact_ref.all_1024_energies,
            dicke_sv,
            net_values=net_values,
            all_weight6_feasible=True,
            noise_model=None,
            use_guarded_rz_dispatch=False,
            use_bartschi_dicke_prep=False,
        )
        p_guarded = np.abs(Statevector.from_instruction(qc_guarded).data) ** 2
        p_unreduced = np.abs(Statevector.from_instruction(qc_unreduced).data) ** 2
        np.testing.assert_allclose(p_guarded, p_unreduced, atol=1e-12)

    t_guarded = transpile(
        qc_guarded, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=42
    )
    t_unreduced = transpile(
        qc_unreduced, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=42
    )
    assert int(t_guarded.count_ops().get("cx", 0)) == 426
    assert int(t_guarded.depth()) == 741
    assert int(t_unreduced.count_ops().get("cx", 0)) == 4259

    ham_v, mix_v = verify_hamiltonian_and_mixer(exact_ref.all_1024_energies, dicke_sv)
    assert ham_v["max_unitary_phase_error_all_1024"] < 1e-12
    assert mix_v["conserves_hamming_weight"] is True


def test_stage2_binding_capacity_and_noisy_mode_trigger_diagonal_fallback() -> None:
    """Verify automatic fallback to DiagonalGate when capacity binds or when noise_model is active."""
    dataset = load_and_verify_workbook()
    s1_df = pd.read_csv(PROJECT_ROOT / "stage1_results.csv")
    candidates_df = prepare_top10_candidates(
        s1_df, dataset.stage2_data_test_only, dataset.constraints
    )
    exact_ref = compute_exact_reference_and_diagonal(candidates_df, dataset.constraints)
    dicke_sv = build_uniform_dicke_statevector(N_CANDIDATES, 6)
    net_values = candidates_df["expected_net_value"].to_numpy(dtype=np.float64)

    # 1. Non-None noise_model must force DiagonalGate fallback even when all_weight6_feasible=True
    qc_noisy_fallback = build_stage2_qaoa_circuit(
        np.array([0.15, 0.25]),
        np.array([0.20, 0.30]),
        exact_ref.all_1024_energies,
        dicke_sv,
        net_values=net_values,
        all_weight6_feasible=True,
        noise_model={"synthetic_noise": True},
        use_guarded_rz_dispatch=True,
        use_bartschi_dicke_prep=True,
    )
    assert "diagonal" in qc_noisy_fallback.count_ops()

    # 2. Binding capacity instance: 6 candidates have capacity 4 (6*4 = 24 > 22)
    stage2_binding_df = dataset.stage2_data_test_only.copy()
    top6_cids = list(s1_df["customer_id"].iloc[:6])
    stage2_binding_df.loc[
        stage2_binding_df["customer_id"].isin(top6_cids), "capacity_units"
    ] = 4

    s2_binding_out = run_stage2_pipeline(
        stage1_results_df=s1_df,
        stage2_test_data_df=stage2_binding_df,
        constraints=Stage2Constraints(10, 6, 25.0, 22.0),
        seed=42,
    )
    assert s2_binding_out.exact_reference.capacity_violating_weight6_subsets > 0
    assert s2_binding_out.circuit_report["guarded_cost_layer_mode"] == "exact_10q_diagonal_gate_fallback"
    assert s2_binding_out.quantum_is_feasible is True
    assert s2_binding_out.quantum_total_capacity <= 22.0
    assert abs(s2_binding_out.quantum_objective - s2_binding_out.exact_reference.optimal_objective) < 1e-9

    for idx in (0, 7, 42, 511, 1023):
        bv = index_to_bitvector_le(idx)
        bs = bitvector_to_bitstring_le(bv)
        np.testing.assert_array_equal(bv, bitstring_le_to_bitvector(bs))


def test_stage3_exact_dp_statevector_and_analytical_derivatives() -> None:
    """Verify Stage 3 DP, 64-outcome enumeration, ideal statevector, and dP(S>=3)/dp_i = P(S_{-i}=2)."""
    s3_json = json.loads((PROJECT_ROOT / "stage3_results.json").read_text(encoding="utf-8"))
    probs = np.array(
        [r["predicted_probability"] for r in s3_json["qubit_to_customer_mapping"]],
        dtype=np.float64,
    )

    dp = compute_exact_poisson_binomial_dp(probs)
    enum64 = compute_exact_64_outcomes(probs)
    qc = build_stage3_bernoulli_circuit(probs, include_measurements=False)
    sv_probs = np.abs(Statevector.from_instruction(qc).data) ** 2
    weights = np.array([bin(i).count("1") for i in range(64)])
    sv_ge3 = float(np.sum(sv_probs[weights >= 3]))

    assert abs(dp["prob_at_least_3"] - enum64["prob_at_least_3"]) < 1e-14
    assert abs(dp["prob_at_least_3"] - sv_ge3) < 1e-14

    shot_ci = s3_json["noiseless_quantum_shot_simulation"]["ci_95_clopper_pearson"]
    mc_ci = s3_json["classical_monte_carlo_simulation"]["ci_95_clopper_pearson"]
    assert shot_ci[0] <= dp["prob_at_least_3"] <= shot_ci[1]
    assert mc_ci[0] <= dp["prob_at_least_3"] <= mc_ci[1]

    deriv_info = compute_exact_stage3_analytical_derivatives(probs)
    assert deriv_info["max_abs_diff_vs_central_difference"] < 1e-10
    for val in deriv_info["analytical_gradient_wrt_p"].values():
        assert 0.0 < val < 1.0

    extremes = verify_stage3_extreme_inputs()
    assert extremes["all_zeros"]["statevector_exact"] == 0.0
    assert extremes["all_ones"]["statevector_exact"] == 1.0


def test_deliverables_and_source_file_hashes() -> None:
    """Verify all required top-level deliverables and unchanged source file SHA-256 hashes."""
    for rel in (
        "README.md",
        "requirements.txt",
        "frozen_config.json",
        "run_pipeline.py",
        "stage1_results.csv",
        "stage2_results.json",
        "stage3_results.json",
    ):
        assert (PROJECT_ROOT / rel).exists(), f"Missing top-level file: {rel}"

    wb_path = resolve_workbook_path()
    assert _sha256(wb_path) == "d2f8e63705af2883c7618c463a6ac2f351119d59a8869d2677f8825148e9a709"

    pdf_path = wb_path.parent / "PROBLEM_STATEMENT_THEATOMIQHACKTHON.pdf"
    if pdf_path.exists():
        assert _sha256(pdf_path) == "8b1d38d02010fbad9c297d57c13085b01d9b26a3741152a21a54cc3609c48363"