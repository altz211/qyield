"""Stage 3: Campaign Outcome Simulation via 6-Qubit Quantum Bernoulli Circuit.

Simulates the probability that at least 3 of the 6 selected customers subscribe
(Hamming weight >= 3, i.e., >= 50% conversion target):
- Each selected customer k in {0..5} is mapped to qubit q_k with rotation:
    theta_k = 2 * arcsin(sqrt(p_k))
  applied via Ry(theta_k)|0>, yielding P(q_k = 1) = sin^2(theta_k / 2) = p_k.
- Compares:
    1. Independent Exact Classical Reference (Poisson-Binomial DP + 64-state enumeration)
    2. Ideal Quantum Statevector (exact 64 amplitudes)
    3. Noiseless Finite-Shot Quantum Simulator (`AerSimulator`, 100,000 shots)
    4. Classical Monte Carlo Simulation (100,000 trials)
    5. Separate Synthetic Noise Experiments:
       - Gate depolarizing noise only (`depolarizing_error` on `ry`, p_depol = 0.02)
       - Readout assignment error only (`ReadoutError` on `measure`, p_ro = 0.015)
       - Combined gate depolarizing + readout assignment error
    6. Exact Analytical Sensitivity Derivatives:
       dP(S >= 3)/dp_i = P(S_{-i} = 2) computed via 5-variable Poisson-Binomial DP,
       verified to machine precision against central finite differences.
- Explicitly documents that all Stage 3 probabilities are model-conditional on the
  Stage 1 predicted probabilities (not observed ground-truth test conversions).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from scipy.stats import beta as beta_dist

STAGE3_SEED = 42
N_SELECTED = 6
SUCCESS_MIN_WEIGHT = 3
STAGE3_SHOTS = 100_000
MC_TRIALS = 100_000
NOISE_P_DEPOL = 0.02
NOISE_P_READOUT = 0.015


def validate_stage3_probabilities(probs: np.ndarray) -> np.ndarray:
    """Validate that the 6 input probabilities are finite and strictly within [0.0, 1.0]."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.shape != (N_SELECTED,):
        raise ValueError(f"Stage 3 expects {N_SELECTED} probabilities, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Stage 3 probabilities contain NaN or infinite values.")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("Stage 3 probabilities must lie in [0.0, 1.0].")
    return arr


def compute_ry_angles(probs: np.ndarray) -> np.ndarray:
    """Compute Ry rotation angles theta_k = 2 * arcsin(sqrt(p_k))."""
    p_clean = validate_stage3_probabilities(probs)
    return 2.0 * np.arcsin(np.sqrt(p_clean))


def compute_exact_poisson_binomial_dp(probs: np.ndarray) -> dict[str, Any]:
    """Compute exact Poisson-Binomial PMF over k=0..6 successes via dynamic programming."""
    p_clean = validate_stage3_probabilities(probs)
    pmf = np.zeros(N_SELECTED + 1, dtype=np.float64)
    pmf[0] = 1.0
    for p in p_clean:
        next_pmf = np.zeros_like(pmf)
        next_pmf[0] = pmf[0] * (1.0 - p)
        next_pmf[1:] = pmf[1:] * (1.0 - p) + pmf[:-1] * p
        pmf = next_pmf

    norm_sum = float(np.sum(pmf))
    if abs(norm_sum - 1.0) > 1e-12:
        raise ValueError(f"Poisson-Binomial PMF failed normalization: sum={norm_sum}")

    prob_ge_3 = float(np.sum(pmf[SUCCESS_MIN_WEIGHT:]))
    return {
        "pmf_by_hamming_weight": {int(k): float(pmf[k]) for k in range(N_SELECTED + 1)},
        "prob_at_least_3": prob_ge_3,
        "normalization_sum": norm_sum,
    }


def compute_exact_stage3_analytical_derivatives(
    probs: np.ndarray,
    customer_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compute exact analytical sensitivity dP(S >= 3)/dp_i = P(S_{-i} = 2) for each customer i.

    By multilinear expansion of independent Bernoulli variables X_1..X_6,
        P(S >= 3) = p_i * P(S_{-i} >= 2) + (1 - p_i) * P(S_{-i} >= 3),
    so the partial derivative with respect to p_i is identically:
        dP(S >= 3) / dp_i = P(S_{-i} = 2).
    """
    p_clean = validate_stage3_probabilities(probs)
    cids = customer_ids if customer_ids is not None else [f"q_{i}" for i in range(N_SELECTED)]
    exact_jacobian: dict[str, float] = {}
    fd_jacobian: dict[str, float] = {}
    max_diff = 0.0
    eps = 1e-6

    for i in range(N_SELECTED):
        others = np.delete(p_clean, i)
        pmf_5 = np.zeros(N_SELECTED, dtype=np.float64)
        pmf_5[0] = 1.0
        for p_other in others:
            nxt = np.zeros_like(pmf_5)
            nxt[0] = pmf_5[0] * (1.0 - p_other)
            nxt[1:] = pmf_5[1:] * (1.0 - p_other) + pmf_5[:-1] * p_other
            pmf_5 = nxt
        exact_deriv = float(pmf_5[SUCCESS_MIN_WEIGHT - 1])  # P(S_{-i} = 2)

        p_plus = p_clean.copy()
        p_minus = p_clean.copy()
        p_plus[i] = min(1.0, p_clean[i] + eps)
        p_minus[i] = max(0.0, p_clean[i] - eps)
        fd_deriv = float(
            (
                compute_exact_poisson_binomial_dp(p_plus)["prob_at_least_3"]
                - compute_exact_poisson_binomial_dp(p_minus)["prob_at_least_3"]
            )
            / (p_plus[i] - p_minus[i])
        )
        diff = abs(exact_deriv - fd_deriv)
        max_diff = max(max_diff, diff)
        exact_jacobian[cids[i]] = exact_deriv
        fd_jacobian[cids[i]] = fd_deriv

    vals = list(exact_jacobian.values())
    return {
        "identity": "dP(S >= 3)/dp_i = P(S_{-i} = 2)",
        "analytical_gradient_wrt_p": exact_jacobian,
        "central_diff_gradient_wrt_p": fd_jacobian,
        "min_derivative": float(min(vals)),
        "max_derivative": float(max(vals)),
        "sum_of_derivatives_uniform_shift_rate": float(sum(vals)),
        "max_abs_diff_vs_central_difference": float(max_diff),
    }


def compute_exact_64_outcomes(probs: np.ndarray) -> dict[str, Any]:
    """Exhaustively enumerate all 2^6 = 64 bitstrings in Qiskit little-endian order 'b_5...b_0'."""
    p_clean = validate_stage3_probabilities(probs)
    outcome_probs: dict[str, float] = {}
    weight_probs = np.zeros(N_SELECTED + 1, dtype=np.float64)

    for idx in range(2**N_SELECTED):
        bits = [(idx >> q) & 1 for q in range(N_SELECTED)]
        bitstr_le = "".join(str(bits[q]) for q in range(N_SELECTED - 1, -1, -1))
        prob = 1.0
        for q, b in enumerate(bits):
            prob *= p_clean[q] if b == 1 else (1.0 - p_clean[q])
        outcome_probs[bitstr_le] = float(prob)
        weight_probs[sum(bits)] += prob

    prob_ge_3 = float(np.sum(weight_probs[SUCCESS_MIN_WEIGHT:]))
    return {
        "outcome_probabilities_le": outcome_probs,
        "pmf_by_hamming_weight": {int(k): float(weight_probs[k]) for k in range(N_SELECTED + 1)},
        "prob_at_least_3": prob_ge_3,
        "total_probability": float(np.sum(weight_probs)),
    }


def build_stage3_bernoulli_circuit(
    probs: np.ndarray,
    include_measurements: bool = False,
) -> QuantumCircuit:
    """Build the 6-qubit independent Bernoulli Ry quantum circuit."""
    angles = compute_ry_angles(probs)
    qc = QuantumCircuit(
        N_SELECTED,
        N_SELECTED if include_measurements else 0,
        name="Stage3_6Q_Bernoulli",
    )
    for q in range(N_SELECTED):
        qc.ry(float(angles[q]), q)
    if include_measurements:
        qc.measure(list(range(N_SELECTED)), list(range(N_SELECTED)))
    return qc


def clopper_pearson_ci_95(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute exact 95% Clopper-Pearson binomial confidence interval for k successes in n trials."""
    if successes <= 0:
        low = 0.0
    else:
        low = float(beta_dist.ppf(alpha / 2.0, successes, total - successes + 1))
    if successes >= total:
        high = 1.0
    else:
        high = float(beta_dist.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return low, high


def _summarize_counts(
    counts: dict[str, int],
    shots: int,
    exact_prob: float,
) -> dict[str, Any]:
    """Aggregate raw bitstring counts into Hamming-weight distribution and success statistics."""
    weight_counts = {k: 0 for k in range(N_SELECTED + 1)}
    success_shots = 0

    for bitstr, c in counts.items():
        clean = bitstr.replace(" ", "")
        w = clean.count("1")
        weight_counts[w] += int(c)
        if w >= SUCCESS_MIN_WEIGHT:
            success_shots += int(c)

    est_prob = float(success_shots / shots)
    ci_low, ci_high = clopper_pearson_ci_95(success_shots, shots)
    std_err = float(np.sqrt(max(0.0, est_prob * (1.0 - est_prob)) / shots))

    return {
        "shots": int(shots),
        "success_shots_weight_ge_3": int(success_shots),
        "estimated_prob_at_least_3": est_prob,
        "abs_error_vs_exact": float(abs(est_prob - exact_prob)),
        "signed_error_vs_exact": float(est_prob - exact_prob),
        "standard_error": std_err,
        "ci_95_clopper_pearson": [ci_low, ci_high],
        "counts_by_hamming_weight": weight_counts,
        "pmf_by_hamming_weight": {k: float(v / shots) for k, v in weight_counts.items()},
        "raw_bitstring_counts": {k: int(v) for k, v in sorted(counts.items())},
    }


def run_noise_experiments(
    qc_meas: QuantumCircuit,
    probs: np.ndarray,
    exact_prob: float,
    shots: int = STAGE3_SHOTS,
    seed: int = STAGE3_SEED,
) -> dict[str, Any]:
    """Run isolated and combined synthetic noise experiments in AerSimulator.

    Separates:
    1. 1-qubit gate depolarizing noise (`depolarizing_error(p_depol, 1)` on `ry`)
    2. Symmetric measurement readout assignment error (`ReadoutError` on `measure`)
    3. Combined gate depolarizing + readout assignment error
    """
    p_clean = validate_stage3_probabilities(probs)

    # 1. Gate depolarizing error only
    nm_depol = NoiseModel()
    nm_depol.add_all_qubit_quantum_error(depolarizing_error(NOISE_P_DEPOL, 1), ["ry"])
    sim_depol = AerSimulator(noise_model=nm_depol, seed_simulator=seed + 1)
    t_depol = transpile(qc_meas, sim_depol, optimization_level=0, seed_transpiler=seed)
    counts_depol = sim_depol.run(t_depol, shots=shots, seed_simulator=seed + 1).result().get_counts()
    p_marginal_depol = (1.0 - NOISE_P_DEPOL) * p_clean + NOISE_P_DEPOL * 0.5
    analytical_depol_prob = compute_exact_poisson_binomial_dp(p_marginal_depol)["prob_at_least_3"]

    # 2. Readout assignment error only
    nm_readout = NoiseModel()
    ro_err = ReadoutError(
        [[1.0 - NOISE_P_READOUT, NOISE_P_READOUT], [NOISE_P_READOUT, 1.0 - NOISE_P_READOUT]]
    )
    nm_readout.add_all_qubit_readout_error(ro_err)
    sim_ro = AerSimulator(noise_model=nm_readout, seed_simulator=seed + 2)
    t_ro = transpile(qc_meas, sim_ro, optimization_level=0, seed_transpiler=seed)
    counts_ro = sim_ro.run(t_ro, shots=shots, seed_simulator=seed + 2).result().get_counts()
    p_marginal_ro = p_clean * (1.0 - 2.0 * NOISE_P_READOUT) + NOISE_P_READOUT
    analytical_ro_prob = compute_exact_poisson_binomial_dp(p_marginal_ro)["prob_at_least_3"]

    # 3. Combined gate depolarizing + readout assignment error
    nm_combined = NoiseModel()
    nm_combined.add_all_qubit_quantum_error(depolarizing_error(NOISE_P_DEPOL, 1), ["ry"])
    nm_combined.add_all_qubit_readout_error(ro_err)
    sim_comb = AerSimulator(noise_model=nm_combined, seed_simulator=seed + 3)
    t_comb = transpile(qc_meas, sim_comb, optimization_level=0, seed_transpiler=seed)
    counts_comb = sim_comb.run(t_comb, shots=shots, seed_simulator=seed + 3).result().get_counts()
    p_marginal_comb = p_marginal_depol * (1.0 - 2.0 * NOISE_P_READOUT) + NOISE_P_READOUT
    analytical_comb_prob = compute_exact_poisson_binomial_dp(p_marginal_comb)["prob_at_least_3"]

    depol_summary = _summarize_counts(counts_depol, shots, exact_prob)
    depol_summary["analytical_noisy_prob_at_least_3"] = float(analytical_depol_prob)
    depol_summary["noise_specification"] = (
        f"1-qubit depolarizing_error(p={NOISE_P_DEPOL}) on all 'ry' gates; zero readout error."
    )

    ro_summary = _summarize_counts(counts_ro, shots, exact_prob)
    ro_summary["analytical_noisy_prob_at_least_3"] = float(analytical_ro_prob)
    ro_summary["noise_specification"] = (
        f"Symmetric ReadoutError(p(1|0)=p(0|1)={NOISE_P_READOUT}) on all measurements; zero gate error."
    )

    comb_summary = _summarize_counts(counts_comb, shots, exact_prob)
    comb_summary["analytical_noisy_prob_at_least_3"] = float(analytical_comb_prob)
    comb_summary["noise_specification"] = (
        f"Combined 1-qubit depolarizing_error(p={NOISE_P_DEPOL}) on 'ry' + "
        f"symmetric ReadoutError(p={NOISE_P_READOUT}) on measurements (unmitigated)."
    )

    return {
        "gate_depolarizing_only": depol_summary,
        "readout_error_only": ro_summary,
        "combined_depolarizing_and_readout": comb_summary,
        "physical_interpretation": (
            "Both single-qubit depolarizing noise and symmetric readout assignment error contract "
            "each qubit's marginal Bernoulli parameter p_k toward 0.5. Whenever the customer "
            "probabilities p_k lie below 0.5, this contraction systematically inflates the "
            "unmitigated tail probability P(Hamming weight >= 3)."
        ),
    }


def verify_stage3_extreme_inputs() -> dict[str, Any]:
    """Run boundary/extreme input sanity checks on the Stage 3 circuit and DP reference."""
    cases = {
        "all_zeros": (np.zeros(6), 0.0),
        "all_ones": (np.ones(6), 1.0),
        "three_ones_three_zeros": (np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), 1.0),
        "two_ones_four_zeros": (np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]), 0.0),
        "uniform_half": (np.full(6, 0.5), 42.0 / 64.0),
    }
    report: dict[str, Any] = {}
    for name, (p_vec, expected_val) in cases.items():
        dp_val = compute_exact_poisson_binomial_dp(p_vec)["prob_at_least_3"]
        qc = build_stage3_bernoulli_circuit(p_vec, include_measurements=False)
        sv_probs = np.abs(Statevector.from_instruction(qc).data) ** 2
        weights = np.array([bin(idx).count("1") for idx in range(64)])
        sv_val = float(np.sum(sv_probs[weights >= SUCCESS_MIN_WEIGHT]))
        if abs(dp_val - expected_val) > 1e-12 or abs(sv_val - expected_val) > 1e-12:
            raise ValueError(
                f"Extreme input check '{name}' failed: expected={expected_val}, dp={dp_val}, sv={sv_val}"
            )
        report[name] = {
            "expected": expected_val,
            "dp_exact": dp_val,
            "statevector_exact": sv_val,
            "abs_diff": float(abs(sv_val - dp_val)),
        }
    return report


@dataclass(frozen=True)
class Stage3Output:
    """Complete Stage 3 simulation output."""

    selected_customers_table: list[dict[str, Any]]
    exact_dp_result: dict[str, Any]
    exact_64_result: dict[str, Any]
    ideal_statevector_result: dict[str, Any]
    noiseless_shot_result: dict[str, Any]
    classical_mc_result: dict[str, Any]
    noise_experiments: dict[str, Any]
    extreme_input_checks: dict[str, Any]
    analytical_derivatives: dict[str, Any]
    circuit_report: dict[str, Any]
    runtime_sec: float


def run_stage3_pipeline(
    selected_df: pd.DataFrame,
    shots: int = STAGE3_SHOTS,
    mc_trials: int = MC_TRIALS,
    seed: int = STAGE3_SEED,
) -> Stage3Output:
    """Execute Stage 3 quantum and classical campaign outcome simulation."""
    t0 = time.perf_counter()
    if len(selected_df) != N_SELECTED:
        raise ValueError(f"Expected {N_SELECTED} selected customers for Stage 3, got {len(selected_df)}.")

    # Preserve exact Stage 2 order: q_k corresponds to row k of selected_df
    cids = selected_df["customer_id"].astype(str).tolist()
    probs = validate_stage3_probabilities(selected_df["predicted_probability"].to_numpy(dtype=np.float64))
    angles = compute_ry_angles(probs)

    selected_table = [
        {
            "qubit_index": int(k),
            "bitstring_char_position_from_right": int(k),
            "bitstring_char_index_in_6char_str": int(N_SELECTED - 1 - k),
            "customer_id": cids[k],
            "predicted_probability": float(probs[k]),
            "ry_angle_rad": float(angles[k]),
            "expected_deposit_value": float(selected_df.iloc[k]["expected_deposit_value"]),
            "contact_cost": float(selected_df.iloc[k]["contact_cost"]),
            "capacity_units": float(selected_df.iloc[k]["capacity_units"]),
            "expected_net_value": float(selected_df.iloc[k]["expected_net_value"]),
        }
        for k in range(N_SELECTED)
    ]

    # 1. Independent Exact Classical Reference (DP + 64-outcome enumeration)
    exact_dp = compute_exact_poisson_binomial_dp(probs)
    exact_64 = compute_exact_64_outcomes(probs)
    exact_prob = exact_dp["prob_at_least_3"]
    if abs(exact_prob - exact_64["prob_at_least_3"]) > 1e-14:
        raise RuntimeError("Mismatch between Poisson-Binomial DP and 64-outcome enumeration.")

    # 2. Ideal Quantum Statevector
    qc_ideal = build_stage3_bernoulli_circuit(probs, include_measurements=False)
    sv = Statevector.from_instruction(qc_ideal)
    sv_probs = np.abs(sv.data) ** 2
    sv_norm = float(np.sum(sv_probs))
    if abs(sv_norm - 1.0) > 1e-12:
        raise ValueError(f"Stage 3 statevector failed normalization: {sv_norm}")

    sv_pmf = np.zeros(N_SELECTED + 1, dtype=np.float64)
    for idx in range(2**N_SELECTED):
        w = int(bin(idx).count("1"))
        sv_pmf[w] += float(sv_probs[idx])
    sv_prob_ge_3 = float(np.sum(sv_pmf[SUCCESS_MIN_WEIGHT:]))

    ideal_sv_result = {
        "prob_at_least_3": sv_prob_ge_3,
        "abs_error_vs_exact_dp": float(abs(sv_prob_ge_3 - exact_prob)),
        "normalization_sum": sv_norm,
        "pmf_by_hamming_weight": {int(k): float(sv_pmf[k]) for k in range(N_SELECTED + 1)},
    }

    # 3. Noiseless Finite-Shot Quantum Simulator
    qc_meas = build_stage3_bernoulli_circuit(probs, include_measurements=True)
    sim_noiseless = AerSimulator(seed_simulator=seed)
    t_qc = transpile(qc_meas, sim_noiseless, optimization_level=1, seed_transpiler=seed)
    counts_noiseless = (
        sim_noiseless.run(t_qc, shots=shots, seed_simulator=seed).result().get_counts()
    )
    noiseless_shot_result = _summarize_counts(counts_noiseless, shots, exact_prob)

    # 4. Classical Monte Carlo Simulation
    rng = np.random.default_rng(seed)
    draws = rng.random((mc_trials, N_SELECTED)) < probs[np.newaxis, :]
    trial_weights = np.sum(draws, axis=1)
    mc_successes = int(np.sum(trial_weights >= SUCCESS_MIN_WEIGHT))
    mc_prob = float(mc_successes / mc_trials)
    mc_ci_low, mc_ci_high = clopper_pearson_ci_95(mc_successes, mc_trials)
    mc_pmf = {
        int(k): float(np.mean(trial_weights == k)) for k in range(N_SELECTED + 1)
    }
    classical_mc_result = {
        "trials": int(mc_trials),
        "seed": int(seed),
        "success_trials_weight_ge_3": mc_successes,
        "estimated_prob_at_least_3": mc_prob,
        "abs_error_vs_exact": float(abs(mc_prob - exact_prob)),
        "signed_error_vs_exact": float(mc_prob - exact_prob),
        "standard_error": float(np.sqrt(max(0.0, mc_prob * (1.0 - mc_prob)) / mc_trials)),
        "ci_95_clopper_pearson": [mc_ci_low, mc_ci_high],
        "pmf_by_hamming_weight": mc_pmf,
    }

    # 5. Separate Synthetic Noise Experiments
    noise_experiments = run_noise_experiments(
        qc_meas=qc_meas,
        probs=probs,
        exact_prob=exact_prob,
        shots=shots,
        seed=seed,
    )

    # 6. Extreme-input verification + Exact Analytical Sensitivity Derivatives
    extreme_checks = verify_stage3_extreme_inputs()
    analytical_derivs = compute_exact_stage3_analytical_derivatives(probs, customer_ids=cids)

    orig_ops = {str(k): int(v) for k, v in qc_meas.count_ops().items()}
    trans_ops = {str(k): int(v) for k, v in t_qc.count_ops().items()}
    circuit_report = {
        "circuit_name": qc_meas.name,
        "logical_qubits": N_SELECTED,
        "physical_qubits": "N/A (local AerSimulator; 6 logical qubits)",
        "original_depth": int(qc_meas.depth()),
        "transpiled_depth": int(t_qc.depth()),
        "original_gate_counts": orig_ops,
        "transpiled_gate_counts": trans_ops,
        "one_qubit_gate_count": int(orig_ops.get("ry", 0)),
        "two_qubit_gate_count": 0,
        "swap_count": 0,
        "bit_ordering_convention": (
            "Qiskit little-endian: in a 6-bit measurement string 'b5 b4 b3 b2 b1 b0', "
            "the rightmost character b0 is qubit q0 (selected customer index 0) and "
            "the leftmost character b5 is qubit q5 (selected customer index 5)."
        ),
        "diagram_text": str(qc_meas.draw(output="text")),
    }

    elapsed = time.perf_counter() - t0
    return Stage3Output(
        selected_customers_table=selected_table,
        exact_dp_result=exact_dp,
        exact_64_result=exact_64,
        ideal_statevector_result=ideal_sv_result,
        noiseless_shot_result=noiseless_shot_result,
        classical_mc_result=classical_mc_result,
        noise_experiments=noise_experiments,
        extreme_input_checks=extreme_checks,
        analytical_derivatives=analytical_derivs,
        circuit_report=circuit_report,
        runtime_sec=float(elapsed),
    )
