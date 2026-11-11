"""Visualization and circuit diagram export utilities for Stages 1, 2, and 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.stage1_predict import Stage1Output
from src.stage2_optimize import Stage2Output
from src.stage3_simulate import Stage3Output


def generate_all_figures(
    stage1_out: Stage1Output,
    stage2_out: Stage2Output,
    stage3_out: Stage3Output,
    results_dir: str | Path,
) -> list[str]:
    """Generate diagnostic figures for Stages 1, 2, and 3 and save under `results_dir`."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    # 1. Stage 1 Model Comparison & Calibration Plot
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    model_names = list(stage1_out.cv_metrics_by_model.keys())
    short_labels = [
        m.replace("-LandmarkLogReg", "")
        .replace("Classical-", "Cls-")
        .replace("Diagnostic-", "Diag-")
        .replace("Exploratory-", "Exp-")
        .replace("Reference-", "Ref-")
        for m in model_names
    ]
    roc_vals = [stage1_out.cv_metrics_by_model[m]["pooled_oof"]["roc_auc"] for m in model_names]
    ap_vals = [
        stage1_out.cv_metrics_by_model[m]["pooled_oof"]["pr_auc_average_precision"]
        for m in model_names
    ]
    brier_vals = [
        stage1_out.cv_metrics_by_model[m]["pooled_oof"]["brier_score"] for m in model_names
    ]

    x = np.arange(len(model_names))
    w = 0.26
    axes[0].bar(x - w, roc_vals, width=w, label="ROC-AUC", color="#1f77b4")
    axes[0].bar(x, ap_vals, width=w, label="PR-AUC (AP)", color="#2ca02c")
    axes[0].bar(x + w, brier_vals, width=w, label="Brier Score (proper, lower better)", color="#d62728")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Score (5-Fold OOF on N=300 Train)")
    axes[0].set_title("Stage 1: Out-of-Fold Ranking & Probability Metrics")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot([0, 0.65], [0, 0.65], "k--", alpha=0.6, label="Perfect Calibration")
    for m_name, label in zip(model_names, short_labels):
        cal = stage1_out.cv_metrics_by_model[m_name]["calibration_curve_5quantile"]
        axes[1].plot(
            cal["prob_pred"],
            cal["prob_true"],
            marker="o",
            linewidth=1.5,
            label=label,
        )
    axes[1].set_xlabel("Mean Predicted Probability (Quantile Bins)")
    axes[1].set_ylabel("Empirical Positive Fraction")
    axes[1].set_title("Stage 1: Reliability / Calibration Comparison (OOF)")
    axes[1].set_xlim(0.0, 0.65)
    axes[1].set_ylim(0.0, 0.65)
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    p1 = out_dir / "stage1_validation_and_calibration.png"
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    saved_paths.append(str(p1))

    # 2. Stage 2 QAOA Sampling & Feasible Subset Landscape
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    top_meas = stage2_out.sampling_summary["top_measured_bitstrings"]
    b_labels = [d["bitstring_little_endian"] for d in top_meas]
    b_probs = [d["shot_probability"] for d in top_meas]
    b_colors = [
        "#2ca02c"
        if d["bitstring_little_endian"] == stage2_out.exact_reference.optimal_bitstring_little_endian
        else ("#1f77b4" if d["is_feasible"] else "#d62728")
        for d in top_meas
    ]
    axes[0].barh(np.arange(len(b_labels))[::-1], b_probs, color=b_colors)
    axes[0].set_yticks(np.arange(len(b_labels))[::-1])
    axes[0].set_yticklabels(b_labels, fontfamily="monospace", fontsize=9)
    axes[0].set_xlabel("Measured Probability (8,192 Shots)")
    axes[0].set_title("Stage 2: Top Measured 10-Qubit XY-QAOA Bitstrings\n(Green = Exact Optimum, Blue = Feasible)")
    axes[0].grid(axis="x", alpha=0.3)

    objs_1024 = stage2_out.exact_reference.all_1024_objectives
    feas_1024 = stage2_out.exact_reference.all_1024_feasible_mask
    weights_1024 = np.array([bin(i).count("1") for i in range(1024)])
    w6_mask = weights_1024 == 6
    w6_feas = w6_mask & feas_1024
    w6_infeas = w6_mask & (~feas_1024)

    axes[1].hist(
        objs_1024[w6_feas],
        bins=18,
        alpha=0.75,
        color="#1f77b4",
        label=f"Feasible K=6 Subsets ({int(np.sum(w6_feas))})",
    )
    if np.any(w6_infeas):
        axes[1].hist(
            objs_1024[w6_infeas],
            bins=10,
            alpha=0.65,
            color="#ff7f0e",
            label=f"Capacity-Violating K=6 Subsets ({int(np.sum(w6_infeas))})",
        )
    axes[1].axvline(
        stage2_out.quantum_objective,
        color="#2ca02c",
        linestyle="--",
        linewidth=2.2,
        label=f"QAOA Selected = {stage2_out.quantum_objective:.4f}",
    )
    axes[1].axvline(
        stage2_out.exact_reference.optimal_objective,
        color="#d62728",
        linestyle=":",
        linewidth=2.2,
        label=f"Exact Optimum = {stage2_out.exact_reference.optimal_objective:.4f}",
    )
    axes[1].set_xlabel("Expected Net Value Z(x)")
    axes[1].set_ylabel("Number of Weight-6 Subsets")
    axes[1].set_title("Stage 2: Exhaustive 210 Weight-6 Subset Distribution")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    p2 = out_dir / "stage2_qaoa_and_subset_distribution.png"
    fig.savefig(p2, dpi=180)
    plt.close(fig)
    saved_paths.append(str(p2))

    # 3. Stage 3 Poisson-Binomial PMF & Noise Impact Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    ks = np.arange(7)
    pmf_exact = [stage3_out.exact_dp_result["pmf_by_hamming_weight"][k] for k in ks]
    pmf_shots = [stage3_out.noiseless_shot_result["pmf_by_hamming_weight"][k] for k in ks]
    pmf_mc = [stage3_out.classical_mc_result["pmf_by_hamming_weight"][k] for k in ks]
    pmf_noisy = [
        stage3_out.noise_experiments["combined_depolarizing_and_readout"]["pmf_by_hamming_weight"][k]
        for k in ks
    ]

    bw = 0.20
    axes[0].bar(ks - 1.5 * bw, pmf_exact, width=bw, label="Exact DP / Ideal SV", color="#1f77b4")
    axes[0].bar(ks - 0.5 * bw, pmf_shots, width=bw, label="Noiseless Shots (100k)", color="#2ca02c")
    axes[0].bar(ks + 0.5 * bw, pmf_mc, width=bw, label="Classical MC (100k)", color="#9467bd")
    axes[0].bar(ks + 1.5 * bw, pmf_noisy, width=bw, label="Noisy Quantum (Depol+RO)", color="#d62728")
    axes[0].axvline(2.5, color="black", linestyle="--", alpha=0.7, label="Target Threshold (k >= 3)")
    axes[0].set_xticks(ks)
    axes[0].set_xlabel("Number of Subscribing Customers k (out of 6)")
    axes[0].set_ylabel("Probability P(Sum = k)")
    axes[0].set_title("Stage 3: Campaign Conversion Distribution (PMF)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    methods = [
        "Exact DP",
        "Ideal SV",
        "Noiseless\nShots",
        "Classical\nMC",
        "Gate Depol\nOnly",
        "Readout Err\nOnly",
        "Combined\nNoise",
    ]
    vals = [
        stage3_out.exact_dp_result["prob_at_least_3"],
        stage3_out.ideal_statevector_result["prob_at_least_3"],
        stage3_out.noiseless_shot_result["estimated_prob_at_least_3"],
        stage3_out.classical_mc_result["estimated_prob_at_least_3"],
        stage3_out.noise_experiments["gate_depolarizing_only"]["estimated_prob_at_least_3"],
        stage3_out.noise_experiments["readout_error_only"]["estimated_prob_at_least_3"],
        stage3_out.noise_experiments["combined_depolarizing_and_readout"]["estimated_prob_at_least_3"],
    ]
    colors_m = ["#1f77b4", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#e377c2", "#d62728"]
    bars = axes[1].bar(np.arange(len(methods)), vals, color=colors_m, width=0.58)
    axes[1].axhline(
        stage3_out.exact_dp_result["prob_at_least_3"],
        color="black",
        linestyle="--",
        alpha=0.7,
        label="Exact Reference",
    )
    for bar, v in zip(bars, vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 0.002,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_xticks(np.arange(len(methods)))
    axes[1].set_xticklabels(methods, fontsize=8.5)
    axes[1].set_ylabel("P(At Least 3 of 6 Subscribe)")
    axes[1].set_title("Stage 3: Target Conversion Probability & Noise Shift")
    axes[1].set_ylim(0.0, max(vals) * 1.18 + 0.01)
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    p3 = out_dir / "stage3_conversion_and_noise_analysis.png"
    fig.savefig(p3, dpi=180)
    plt.close(fig)
    saved_paths.append(str(p3))

    return saved_paths
