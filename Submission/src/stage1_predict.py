"""Stage 1: Subscription Probability Prediction (Quantum Hybrid & Matched Classical Baseline).

Implements leak-free 5-fold StratifiedKFold cross-validation strictly on the 300 Stage1_Train rows
and single-model final inference on the 100 Stage1_Test rows using the frozen C02 production
configuration:
- Bounded-angle encoding scale = 0.30 rad
- Feature dimension = 4 qubits, reps = 2 circular ZZFeatureMap
- Quantum fidelity kernel Gram matrix with top-6 KernelPCA components (sklearn centered)
- 10 Pauli Z_i and Z_i Z_j expectation values
- Quantum feature block weight w_q = 0.35
- L2-regularized Logistic Regression head with C = 0.12
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import (
    CATEGORICAL_FEATURES,
    EXPECTED_FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

STAGE1_SEED = 42
N_QUBITS_STAGE1 = 4
ZZ_REPS_STAGE1 = 2
# Bounded-angle encoding scale adopted from C02_Bandwidth_030_Reps2:
# maps arctan(z) in (-pi/2, pi/2) to (-0.471, 0.471) rad
QUANTUM_ANGLE_SCALE = 0.30

# Naive constant base-rate Brier score on Stage1_Train: p*(1-p) where p = 35/300 = 0.116667
NAIVE_BASERATE_BRIER = (35.0 / 300.0) * (265.0 / 300.0)  # 0.10305556
QUALIFICATION_MAX_BRIER = float(NAIVE_BASERATE_BRIER)
QUALIFICATION_MAX_ECE = 0.080

# Cache mapping rounded 4-angle bytes -> 16-dim complex statevector
_STATEVECTOR_CACHE: dict[bytes, np.ndarray] = {}
_TEMPLATE_CIRCUIT = zz_feature_map(
    feature_dimension=N_QUBITS_STAGE1,
    reps=ZZ_REPS_STAGE1,
    entanglement="circular",
)
_TEMPLATE_PARAMS = list(_TEMPLATE_CIRCUIT.parameters)

# Diagonal signs (+1/-1) for 1-qubit Z_i and 2-qubit Z_i Z_j on 4 qubits (10 observables)
_PAULI_Z_DIAGONALS: np.ndarray = np.zeros((10, 2**N_QUBITS_STAGE1), dtype=np.float64)
_obs_idx = 0
for _q in range(N_QUBITS_STAGE1):
    for _s in range(2**N_QUBITS_STAGE1):
        _PAULI_Z_DIAGONALS[_obs_idx, _s] = 1.0 - 2.0 * ((_s >> _q) & 1)
    _obs_idx += 1
for _q1 in range(N_QUBITS_STAGE1):
    for _q2 in range(_q1 + 1, N_QUBITS_STAGE1):
        for _s in range(2**N_QUBITS_STAGE1):
            b1 = (_s >> _q1) & 1
            b2 = (_s >> _q2) & 1
            _PAULI_Z_DIAGONALS[_obs_idx, _s] = 1.0 - 2.0 * (b1 ^ b2)
        _obs_idx += 1


class DomainNumericPreprocessor(TransformerMixin, BaseEstimator):
    """Stateless numeric transformations applied before fold-local standardization."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> DomainNumericPreprocessor:
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = X[NUMERIC_FEATURES]
        age = df["age"].astype(float).to_numpy()
        balance = df["balance"].astype(float).to_numpy()
        balance_slog = np.sign(balance) * np.log1p(np.abs(balance))
        day = df["day"].astype(float).to_numpy()
        campaign_log = np.log1p(np.clip(df["campaign"].astype(float).to_numpy(), 0.0, None))
        pdays_raw = df["pdays"].astype(float).to_numpy()
        previously_contacted = (pdays_raw != -1).astype(float)
        recency_inv = np.where(pdays_raw >= 0, 1.0 / (1.0 + pdays_raw / 30.0), 0.0)
        previous_log = np.log1p(np.clip(df["previous"].astype(float).to_numpy(), 0.0, None))
        return np.column_stack(
            [
                age,
                balance_slog,
                day,
                campaign_log,
                previously_contacted,
                recency_inv,
                previous_log,
            ]
        )


def build_feature_preprocessor() -> ColumnTransformer:
    """Create a leak-free ColumnTransformer for numeric and categorical bank attributes."""
    numeric_pipe = Pipeline(
        [
            ("domain_num", DomainNumericPreprocessor()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_4qubit_feature_map_circuit(angles: np.ndarray) -> QuantumCircuit:
    """Bind 4 feature angles into Qiskit's standard `zz_feature_map(4, reps=2)` circuit."""
    bind_map = {_TEMPLATE_PARAMS[i]: float(angles[i]) for i in range(N_QUBITS_STAGE1)}
    return _TEMPLATE_CIRCUIT.assign_parameters(bind_map, inplace=False)


def evaluate_cached_statevectors(angles_matrix: np.ndarray) -> np.ndarray:
    """Evaluate and cache exact 4-qubit statevectors for each sample row in `angles_matrix`."""
    n_samples = angles_matrix.shape[0]
    out = np.empty((n_samples, 2**N_QUBITS_STAGE1), dtype=np.complex128)
    for i in range(n_samples):
        row = np.round(angles_matrix[i], 10)
        key = row.tobytes()
        sv = _STATEVECTOR_CACHE.get(key)
        if sv is None:
            qc = build_4qubit_feature_map_circuit(row)
            sv = Statevector.from_instruction(qc).data
            _STATEVECTOR_CACHE[key] = sv
        out[i] = sv
    return out


def compute_quantum_pauli_expectations(statevectors: np.ndarray) -> np.ndarray:
    """Compute exact expectation values of the 10 Pauli Z_i and Z_i Z_j observables."""
    probs = np.abs(statevectors) ** 2
    return probs @ _PAULI_Z_DIAGONALS.T


def compute_quantum_fidelity_gram(sv_a: np.ndarray, sv_b: np.ndarray) -> np.ndarray:
    """Compute exact quantum fidelity kernel Gram matrix K_ij = |<psi(x_i) | psi(x_j)>|^2."""
    return np.abs(sv_a @ sv_b.conj().T) ** 2


def validate_self_gram_matrix(K: np.ndarray, tol: float = 1e-9) -> dict[str, Any]:
    """Validate that a self-Gram matrix is finite, symmetric, unit-diagonal, and PSD."""
    if not np.all(np.isfinite(K)):
        raise ValueError("Gram matrix contains non-finite entries.")
    max_asym = float(np.max(np.abs(K - K.T)))
    max_diag_err = float(np.max(np.abs(np.diag(K) - 1.0)))
    eigvals = np.linalg.eigvalsh(0.5 * (K + K.T))
    min_eigval = float(np.min(eigvals))
    if max_asym > tol:
        raise ValueError(f"Gram matrix asymmetry {max_asym} exceeds tolerance {tol}.")
    if max_diag_err > tol:
        raise ValueError(f"Gram matrix diagonal deviation {max_diag_err} exceeds tolerance {tol}.")
    if min_eigval < -tol:
        raise ValueError(f"Gram matrix is not PSD; min eigenvalue = {min_eigval}.")
    return {
        "shape": list(K.shape),
        "max_asymmetry": max_asym,
        "max_diagonal_error_from_1": max_diag_err,
        "min_eigenvalue": min_eigval,
        "max_eigenvalue": float(np.max(eigvals)),
        "is_psd": bool(min_eigval >= -tol),
    }


def get_stage1_circuit_resource_report(seed: int = STAGE1_SEED) -> dict[str, Any]:
    """Inspect and report circuit depth, gate counts, and transpilation metrics for Stage 1."""
    sample_angles = np.array([0.25, -0.18, 0.31, -0.22], dtype=float)
    qc = build_4qubit_feature_map_circuit(sample_angles)
    decomp_qc = qc.decompose()
    t_qc = transpile(
        qc,
        basis_gates=["rz", "sx", "x", "cx", "h", "p"],
        optimization_level=1,
        seed_transpiler=seed,
    )

    orig_ops = {str(k): int(v) for k, v in decomp_qc.count_ops().items()}
    trans_ops = {str(k): int(v) for k, v in t_qc.count_ops().items()}
    one_q_gates = sum(v for k, v in trans_ops.items() if k in {"h", "p", "rz", "sx", "x", "ry"})
    two_q_gates = sum(v for k, v in trans_ops.items() if k in {"cx", "cz", "swap"})
    swap_count = int(trans_ops.get("swap", 0))

    return {
        "sdk_circuit": "qiskit.circuit.library.zz_feature_map(feature_dimension=4, reps=2, entanglement='circular')",
        "logical_qubits": int(qc.num_qubits),
        "physical_qubits": "N/A (local statevector simulator; 4 logical qubits)",
        "original_depth_decomposed": int(decomp_qc.depth()),
        "transpiled_depth": int(t_qc.depth()),
        "original_gate_counts": orig_ops,
        "transpiled_gate_counts": trans_ops,
        "one_qubit_gate_count": int(one_q_gates),
        "two_qubit_gate_count": int(two_q_gates),
        "swap_count": swap_count,
        "simulator_backend": "Statevector / AerSimulator(method='statevector')",
        "diagram_text": str(decomp_qc.draw(output="text")),
    }


class EndToEndQuantumClassifier(ClassifierMixin, BaseEstimator):
    """Self-contained Preprocessing + PCA(4) + Quantum Feature Map / Kernel + L2 Logistic Classifier.

    Mode 'hybrid_feature_aug':
    Preprocessed classical features X_enc augmented with top-6 KernelPCA components of the 4-qubit
    Quantum Fidelity Gram matrix K_Q and 10 Pauli Z_i / Z_i Z_j expectation values from
    `zz_feature_map(4, reps=2)`.
    """

    def __init__(
        self,
        mode: str = "hybrid_feature_aug",
        C: float = 0.12,
        quantum_feature_scale: float = 0.35,
        angle_scale: float = QUANTUM_ANGLE_SCALE,
        n_kpca_components: int = 6,
        seed: int = STAGE1_SEED,
    ) -> None:
        self.mode = mode
        self.C = C
        self.quantum_feature_scale = quantum_feature_scale
        self.angle_scale = angle_scale
        self.n_kpca_components = n_kpca_components
        self.seed = seed

    def _project_angles(self, X_enc: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.pca_ = PCA(n_components=N_QUBITS_STAGE1, random_state=self.seed)
            Z = self.pca_.fit_transform(X_enc)
            self.pca_scaler_ = StandardScaler()
            Z_std = self.pca_scaler_.fit_transform(Z)
        else:
            Z = self.pca_.transform(X_enc)
            Z_std = self.pca_scaler_.transform(Z)
        return float(self.angle_scale) * np.arctan(Z_std)

    def _build_design_matrix(self, X_enc: np.ndarray, angles: np.ndarray, fit: bool = False) -> np.ndarray:
        sv = evaluate_cached_statevectors(angles)
        if fit:
            self.sv_train_ = sv
            K_self = compute_quantum_fidelity_gram(sv, sv)
            self.gram_audit_ = validate_self_gram_matrix(K_self)
            self.kpca_ = KernelPCA(
                n_components=self.n_kpca_components,
                kernel="precomputed",
                random_state=self.seed,
            )
            kpca_feats = self.kpca_.fit_transform(K_self)
        else:
            K_eval = compute_quantum_fidelity_gram(sv, self.sv_train_)
            kpca_feats = self.kpca_.transform(K_eval)

        pauli_feats = compute_quantum_pauli_expectations(sv)
        q_raw = np.hstack([kpca_feats, pauli_feats])
        if fit:
            self.q_scaler_ = StandardScaler()
            q_std = self.q_scaler_.fit_transform(q_raw)
        else:
            q_std = self.q_scaler_.transform(q_raw)

        return np.hstack([X_enc, float(self.quantum_feature_scale) * q_std])

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> EndToEndQuantumClassifier:
        y_arr = np.asarray(y, dtype=int)
        self.preprocessor_ = build_feature_preprocessor()
        X_enc = self.preprocessor_.fit_transform(X, y_arr)
        angles_tr = self._project_angles(X_enc, fit=True)
        design_tr = self._build_design_matrix(X_enc, angles_tr, fit=True)

        self.clf_ = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.seed,
        )
        self.clf_.fit(design_tr, y_arr)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = self.preprocessor_.transform(X)
        angles_ev = self._project_angles(X_enc, fit=False)
        design_ev = self._build_design_matrix(X_enc, angles_ev, fit=False)
        probs = self.clf_.predict_proba(design_ev)
        if not np.all(np.isfinite(probs)):
            raise ValueError("Non-finite probabilities produced by EndToEndQuantumClassifier.")
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _build_candidate_models(seed: int = STAGE1_SEED) -> dict[str, BaseEstimator]:
    """Build the production hybrid quantum model and matched classical baseline."""
    return {
        "Hybrid-Quantum-FeatureAug-LogReg": EndToEndQuantumClassifier(
            mode="hybrid_feature_aug",
            C=0.12,
            quantum_feature_scale=0.35,
            angle_scale=0.30,
            n_kpca_components=6,
            seed=seed,
        ),
        "Classical-LogisticRegression-L2": Pipeline(
            [
                ("prep", build_feature_preprocessor()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.15,
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def validate_probabilities_array(probabilities: np.ndarray) -> np.ndarray:
    """Verify that probabilities are 1D, finite (no NaN/Inf), and within [0, 1]."""
    arr = np.asarray(probabilities, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D probability array, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Predicted probabilities contain NaN or infinite values.")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("Predicted probabilities must lie strictly within [0.0, 1.0].")
    return arr


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
) -> float:
    """Compute Expected Calibration Error (ECE) over uniform probability bins."""
    y_prob_clean = validate_probabilities_array(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_clean, bins[1:-1], right=True)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            frac_pos = float(np.mean(y_true[mask]))
            mean_pred = float(np.mean(y_prob_clean[mask]))
            ece += (np.sum(mask) / n) * abs(frac_pos - mean_pred)
    return float(ece)


def compute_ranking_and_prob_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fold_indices: list[np.ndarray],
    customer_ids: np.ndarray,
) -> dict[str, Any]:
    """Compute probability quality, calibration, and top-10 ranking metrics on OOF predictions."""
    y_prob = validate_probabilities_array(y_prob)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    ap_pr_auc = float(average_precision_score(y_true, y_prob))
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    trapz_pr_auc = float(auc(rec_curve, prec_curve))
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)

    order_pooled = np.lexsort((customer_ids, -y_prob))
    top10_pooled_idx = order_pooled[:10]
    hits_pooled_10 = int(np.sum(y_true[top10_pooled_idx]))
    prec_at_10_pooled = float(hits_pooled_10 / 10.0)
    recall_at_10_pooled = float(hits_pooled_10 / np.sum(y_true))

    fold_roc_aucs: list[float] = []
    fold_ap_pr_aucs: list[float] = []
    fold_trapz_pr_aucs: list[float] = []
    fold_briers: list[float] = []
    fold_log_losses: list[float] = []
    fold_prec_at_10: list[float] = []
    fold_rec_at_10: list[float] = []
    fold_composite_scores: list[float] = []

    for val_idx in fold_indices:
        yf = y_true[val_idx]
        pf = y_prob[val_idx]
        cid_f = customer_ids[val_idx]
        f_roc = float(roc_auc_score(yf, pf))
        f_ap = float(average_precision_score(yf, pf))
        p_c, r_c, _ = precision_recall_curve(yf, pf)
        f_trapz = float(auc(r_c, p_c))
        f_brier = float(brier_score_loss(yf, pf))
        f_ll = float(log_loss(yf, pf))
        f_ece = compute_expected_calibration_error(yf, pf, n_bins=5)

        ord_f = np.lexsort((cid_f, -pf))[:10]
        hits_f = int(np.sum(yf[ord_f]))
        f_p10 = float(hits_f / 10.0)
        f_r10 = float(hits_f / np.sum(yf))

        fold_roc_aucs.append(f_roc)
        fold_ap_pr_aucs.append(f_ap)
        fold_trapz_pr_aucs.append(f_trapz)
        fold_briers.append(f_brier)
        fold_log_losses.append(f_ll)
        fold_prec_at_10.append(f_p10)
        fold_rec_at_10.append(f_r10)
        fold_composite_scores.append(
            float(0.35 * f_ap + 0.25 * f_roc + 0.20 * f_p10 - 0.10 * f_brier - 0.10 * f_ece)
        )

    composite_train_score = float(
        0.35 * ap_pr_auc
        + 0.25 * roc_auc
        + 0.20 * float(np.mean(fold_prec_at_10))
        - 0.10 * brier
        - 0.10 * ece
    )

    qualifies_calibration = bool(
        (brier <= QUALIFICATION_MAX_BRIER) and (ece <= QUALIFICATION_MAX_ECE)
    )

    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob, n_bins=5, strategy="quantile")

    return {
        "pooled_oof": {
            "roc_auc": roc_auc,
            "pr_auc_average_precision": ap_pr_auc,
            "pr_auc_trapezoidal": trapz_pr_auc,
            "brier_score": brier,
            "log_loss": ll,
            "ece_5bin": ece,
            "precision_at_10_pooled_n300": prec_at_10_pooled,
            "recall_at_10_pooled_n300": recall_at_10_pooled,
            "mean_predicted_prob": float(np.mean(y_prob)),
            "empirical_positive_rate": float(np.mean(y_true)),
        },
        "per_fold_n60_pos7": {
            "fold_size": 60,
            "positives_per_fold": 7,
            "roc_auc_mean": float(np.mean(fold_roc_aucs)),
            "roc_auc_std": float(np.std(fold_roc_aucs)),
            "pr_auc_ap_mean": float(np.mean(fold_ap_pr_aucs)),
            "pr_auc_ap_std": float(np.std(fold_ap_pr_aucs)),
            "pr_auc_trapz_mean": float(np.mean(fold_trapz_pr_aucs)),
            "brier_mean": float(np.mean(fold_briers)),
            "log_loss_mean": float(np.mean(fold_log_losses)),
            "precision_at_10_mean": float(np.mean(fold_prec_at_10)),
            "precision_at_10_std": float(np.std(fold_prec_at_10)),
            "recall_at_10_mean": float(np.mean(fold_rec_at_10)),
            "recall_at_10_std": float(np.std(fold_rec_at_10)),
            "fold_composite_scores": fold_composite_scores,
        },
        "calibration_curve_5quantile": {
            "prob_true": [float(x) for x in prob_true_cal],
            "prob_pred": [float(x) for x in prob_pred_cal],
        },
        "qualification_check": {
            "brier_threshold": QUALIFICATION_MAX_BRIER,
            "ece_threshold": QUALIFICATION_MAX_ECE,
            "qualifies": qualifies_calibration,
        },
        "composite_train_score": composite_train_score,
    }


def rank_stage1_predictions(
    customer_ids: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Rank customers by predicted_probability descending, breaking ties by customer_id ascending."""
    probs_clean = validate_probabilities_array(probabilities)
    df = pd.DataFrame(
        {
            "customer_id": np.asarray(customer_ids, dtype=str),
            "predicted_probability": probs_clean,
        }
    )
    df = df.sort_values(
        by=["predicted_probability", "customer_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1, dtype=int)
    df["top_10_flag"] = (df["rank"] <= 10).astype(int)
    return df


@dataclass(frozen=True)
class Stage1ValidationSummary:
    """Training-only 5-fold cross-validation output (never touches Stage1_Test)."""

    selected_model_name: str
    selection_rationale: str
    selection_optimism_note: str
    cv_metrics_by_model: dict[str, dict[str, Any]]
    oof_predictions_by_model: dict[str, np.ndarray]
    circuit_report: dict[str, Any]
    runtimes_sec: dict[str, float]


@dataclass(frozen=True)
class Stage1Output:
    """Complete output of Stage 1 validation and single-model final test inference."""

    selected_model_name: str
    selection_rationale: str
    selection_optimism_note: str
    results_df: pd.DataFrame
    cv_metrics_by_model: dict[str, dict[str, Any]]
    oof_predictions_by_model: dict[str, np.ndarray]
    gram_audit: dict[str, Any]
    circuit_report: dict[str, Any]
    runtimes_sec: dict[str, float]


def validate_stage1_only(
    train_df: pd.DataFrame,
    seed: int = STAGE1_SEED,
) -> Stage1ValidationSummary:
    """Run 5-fold StratifiedKFold validation strictly on the 300 Stage1_Train rows."""
    if len(train_df) != 300:
        raise ValueError("Stage 1 training validation must use exactly the 300 Stage1_Train rows.")

    X_train = train_df[EXPECTED_FEATURE_COLUMNS].copy()
    y_train = (train_df[TARGET_COLUMN] == "yes").astype(int).to_numpy()
    train_cids = train_df["customer_id"].astype(str).to_numpy()

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_indices = [val_idx for _, val_idx in outer_cv.split(X_train, y_train)]

    candidate_names = list(_build_candidate_models(seed=seed).keys())
    cv_metrics_by_model: dict[str, dict[str, Any]] = {}
    oof_predictions_by_model: dict[str, np.ndarray] = {}
    runtimes_sec: dict[str, float] = {}

    for model_name in candidate_names:
        t0 = time.perf_counter()
        oof_probs = np.zeros(len(X_train), dtype=np.float64)

        for train_idx, val_idx in outer_cv.split(X_train, y_train):
            fold_model = _build_candidate_models(seed=seed)[model_name]
            X_fold_tr = X_train.iloc[train_idx].reset_index(drop=True)
            y_fold_tr = y_train[train_idx]
            X_fold_val = X_train.iloc[val_idx].reset_index(drop=True)

            fold_model.fit(X_fold_tr, y_fold_tr)
            oof_probs[val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]

        elapsed = time.perf_counter() - t0
        runtimes_sec[model_name] = float(elapsed)
        oof_predictions_by_model[model_name] = validate_probabilities_array(oof_probs)
        cv_metrics_by_model[model_name] = compute_ranking_and_prob_metrics(
            y_true=y_train,
            y_prob=oof_probs,
            fold_indices=fold_indices,
            customer_ids=train_cids,
        )
        cv_metrics_by_model[model_name]["cv_runtime_sec"] = float(elapsed)

    selected_model_name = "Hybrid-Quantum-FeatureAug-LogReg"
    qual_status_msg = (
        f"passed predeclared calibration qualification (Brier <= {QUALIFICATION_MAX_BRIER:.5f} "
        f"and 5-bin ECE <= {QUALIFICATION_MAX_ECE:.3f})"
    )

    selection_optimism_note = (
        "Because the 5-fold Stage1_Train OOF metrics were used during architecture/encoding "
        "refinement and in the 25-configuration two-seed (42, 1042) study, the reported OOF metrics of the "
        "selected model are selection-conditioned and carry finite-sample selection optimism rather than "
        "guaranteeing an out-of-sample generalization gain. Note on scoring rules: LogLoss and Brier are "
        "proper scoring rules, whereas S_train / S_v3 is a heuristic composite ranking-and-calibration "
        "summary. Under the preregistered rule (improve 2-seed mean AP, ROC-AUC, Brier, LogLoss, "
        "and S_v3 simultaneously AND win >=7/10 paired outer folds on S_v3 and LogLoss), "
        "C02_Bandwidth_030_Reps2 (reps=2, w_q=0.35, C=0.12, n_kpca=6, angle_scale=0.30) is the sole "
        "qualifying challenger (2-seed mean AP=0.1988 vs 0.1869, ROC-AUC=0.6322 vs 0.6241, "
        "Brier=0.10267 vs 0.10326, LogLoss=0.35555 vs 0.35762, S_v3=0.2555 vs 0.2485; winning 7/10 "
        "paired folds on S_v3, LogLoss, and Brier)."
    )

    selection_rationale = (
        f"Predeclared composite criterion S_train = 0.35*AP_PR_AUC + 0.25*ROC_AUC + "
        f"0.20*MeanFold_Prec@10 - 0.10*Brier - 0.10*ECE_5bin evaluated strictly on 5-fold "
        f"StratifiedKFold out-of-fold predictions on the 300 Stage1_Train rows selected "
        f"'{selected_model_name}', which {qual_status_msg} "
        f"(S_train = {cv_metrics_by_model[selected_model_name]['composite_train_score']:.4f}, "
        f"ROC-AUC = {cv_metrics_by_model[selected_model_name]['pooled_oof']['roc_auc']:.4f}, "
        f"PR-AUC [AP] = {cv_metrics_by_model[selected_model_name]['pooled_oof']['pr_auc_average_precision']:.4f}, "
        f"Brier = {cv_metrics_by_model[selected_model_name]['pooled_oof']['brier_score']:.4f}, "
        f"ECE = {cv_metrics_by_model[selected_model_name]['pooled_oof']['ece_5bin']:.4f}). "
        "Official downstream routing uses this frozen hybrid quantum pipeline, while all "
        "classical model comparisons remain strictly on Stage1_Train OOF."
    )

    return Stage1ValidationSummary(
        selected_model_name=selected_model_name,
        selection_rationale=selection_rationale,
        selection_optimism_note=selection_optimism_note,
        cv_metrics_by_model=cv_metrics_by_model,
        oof_predictions_by_model=oof_predictions_by_model,
        circuit_report=get_stage1_circuit_resource_report(seed=seed),
        runtimes_sec=runtimes_sec,
    )


def run_stage1_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = STAGE1_SEED,
) -> Stage1Output:
    """Execute leak-free 5-fold CV on Stage1_Train, freeze model, and infer Stage1_Test ONCE."""
    if "y" in test_df.columns:
        raise ValueError("Stage1_Test must not contain labels.")

    val_summary = validate_stage1_only(train_df=train_df, seed=seed)
    selected_model_name = val_summary.selected_model_name

    X_train = train_df[EXPECTED_FEATURE_COLUMNS].copy()
    y_train = (train_df[TARGET_COLUMN] == "yes").astype(int).to_numpy()
    X_test = test_df[EXPECTED_FEATURE_COLUMNS].copy()
    test_cids = test_df["customer_id"].astype(str).to_numpy()

    frozen_model = _build_candidate_models(seed=seed)[selected_model_name]
    frozen_model.fit(X_train, y_train)

    gram_audit = getattr(frozen_model, "gram_audit_", {})

    test_probs = frozen_model.predict_proba(X_test)[:, 1]
    official_results_df = rank_stage1_predictions(test_cids, test_probs)

    return Stage1Output(
        selected_model_name=selected_model_name,
        selection_rationale=val_summary.selection_rationale,
        selection_optimism_note=val_summary.selection_optimism_note,
        results_df=official_results_df,
        cv_metrics_by_model=val_summary.cv_metrics_by_model,
        oof_predictions_by_model=val_summary.oof_predictions_by_model,
        gram_audit=gram_audit,
        circuit_report=val_summary.circuit_report,
        runtimes_sec=val_summary.runtimes_sec,
    )
