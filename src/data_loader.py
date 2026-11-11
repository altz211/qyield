"""Data loading, schema verification, and hard data boundary enforcement.

This module loads `bank_campaign_participant_release.xlsx` (from `data/` or root),
validates all sheets, documents the 400-row vs. 100-row Stage2_Data discrepancy between
the workbook and PDF specification, and enforces strict separation between Stage1_Train
(300 rows) and Stage1_Test (100 rows).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_FEATURE_COLUMNS: list[str] = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

NUMERIC_FEATURES: list[str] = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_FEATURES: list[str] = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

TARGET_COLUMN: str = "y"

FORBIDDEN_STAGE1_COLUMNS: set[str] = {
    "customer_id",
    "y",
    "duration",
    "customer_segment",
    "contact_cost",
    "expected_deposit_value",
    "capacity_units",
}


@dataclass(frozen=True)
class Stage2Constraints:
    """Operational constraints for Stage 2 customer selection."""

    candidate_count_n: int
    exact_selection_k: int
    budget_limit_c: float
    capacity_limit_h: float


@dataclass(frozen=True)
class CampaignDataset:
    """Verified dataset container with strict train/test separation."""

    readme_df: pd.DataFrame
    stage1_train: pd.DataFrame
    stage1_test: pd.DataFrame
    stage2_data_all: pd.DataFrame
    stage2_data_test_only: pd.DataFrame
    constraints: Stage2Constraints
    audit_summary: dict[str, Any]


def resolve_workbook_path(workbook_path: str | Path | None = None) -> Path:
    """Resolve `bank_campaign_participant_release.xlsx` in `data/` or project root."""
    project_root = Path(__file__).resolve().parent.parent
    if workbook_path is not None:
        candidate = Path(workbook_path)
        if candidate.exists():
            return candidate
        alt = project_root / "data" / candidate.name
        if alt.exists():
            return alt
    for rel in (
        project_root / "data" / "bank_campaign_participant_release.xlsx",
        project_root / "bank_campaign_participant_release.xlsx",
    ):
        if rel.exists():
            return rel
    raise FileNotFoundError(
        f"Workbook not found at {workbook_path} or under {project_root / 'data'}"
    )


def load_and_verify_stage1_train_only(workbook_path: str | Path | None = None) -> pd.DataFrame:
    """Read and validate ONLY the Stage1_Train sheet (300 rows) for `--validate-only` mode.

    Does not read Stage1_Test, Stage2_Data, or Stage2_Constraints.
    """
    path = resolve_workbook_path(workbook_path)
    stage1_train = pd.read_excel(path, sheet_name="Stage1_Train")
    if stage1_train.shape != (300, 17):
        raise ValueError(f"Expected Stage1_Train shape (300, 17), got {stage1_train.shape}")
    if stage1_train.isna().sum().sum() != 0:
        raise ValueError("Stage1_Train contains unexpected null values.")

    expected_train_cols = ["customer_id"] + EXPECTED_FEATURE_COLUMNS + ["y"]
    if list(stage1_train.columns) != expected_train_cols:
        raise ValueError(f"Unexpected Stage1_Train columns: {list(stage1_train.columns)}")
    if "duration" in stage1_train.columns:
        raise ValueError("Forbidden column 'duration' found in Stage1_Train.")
    if len(set(stage1_train["customer_id"])) != 300:
        raise ValueError("Stage1_Train customer_id values are not unique.")

    y_counts = stage1_train["y"].value_counts().to_dict()
    if y_counts.get("yes", 0) != 35 or y_counts.get("no", 0) != 265:
        raise ValueError(f"Unexpected Stage1_Train label counts: {y_counts}")

    return stage1_train


def load_and_verify_workbook(workbook_path: str | Path | None = None) -> CampaignDataset:
    """Load the participant workbook and run strict schema and boundary checks."""
    path = resolve_workbook_path(workbook_path)
    xl = pd.ExcelFile(path)
    required_sheets = {
        "README",
        "Stage1_Train",
        "Stage1_Test",
        "Stage2_Data",
        "Stage2_Constraints",
    }
    missing_sheets = required_sheets - set(xl.sheet_names)
    if missing_sheets:
        raise ValueError(f"Workbook missing required sheets: {missing_sheets}")

    readme_df = pd.read_excel(xl, sheet_name="README")
    stage1_train = pd.read_excel(xl, sheet_name="Stage1_Train")
    stage1_test = pd.read_excel(xl, sheet_name="Stage1_Test")
    stage2_data = pd.read_excel(xl, sheet_name="Stage2_Data")
    constraints_df = pd.read_excel(xl, sheet_name="Stage2_Constraints")

    # 1. Verify shapes and null counts
    if stage1_train.shape != (300, 17):
        raise ValueError(f"Expected Stage1_Train shape (300, 17), got {stage1_train.shape}")
    if stage1_test.shape != (100, 16):
        raise ValueError(f"Expected Stage1_Test shape (100, 16), got {stage1_test.shape}")
    if stage2_data.shape != (400, 5):
        raise ValueError(f"Expected Stage2_Data shape (400, 5), got {stage2_data.shape}")

    if stage1_train.isna().sum().sum() != 0:
        raise ValueError("Stage1_Train contains unexpected null values.")
    if stage1_test.isna().sum().sum() != 0:
        raise ValueError("Stage1_Test contains unexpected null values.")
    if stage2_data.isna().sum().sum() != 0:
        raise ValueError("Stage2_Data contains unexpected null values.")

    # 2. Verify columns and absence of forbidden leakage fields
    expected_train_cols = ["customer_id"] + EXPECTED_FEATURE_COLUMNS + ["y"]
    expected_test_cols = ["customer_id"] + EXPECTED_FEATURE_COLUMNS
    if list(stage1_train.columns) != expected_train_cols:
        raise ValueError(f"Unexpected Stage1_Train columns: {list(stage1_train.columns)}")
    if list(stage1_test.columns) != expected_test_cols:
        raise ValueError(f"Unexpected Stage1_Test columns: {list(stage1_test.columns)}")
    if "y" in stage1_test.columns:
        raise ValueError("Stage1_Test must not contain target column 'y'.")
    if "duration" in stage1_train.columns or "duration" in stage1_test.columns:
        raise ValueError("Forbidden column 'duration' found in data.")

    # 3. Verify ID uniqueness, disjointness, and union coverage
    train_ids = set(stage1_train["customer_id"])
    test_ids = set(stage1_test["customer_id"])
    stage2_ids = set(stage2_data["customer_id"])

    if len(train_ids) != 300:
        raise ValueError("Stage1_Train customer_id values are not unique.")
    if len(test_ids) != 100:
        raise ValueError("Stage1_Test customer_id values are not unique.")
    if len(stage2_ids) != 400:
        raise ValueError("Stage2_Data customer_id values are not unique.")
    if train_ids & test_ids:
        raise ValueError(f"Train and test IDs overlap: {train_ids & test_ids}")
    if (train_ids | test_ids) != stage2_ids:
        raise ValueError("Stage2_Data IDs do not equal the union of Stage1_Train and Stage1_Test IDs.")

    # 4. Verify target distribution on Stage1_Train (35 yes, 265 no)
    y_counts = stage1_train["y"].value_counts().to_dict()
    if y_counts.get("yes", 0) != 35 or y_counts.get("no", 0) != 265:
        raise ValueError(f"Unexpected Stage1_Train label counts: {y_counts}")

    # 5. Restrict Stage2_Data strictly to the 100 Stage1_Test IDs for downstream use
    stage2_test_only = (
        stage2_data[stage2_data["customer_id"].isin(test_ids)]
        .copy()
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    if len(stage2_test_only) != 100:
        raise ValueError("Filtered Stage2_Data for test IDs does not have 100 rows.")

    # 6. Parse Stage2_Constraints sheet
    param_map = dict(zip(constraints_df["symbol"].astype(str).str.strip(), constraints_df["value"]))
    constraints = Stage2Constraints(
        candidate_count_n=int(param_map["N"]),
        exact_selection_k=int(param_map["K"]),
        budget_limit_c=float(param_map["C"]),
        capacity_limit_h=float(param_map["H"]),
    )
    if (
        constraints.candidate_count_n != 10
        or constraints.exact_selection_k != 6
        or constraints.budget_limit_c != 25.0
        or constraints.capacity_limit_h != 22.0
    ):
        raise ValueError(f"Unexpected Stage 2 constraints parsed: {constraints}")

    audit_summary = {
        "workbook_path": str(path.resolve()),
        "sheets": xl.sheet_names,
        "stage1_train_shape": list(stage1_train.shape),
        "stage1_test_shape": list(stage1_test.shape),
        "stage2_data_all_shape": list(stage2_data.shape),
        "stage2_data_test_filtered_shape": list(stage2_test_only.shape),
        "train_target_counts": {"yes": int(y_counts["yes"]), "no": int(y_counts["no"])},
        "train_positive_rate": float(y_counts["yes"] / len(stage1_train)),
        "train_test_id_overlap_count": 0,
        "null_count_total": 0,
        "discrepancy_note": (
            "PDF Section 3 states Stage2_Data has operational inputs for the 100 test customers, "
            "whereas the workbook sheet Stage2_Data contains 400 rows covering the union of "
            "Stage1_Train (300) and Stage1_Test (100). We restrict Stage 2 joins strictly to the "
            "100 Stage1_Test customer_ids and their top-10 ranked candidates."
        ),
        "constraints": {
            "N": constraints.candidate_count_n,
            "K": constraints.exact_selection_k,
            "C": constraints.budget_limit_c,
            "H": constraints.capacity_limit_h,
        },
    }

    return CampaignDataset(
        readme_df=readme_df,
        stage1_train=stage1_train,
        stage1_test=stage1_test,
        stage2_data_all=stage2_data,
        stage2_data_test_only=stage2_test_only,
        constraints=constraints,
        audit_summary=audit_summary,
    )
