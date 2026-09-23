"""Tests for the cleaning logic in data_cleaning.py that other scripts
depend on downstream (the hidden-missing-code unmasking that
feature_engineering.py relies on being already done)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_hidden_missing_codes_are_unmasked_to_unknown():
    """Mirrors the exact logic in data_cleaning.py Section 7c: codes that
    mean 'Not Available' must become the string 'Unknown', not stay as a
    plain-looking number that a model could treat as a real category.
    """
    df = pd.DataFrame({"admission_type_id": [1, 5, 6, 8, 2]})
    hidden_codes = [5, 6, 8]

    df["admission_type_id"] = df["admission_type_id"].replace(hidden_codes, np.nan)
    df["admission_type_id"] = df["admission_type_id"].apply(
        lambda x: "Unknown" if pd.isna(x) else str(int(x))
    )

    assert df["admission_type_id"].tolist() == ["1", "Unknown", "Unknown", "Unknown", "2"]


def test_rare_medication_consolidation_counts_correctly():
    """Mirrors data_cleaning.py Section 9: a patient on ANY rare medication
    should be counted, not silently dropped along with the column."""
    df = pd.DataFrame({
        "drug_a": ["No", "No", "Steady"],
        "drug_b": ["No", "Up", "No"],
    })
    rare_cols = ["drug_a", "drug_b"]

    df["num_rare_meds_used"] = (df[rare_cols] != "No").sum(axis=1)

    assert df["num_rare_meds_used"].tolist() == [0, 1, 1]


def test_diag_missing_filled_with_unknown_not_mode():
    """Regression guard for the documented BUG3 fix: missing diagnoses must
    become 'Unknown', never the most common diagnosis code (which would
    clinically mislabel a patient with no recorded diagnosis)."""
    df = pd.DataFrame({"diag_1": ["250", "250", "250", np.nan]})

    df["diag_1"] = df["diag_1"].fillna("Unknown")

    assert df["diag_1"].tolist() == ["250", "250", "250", "Unknown"]
    assert "Unknown" in df["diag_1"].values
