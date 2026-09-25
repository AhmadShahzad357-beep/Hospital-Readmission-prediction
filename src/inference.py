"""Single-case inference: rebuilds the exact feature pipeline used in
feature_engineering.py for ONE new case, using the SAVED artifacts
(scaler.pkl, onehot_encoder.pkl, nominal_cols.pkl, iqr_bounds.pkl,
feature_names.csv) instead of re-fitting anything. No pipeline step here
is fit -- every fitted object is loaded, exactly as saved by
feature_engineering.py, so a live case gets the SAME transform as any
training/test row.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

_scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")
_ohe = joblib.load(PROCESSED_DIR / "onehot_encoder.pkl")
_nominal_cols = joblib.load(PROCESSED_DIR / "nominal_cols.pkl")
_iqr_bounds = joblib.load(PROCESSED_DIR / "iqr_bounds.pkl")
_feature_names = pd.read_csv(PROCESSED_DIR / "feature_names.csv")["feature"].tolist()

# Exact same lists as feature_engineering.py -- kept in sync by hand, not
# re-derived, so a change there must be mirrored here.
NUMERIC_SCALE_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_numeric", "num_med_changes", "num_meds_active", "num_med_increase",
    "num_med_decrease", "total_prior_visits", "utilization_score", "lab_per_day",
    "procedure_per_day", "med_per_diagnosis", "comorbidity_diversity",
    "polypharmacy_level", "prior_encounter_count", "prior_readmission_count",
]

AGE_MAPPING = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}


def get_case_options() -> dict:
    """Exact category values the model was trained on, per nominal column
    (read from the saved OneHotEncoder itself, not guessed).
    """
    options = {}
    for col, cats in zip(_nominal_cols, _ohe.categories_):
        options[col] = sorted(cats.tolist())
    options["age_bracket"] = list(AGE_MAPPING.keys())
    options["insulin"] = ["No", "Steady", "Up", "Down"]
    options["gender"] = ["Male", "Female"]
    options["change"] = ["Ch", "No"]
    options["diabetesMed"] = ["Yes", "No"]
    return options


def build_feature_row(case: dict) -> pd.DataFrame:
    """case: a dict of raw form inputs (see get_case_options() for the
    allowed values of each categorical field). Returns a single-row
    DataFrame with columns in the exact order feature_names.csv expects,
    ready for model.predict_proba().
    """
    row = {}

    # -- direct numeric inputs --
    for k in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
              "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
              "num_med_changes", "num_meds_active", "num_med_increase", "num_med_decrease",
              "prior_encounter_count", "prior_readmission_count"]:
        row[k] = float(case.get(k, 0))

    # -- age --
    row["age_numeric"] = AGE_MAPPING.get(case.get("age_bracket"), 55)

    # -- insulin one-hot + flags --
    insulin = case.get("insulin", "No")
    row["insulin_No"] = int(insulin == "No")
    row["insulin_Steady"] = int(insulin == "Steady")
    row["insulin_Up"] = int(insulin == "Up")
    row["insulin_Down"] = int(insulin == "Down")
    row["insulin_adjusted"] = int(insulin in ("Up", "Down"))
    row["insulin_increased"] = int(insulin == "Up")

    # -- simple binary encodes --
    row["gender_binary"] = int(case.get("gender") == "Male")
    row["change_binary"] = int(case.get("change") == "Ch")
    row["diabetesMed_binary"] = int(case.get("diabetesMed") == "Yes")

    # -- diagnosis-group derived flags --
    d1, d2, d3 = case.get("diag_1_group", "Unknown"), case.get("diag_2_group", "Unknown"), case.get("diag_3_group", "Unknown")
    row["primary_diag_diabetes"] = int(d1 == "Diabetes")
    row["comorbidity_diversity"] = len({d1, d2, d3} - {"Unknown", "Other"})

    # -- polypharmacy level, same bin edges as training --
    nm = row["num_medications"]
    row["polypharmacy_level"] = 0 if nm <= 9 else (1 if nm <= 19 else 2)

    # -- utilisation --
    row["total_prior_visits"] = row["number_outpatient"] + row["number_emergency"] + row["number_inpatient"]
    row["prior_inpatient_flag"] = int(row["number_inpatient"] > 0)
    row["prior_emergency_flag"] = int(row["number_emergency"] > 0)
    row["utilization_score"] = row["number_inpatient"] * 3 + row["number_emergency"] * 2 + row["number_outpatient"] * 1

    # -- clinical composites --
    row["lab_per_day"] = row["num_lab_procedures"] / max(row["time_in_hospital"], 1)
    row["procedure_per_day"] = row["num_procedures"] / max(row["time_in_hospital"], 1)
    row["med_per_diagnosis"] = row["num_medications"] / max(row["number_diagnoses"], 1)
    row["extended_stay"] = int(row["time_in_hospital"] > 7)
    row["high_complexity"] = int((row["num_lab_procedures"] > 40) and (row["number_diagnoses"] > 7))

    # -- patient history flag (derived from the count field) --
    row["prior_readmission_flag"] = int(row["prior_readmission_count"] > 0)

    row_df = pd.DataFrame([row])

    # -- IQR capping, same bounds saved during training --
    for col, bounds in _iqr_bounds.items():
        if col in row_df.columns:
            row_df[col] = row_df[col].clip(lower=bounds["lower"], upper=bounds["upper"])

    # -- nominal columns for OHE, taken directly from the form (already the
    #    grouped/post-processing labels, same values the model was trained on) --
    nominal_row = pd.DataFrame([{col: case.get(col, "Unknown") for col in _nominal_cols}])
    ohe_arr = _ohe.transform(nominal_row)
    ohe_df = pd.DataFrame(ohe_arr, columns=_ohe.get_feature_names_out(_nominal_cols))

    full_row = pd.concat([row_df, ohe_df], axis=1)

    # -- scale the same numeric columns, same fitted scaler --
    present_scale_cols = [c for c in NUMERIC_SCALE_COLS if c in full_row.columns]
    full_row[present_scale_cols] = _scaler.transform(full_row[present_scale_cols])

    # -- select + order exactly as feature_names.csv (drop anything else,
    #    fill any missing expected column with 0 -- should not normally happen) --
    final_row = full_row.reindex(columns=_feature_names, fill_value=0.0)
    return final_row
