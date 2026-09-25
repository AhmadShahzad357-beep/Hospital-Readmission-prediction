from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "reports" / "results"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
DIST_DIR = BASE_DIR / "dashboard" / "dist"

app = FastAPI(title="Hospital Readmission Model -- Results Dashboard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

MODELS_DIR = BASE_DIR / "models"
_model = joblib.load(MODELS_DIR / "lgbm_final.pkl")
_calibrator = joblib.load(MODELS_DIR / "calibrator.pkl")
_thresholds = joblib.load(MODELS_DIR / "decision_threshold.pkl")

from src.inference import get_case_options, build_feature_row


class CaseInput(BaseModel):
    time_in_hospital: float = 3
    num_lab_procedures: float = 40
    num_procedures: float = 1
    num_medications: float = 15
    number_outpatient: float = 0
    number_emergency: float = 0
    number_inpatient: float = 0
    number_diagnoses: float = 7
    num_med_changes: float = 0
    num_meds_active: float = 1
    num_med_increase: float = 0
    num_med_decrease: float = 0
    prior_encounter_count: float = 0
    prior_readmission_count: float = 0
    age_bracket: str = "[60-70)"
    insulin: str = "No"
    gender: str = "Male"
    change: str = "No"
    diabetesMed: str = "Yes"
    diag_1_group: str = "Diabetes"
    diag_2_group: str = "Unknown"
    diag_3_group: str = "Unknown"
    race: str = "Caucasian"
    A1C_category: str = "Unknown"
    glu_serum_category: str = "Unknown"
    admission_type_group: str = "Emergency"
    admission_source_group: str = "Emergency"
    discharge_group: str = "Home"
    medical_specialty_grouped: str = "Unknown"
    payer_code_grouped: str = "Unknown"


@app.get("/api/options")
def options():
    return get_case_options()


@app.post("/api/predict")
def predict(case: CaseInput):
    row = build_feature_row(case.model_dump())
    raw_p = float(_model.predict_proba(row.values)[:, 1][0])
    p_clip = np.clip(raw_p, 1e-6, 1 - 1e-6)
    logit_p = np.log(p_clip / (1 - p_clip))
    calibrated_p = float(_calibrator.predict_proba(np.array([[logit_p]]))[:, 1][0])
    threshold = _thresholds["threshold"]
    return {
        "raw_probability": raw_p,
        "calibrated_probability": calibrated_p,
        "threshold": threshold,
        "decision": "likely readmitted within 30 days" if calibrated_p >= threshold else "likely not readmitted within 30 days",
        "cost_sensitive_threshold": _thresholds.get("cost_sensitive_threshold"),
    }


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_csv_records(path: Path):
    if not path.exists():
        return None
    return json.loads(pd.read_csv(path).to_json(orient="records"))


@app.get("/api/summary")
def summary():
    lgbm = _read_json(RESULTS_DIR / "lgbm_results.json")
    test_final = _read_json(RESULTS_DIR / "test_metrics_final.json")
    ece = _read_json(RESULTS_DIR / "calibration_ece_summary.json")
    if lgbm is None:
        raise HTTPException(404, "lgbm_results.json not found -- run model_training.py first.")
    return {
        "baseline_cv": lgbm.get("baseline_cv"),
        "tuned_cv": lgbm.get("tuned_cv"),
        "oof_roc_auc": lgbm.get("oof_roc_auc"),
        "oof_pr_auc": lgbm.get("oof_pr_auc"),
        "best_threshold": lgbm.get("best_threshold"),
        "cost_sensitive_threshold": lgbm.get("cost_sensitive_threshold"),
        "fn_cost": lgbm.get("fn_cost"),
        "fp_cost": lgbm.get("fp_cost"),
        "test_metrics_bt": lgbm.get("test_metrics_bt"),
        "test_metrics_cost": lgbm.get("test_metrics_cost"),
        "test_metrics_final": test_final,
        "ece": ece,
        "best_params": lgbm.get("best_params"),
        "n_trials": lgbm.get("n_trials"),
    }


@app.get("/api/fairness")
def fairness():
    data = _read_csv_records(RESULTS_DIR / "addition6_fairness_audit.csv")
    if data is None:
        raise HTTPException(404, "addition6_fairness_audit.csv not found -- run fairness_audit.py first.")
    return {"rows": data}


@app.get("/api/risk-stratification")
def risk_strat():
    data = _read_csv_records(RESULTS_DIR / "addition7_risk_stratification.csv")
    if data is None:
        raise HTTPException(404, "addition7_risk_stratification.csv not found -- run risk_stratification.py first.")
    return {"rows": data}


@app.get("/api/cost-benefit")
def cost_benefit():
    data = _read_csv_records(RESULTS_DIR / "addition8_cost_benefit_simulation.csv")
    if data is None:
        raise HTTPException(404, "addition8_cost_benefit_simulation.csv not found -- run cost_benefit_simulation.py first.")
    return {"rows": data}


@app.get("/api/xgboost-comparison")
def xgboost_comparison():
    data = _read_csv_records(RESULTS_DIR / "addition5_xgboost_comparison.csv")
    if data is None:
        raise HTTPException(404, "addition5_xgboost_comparison.csv not found -- run model_comparison_xgboost.py first.")
    return {"rows": data}


@app.get("/api/shap-importance")
def shap_importance():
    data = _read_csv_records(RESULTS_DIR / "shap_importance.csv")
    if data is None:
        raise HTTPException(404, "shap_importance.csv not found -- run model_evaluation.py first.")
    return {"rows": data[:20]}


DASHBOARD_MISSING_HTML = "<h3>API is running.</h3><p>Dashboard not found at dashboard/dist/index.html. Open <a href='/docs'>/docs</a> to try the API directly.</p>"


@app.get("/", response_class=HTMLResponse)
def root():
    index_path = DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return DASHBOARD_MISSING_HTML


if __name__ == "__main__":
    import argparse
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    print(f"Open http://{a.host}:{a.port}   (API docs: http://{a.host}:{a.port}/docs)")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
