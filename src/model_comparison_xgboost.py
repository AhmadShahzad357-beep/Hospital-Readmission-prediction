"""ADDITION 5 -- XGBoost comparison against the tuned LightGBM.
Standalone script: does not touch model_training.py, no retraining risk.
Uses the same StratifiedGroupKFold CV scheme so the comparison is fair.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "reports" / "results"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
groups = pd.read_csv(PROCESSED_DIR / "train_patient_groups.csv").iloc[:, 0].values

pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
scale_pos_weight = neg / pos

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
aucs, praucs = [], []
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train, groups), 1):
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="auc",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    p = model.predict_proba(X_train.iloc[va_idx])[:, 1]
    auc = roc_auc_score(y_train.iloc[va_idx], p)
    prauc = average_precision_score(y_train.iloc[va_idx], p)
    aucs.append(auc); praucs.append(prauc)
    print(f"  Fold {fold}: ROC-AUC={auc:.4f} | PR-AUC={prauc:.4f}")

xgb_auc_mean, xgb_prauc_mean = np.mean(aucs), np.mean(praucs)
print(f"\nXGBoost 5-fold CV: mean ROC-AUC={xgb_auc_mean:.4f} +/- {np.std(aucs):.4f}, "
      f"mean PR-AUC={xgb_prauc_mean:.4f} +/- {np.std(praucs):.4f}")

lgbm_results_path = RESULTS_DIR / "lgbm_results.json"
if lgbm_results_path.exists():
    lgbm_results = json.loads(lgbm_results_path.read_text())
    lgbm_baseline_auc = np.mean(list(lgbm_results["baseline_cv"]["roc_auc"].values()))
    print(f"LightGBM baseline 5-fold CV mean ROC-AUC (for comparison): {lgbm_baseline_auc:.4f}")
    winner = "XGBoost" if xgb_auc_mean > lgbm_baseline_auc else "LightGBM"
    print(f"\nDecision: {winner} has the higher mean CV AUC. LightGBM is kept as the "
          f"production model because it is already tuned, calibrated, and threshold-selected "
          f"end-to-end; if XGBoost's edge holds up under the same tuning budget, that is a "
          f"candidate for future work, not a same-session swap.")
else:
    lgbm_baseline_auc = None
    print("lgbm_results.json not found -- run model_training.py first for a full comparison.")

comparison = pd.DataFrame({
    "model": ["XGBoost (default-tuned)", "LightGBM (baseline CV)"],
    "mean_roc_auc": [xgb_auc_mean, lgbm_baseline_auc],
    "mean_pr_auc": [xgb_prauc_mean, None],
})
comparison.to_csv(RESULTS_DIR / "addition5_xgboost_comparison.csv", index=False)
print(f"\nSaved: {RESULTS_DIR / 'addition5_xgboost_comparison.csv'}")
