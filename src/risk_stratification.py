"""ADDITION 7 -- Risk stratification: bucket every test patient into
Low / Medium / High readmission risk, so care teams get an actionable
label instead of a raw probability. Cutoffs are relative to the dataset's
overall base rate (not arbitrary fixed numbers), and are a labeled,
adjustable business choice -- not derived from the model itself.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "reports" / "results"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

model = joblib.load(MODELS_DIR / "lgbm_final.pkl")
calibrator = joblib.load(MODELS_DIR / "calibrator.pkl")

raw_p = model.predict_proba(X_test.values)[:, 1]
p_clip = np.clip(raw_p, 1e-6, 1 - 1e-6)
logit_p = np.log(p_clip / (1 - p_clip))
calibrated_p = calibrator.predict_proba(logit_p.reshape(-1, 1))[:, 1]

base_rate = y_test.mean()
LOW_CUT = base_rate / 2
HIGH_CUT = base_rate * 2
print(f"Base readmission rate: {base_rate:.3f}  |  Low cutoff: {LOW_CUT:.3f}  |  High cutoff: {HIGH_CUT:.3f}")

bucket = pd.cut(calibrated_p, bins=[-0.01, LOW_CUT, HIGH_CUT, 1.01], labels=["Low", "Medium", "High"])

summary = pd.DataFrame({"probability": calibrated_p, "actual": y_test.values, "bucket": bucket}) \
    .groupby("bucket", observed=True).agg(
        n=("actual", "size"),
        actual_readmit_rate=("actual", "mean"),
        mean_predicted=("probability", "mean"),
    ).reset_index()
print(summary.to_string(index=False))

summary.to_csv(RESULTS_DIR / "addition7_risk_stratification.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 5))
colors = {"Low": "#2E8B57", "Medium": "#D98E04", "High": "#C0392B"}
ax.bar(summary["bucket"], summary["actual_readmit_rate"], color=[colors[b] for b in summary["bucket"]])
for i, row in summary.iterrows():
    ax.text(i, row["actual_readmit_rate"] + 0.01, f"n={row['n']:,}", ha="center", fontsize=9)
ax.set_ylabel("Actual 30-day readmission rate")
ax.set_title("ADDITION 7 -- Risk Stratification: Low / Medium / High")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "addition7_risk_stratification.png", dpi=200)
plt.close(fig)
print(f"Saved: {RESULTS_DIR / 'addition7_risk_stratification.csv'}")
print(f"Saved: {FIGURES_DIR / 'addition7_risk_stratification.png'}")
