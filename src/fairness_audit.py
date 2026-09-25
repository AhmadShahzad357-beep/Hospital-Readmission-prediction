"""ADDITION 6 -- Fairness audit: selection rate and recall by gender and
race, using the same patient-grouped split as feature_engineering.py
(reproduced here, not re-run from scratch) so raw gender/race labels --
dropped during feature selection because their model-signal was weak --
can still be matched to the correct test rows for this audit.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "reports" / "results"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

RANDOM_STATE = 42  # must match feature_engineering.py's Stage B split exactly

# Reproduce the same patient-grouped train/test split used in
# feature_engineering.py, to recover raw gender/race for the test rows.
cleaned = pd.read_csv(PROCESSED_DIR / "diabetic_data_cleaned.csv", low_memory=False)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
_, test_idx = next(gss.split(cleaned, groups=cleaned["patient_nbr"]))
raw_test = cleaned.iloc[test_idx].reset_index(drop=True)

X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

if len(raw_test) != len(X_test):
    raise ValueError(
        f"Row count mismatch: reproduced split has {len(raw_test)} rows, "
        f"X_test.csv has {len(X_test)} rows -- the split parameters "
        f"(random_state, test_size, groups) no longer match feature_engineering.py."
    )
print(f"Reproduced split matches X_test.csv row count: {len(X_test)} rows -- gender/race attached correctly.")

model = joblib.load(MODELS_DIR / "lgbm_final.pkl")
calibrator = joblib.load(MODELS_DIR / "calibrator.pkl")
thresholds = joblib.load(MODELS_DIR / "decision_threshold.pkl")
THRESHOLD = thresholds["threshold"]

raw_p = model.predict_proba(X_test.values)[:, 1]
p_clip = np.clip(raw_p, 1e-6, 1 - 1e-6)
logit_p = np.log(p_clip / (1 - p_clip))
calibrated_p = calibrator.predict_proba(logit_p.reshape(-1, 1))[:, 1]
pred = (calibrated_p >= THRESHOLD).astype(int)

def group_metrics(mask, label):
    n = mask.sum()
    if n == 0:
        return None
    selection_rate = pred[mask].mean()
    actual_rate = y_test[mask].mean()
    tp = ((pred[mask] == 1) & (y_test[mask] == 1)).sum()
    actual_pos = (y_test[mask] == 1).sum()
    recall = tp / actual_pos if actual_pos > 0 else float("nan")
    return {"group": label, "n": int(n), "selection_rate": selection_rate,
            "actual_readmit_rate": actual_rate, "recall": recall}

rows = []
for g in sorted(raw_test["gender"].dropna().unique()):
    rows.append(group_metrics((raw_test["gender"] == g).values, f"gender={g}"))

for r in sorted(raw_test["race"].dropna().unique()):
    rows.append(group_metrics((raw_test["race"] == r).values, f"race={r}"))

fairness_df = pd.DataFrame([r for r in rows if r is not None])
print(fairness_df.to_string(index=False))

max_gap = fairness_df["selection_rate"].max() - fairness_df["selection_rate"].min()
print(f"\nSelection-rate gap across groups: {max_gap:.3f}")
print("This is a screening-level check (selection rate and recall parity), not a full "
      "fairness certification. A gap here does not by itself prove or disprove bias -- it "
      "flags where a deeper review is warranted before any operational use.")

fairness_df.to_csv(RESULTS_DIR / "addition6_fairness_audit.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(fairness_df["group"], fairness_df["selection_rate"], color="#2E5EAA", label="Selection rate")
ax.bar(fairness_df["group"], fairness_df["actual_readmit_rate"], color="#D98E04", alpha=0.5, label="Actual readmit rate")
ax.set_ylabel("Rate")
ax.set_title("ADDITION 6 -- Fairness Screen: Selection Rate vs Actual Rate by Group")
ax.legend()
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "addition6_fairness_audit.png", dpi=200)
plt.close(fig)
print(f"Saved: {RESULTS_DIR / 'addition6_fairness_audit.csv'}")
print(f"Saved: {FIGURES_DIR / 'addition6_fairness_audit.png'}")
