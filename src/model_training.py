# ── Imports ───────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import json
import os
import time

from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve,
    f1_score, brier_score_loss, log_loss,
    precision_score, recall_score
)
from sklearn.calibration import calibration_curve

import lightgbm as lgb

# ── Paths ─────────────────────────────────────────────────────────────────────
# Dynamic path: works regardless of where the project folder is located/renamed
BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / 'data/processed'
MODELS_DIR    = BASE_DIR / 'models'
FIGURES_DIR   = BASE_DIR / 'reports/figures'
RESULTS_DIR   = BASE_DIR / 'reports/results'

for d in [MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_CV_FOLDS   = 5

# ── Design tokens ─────────────────────────────────────────────────────────────
PALETTE = {
    'bg'       : '#FFFFFF',
    'panel'    : '#F7FAFC',
    'spine'    : '#D1DCE5',
    'title'    : '#0D1B2A',
    'label'    : '#2C3E50',
    'tick'     : '#4A5568',
    'positive' : '#C0392B',
    'neutral'  : '#2E6EA6',
    'negative' : '#2E8B57',
    'accent'   : '#E67E22',
    'lgbm'     : '#1A7A4A',
    'optuna'   : '#7B3F6E',
}
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : False,
    'savefig.dpi'      : 300,
})

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING PREPROCESSED DATA")
print("=" * 70)

X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
X_test  = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv').squeeze()
y_test  = pd.read_csv(PROCESSED_DIR / 'y_test.csv').squeeze()

print(f"  X_train : {X_train.shape}  |  X_test : {X_test.shape}")
print(f"  y_train : {y_train.shape}  |  y_test : {y_test.shape}")
print(f"  Train positive: {y_train.mean()*100:.2f}%  |  Test positive: {y_test.mean()*100:.2f}%")

# ── Load patient numbers for GroupKFold (must re-read from raw processed data)
# patient_nbr was dropped AFTER split in Code 3, so we reconstruct groups
# from the original raw file using the same GroupShuffleSplit indices.
# Since we don't have indices saved, we use a SAFE fallback:
# StratifiedGroupKFold with synthetic groups (range) → degrades to StratifiedKFold
# NOTE: If you saved patient_nbr indices, load them here instead.
try:
    groups_train = pd.read_csv(PROCESSED_DIR / 'train_patient_groups.csv').squeeze()
    print(f"  Patient groups loaded: {groups_train.nunique()} unique patients")
except FileNotFoundError:
    print("  ⚠️  train_patient_groups.csv not found — using row-index groups")
    print("      (StratifiedGroupKFold will behave like StratifiedKFold)")
    groups_train = pd.Series(range(len(X_train)))

# ── Class imbalance ratio ─────────────────────────────────────────────────────
neg_count        = (y_train == 0).sum()
pos_count        = (y_train == 1).sum()
SCALE_POS_WEIGHT = neg_count / pos_count
print(f"\n  Imbalance ratio (scale_pos_weight): {SCALE_POS_WEIGHT:.4f}")

# ── CV strategy ───────────────────────────────────────────────────────────────
cv = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

X_arr = X_train.values
y_arr = y_train.values
g_arr = groups_train.values

# Materialize the folds once so Stage 1 and Stage 2 use the exact same
# splits (fair comparison) and Stage 2's manual search loop can reuse them
# without recomputing StratifiedGroupKFold on every candidate.
cv_splits = list(cv.split(X_arr, y_arr, g_arr))


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def compute_all_metrics(y_true, y_prob, threshold=0.5):
    """Return dict of all classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'roc_auc'   : roc_auc_score(y_true, y_prob),
        'pr_auc'    : average_precision_score(y_true, y_prob),
        'f1'        : f1_score(y_true, y_pred, zero_division=0),
        'precision' : precision_score(y_true, y_pred, zero_division=0),
        'recall'    : recall_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'brier'     : brier_score_loss(y_true, y_prob),
        'log_loss'  : log_loss(y_true, y_prob),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def find_best_threshold(y_true, y_prob):
    """Find threshold that maximises F1 on validation set."""
    thresholds = np.linspace(0.05, 0.95, 100)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# -- ADDITION 2 FIX: PLATT CALIBRATION -----------------------------------
# The calibration check in model_evaluation.py found ECE=0.35 -- scale_pos_weight
# (used to fight class imbalance) biases raw probabilities upward to help
# ranking (AUC), at the cost of a "60%" prediction actually meaning 60%.
# Platt scaling (a tiny logistic regression on logit(p)) fixes this. It is
# fit ONLY on out-of-fold (OOF) train predictions, never on the test set --
# the same no-leakage rule as the threshold selection right below it.
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt_calibrator(oof_probs, oof_trues):
    from sklearn.linear_model import LogisticRegression
    calibrator = LogisticRegression(C=1e6, max_iter=1000)
    calibrator.fit(_logit(oof_probs).reshape(-1, 1), oof_trues)
    return calibrator


def apply_calibration(p, calibrator):
    return calibrator.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


# -- ADDITION 3: COST-SENSITIVE THRESHOLD --------------------------------
# A missed at-risk patient (false negative) is more costly to a hospital
# than an unnecessary follow-up call (false positive) -- FN_COST/FP_COST
# below encode that a missed readmission is judged 3x as costly as a
# needless call. This is a business judgment call, not something derived
# from the data, so it is a named constant that can be changed and
# re-run rather than a number buried in the threshold-selection logic.
FN_COST = 3.0
FP_COST = 1.0


def find_cost_sensitive_threshold(y_true, y_prob, fn_cost=FN_COST, fp_cost=FP_COST):
    """Threshold that minimises fn_cost*FN + fp_cost*FP, swept the same
    way find_best_threshold() sweeps for F1 -- same OOF-only rule applies.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_t, best_cost = 0.5, float('inf')
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        fn = ((pred == 0) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t, best_cost


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — BASELINE LightGBM  (5-Fold StratifiedGroupKFold CV)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 1 — BASELINE LightGBM  (5-Fold CV with early stopping)")
print("=" * 70)

BASELINE_PARAMS = {
    'objective'       : 'binary',
    # NOTE: 'auc' MUST come first (with first_metric_only=True on the
    # early_stopping callback below). With scale_pos_weight this high,
    # weighted binary_logloss gets worse almost immediately even while AUC
    # keeps improving — if logloss is checked first, early stopping locks
    # the model at iteration ~1 (a near-stump), which is exactly why
    # F1/Precision/Recall were coming out 0.
    'metric'          : ['auc', 'binary_logloss'],
    'boosting_type'   : 'gbdt',
    'n_estimators'    : 2000,
    'learning_rate'   : 0.05,
    'num_leaves'      : 63,
    'max_depth'       : -1,
    'min_child_samples': 50,
    'subsample'       : 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha'       : 0.1,
    'reg_lambda'       : 0.1,
    'scale_pos_weight': SCALE_POS_WEIGHT,
    'random_state'    : RANDOM_STATE,
    'n_jobs'          : -1,
    'verbose'         : -1,
}

baseline_cv_metrics   = []
baseline_fold_models  = []
baseline_train_curves = []   # for learning curve plot
baseline_val_curves   = []
baseline_best_iters   = []

fold_num = 0
for train_idx, val_idx in cv.split(X_arr, y_arr, g_arr):
    fold_num += 1
    X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
    y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

    model = lgb.LGBMClassifier(**BASELINE_PARAMS)

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, first_metric_only=True, verbose=False),
        lgb.log_evaluation(period=-1),
        lgb.record_evaluation(evals_result := {}),
    ]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        eval_names=['train', 'val'],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration_
    baseline_best_iters.append(best_iter)

    # Learning curves
    baseline_train_curves.append(evals_result['train']['binary_logloss'])
    baseline_val_curves.append(evals_result['val']['binary_logloss'])

    y_prob_val = model.predict_proba(X_val)[:, 1]
    metrics    = compute_all_metrics(y_val, y_prob_val)
    metrics['best_iter'] = best_iter
    metrics['fold']      = fold_num

    baseline_cv_metrics.append(metrics)
    baseline_fold_models.append(model)

    print(f"  Fold {fold_num}: ROC-AUC={metrics['roc_auc']:.4f} | "
          f"PR-AUC={metrics['pr_auc']:.4f} | F1={metrics['f1']:.4f} | "
          f"Best iter={best_iter}")

baseline_cv_df = pd.DataFrame(baseline_cv_metrics)
print(f"\n  ─── Baseline CV Summary ───")
for col in ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall', 'brier', 'log_loss']:
    m, s = baseline_cv_df[col].mean(), baseline_cv_df[col].std()
    print(f"  {col:12}: {m:.4f} ± {s:.4f}")
print(f"  Best iters: {baseline_best_iters}  (avg={np.mean(baseline_best_iters):.0f})")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 GRAPHS — Baseline CV
# ══════════════════════════════════════════════════════════════════════════════

# ── Graph 1a: CV Metric Bar Chart ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Stage 1 — Baseline LightGBM: 5-Fold CV Metrics',
             fontsize=14, fontweight='bold', color=PALETTE['title'])

metrics_to_plot = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall', 'specificity']
metric_labels   = ['ROC-AUC', 'PR-AUC', 'F1', 'Precision', 'Recall', 'Specificity']
bar_colors      = [PALETTE['lgbm'], PALETTE['neutral'], PALETTE['accent'],
                   PALETTE['positive'], PALETTE['optuna'], '#5B8DB8']

means = [baseline_cv_df[m].mean() for m in metrics_to_plot]
stds  = [baseline_cv_df[m].std()  for m in metrics_to_plot]

x = np.arange(len(metrics_to_plot))
bars = axes[0].bar(x, means, yerr=stds, width=0.6,
                   color=bar_colors, edgecolor='white', linewidth=0.8,
                   error_kw=dict(ecolor='#888', elinewidth=1.2, capsize=5))
for bar, mean, std in zip(bars, means, stds):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 mean + std + 0.005,
                 f'{mean:.4f}', ha='center', va='bottom',
                 fontsize=8.5, fontweight='bold', color=PALETTE['title'])

axes[0].set_xticks(x)
axes[0].set_xticklabels(metric_labels, rotation=15, ha='right', fontsize=9)
axes[0].set_ylim(0, 1.10)
axes[0].set_ylabel('Score', fontsize=10, color=PALETTE['label'])
axes[0].set_title('CV Metrics (Mean ± Std)', fontsize=11, fontweight='bold',
                  color=PALETTE['title'])
axes[0].spines['left'].set_color(PALETTE['spine'])
axes[0].spines['bottom'].set_color(PALETTE['spine'])
axes[0].tick_params(length=0)

# Per-fold lines
folds_x = np.arange(1, N_CV_FOLDS + 1)
for metric, label, color in zip(
    ['roc_auc', 'pr_auc', 'f1'],
    ['ROC-AUC', 'PR-AUC', 'F1'],
    [PALETTE['lgbm'], PALETTE['neutral'], PALETTE['accent']]
):
    vals = baseline_cv_df[metric].values
    axes[1].plot(folds_x, vals, 'o-', color=color, linewidth=2,
                 markersize=7, label=f'{label} (μ={vals.mean():.4f})')
    axes[1].axhline(vals.mean(), color=color, linewidth=1.0,
                    linestyle='--', alpha=0.5)

axes[1].set_xlabel('Fold', fontsize=10, color=PALETTE['label'])
axes[1].set_ylabel('Score', fontsize=10, color=PALETTE['label'])
axes[1].set_title('Per-Fold Performance', fontsize=11, fontweight='bold',
                  color=PALETTE['title'])
axes[1].set_xticks(folds_x)
axes[1].legend(fontsize=8.5, frameon=False)
axes[1].spines['left'].set_color(PALETTE['spine'])
axes[1].spines['bottom'].set_color(PALETTE['spine'])
axes[1].tick_params(length=0)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGURES_DIR / 'stage1_baseline_cv_metrics.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("\n  Saved: stage1_baseline_cv_metrics.png")

# ── Graph 1b: Learning Curves (avg across folds) ─────────────────────────────
max_len = max(len(c) for c in baseline_train_curves)
train_padded = np.array([np.pad(c, (0, max_len - len(c)), constant_values=c[-1])
                         for c in baseline_train_curves])
val_padded   = np.array([np.pad(c, (0, max_len - len(c)), constant_values=c[-1])
                         for c in baseline_val_curves])

train_mean = train_padded.mean(axis=0)
val_mean   = val_padded.mean(axis=0)
val_std    = val_padded.std(axis=0)
iters      = np.arange(1, max_len + 1)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(iters, train_mean, color=PALETTE['lgbm'], linewidth=2, label='Train Loss (avg)')
ax.plot(iters, val_mean,   color=PALETTE['positive'], linewidth=2, label='Val Loss (avg)')
ax.fill_between(iters, val_mean - val_std, val_mean + val_std,
                color=PALETTE['positive'], alpha=0.15, label='Val ± 1 std')
avg_best = int(np.mean(baseline_best_iters))
ax.axvline(avg_best, color=PALETTE['accent'], linewidth=1.5,
           linestyle='--', label=f'Avg early stop = {avg_best}')
ax.set_xlabel('Boosting Rounds', fontsize=10, color=PALETTE['label'])
ax.set_ylabel('Binary Log-Loss', fontsize=10, color=PALETTE['label'])
ax.set_title('Stage 1 — Baseline LightGBM: Learning Curves (5-Fold Avg)',
             fontsize=12, fontweight='bold', color=PALETTE['title'])
ax.legend(fontsize=9, frameon=False)
ax.spines['left'].set_color(PALETTE['spine'])
ax.spines['bottom'].set_color(PALETTE['spine'])
ax.tick_params(length=0)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'stage1_baseline_learning_curves.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: stage1_baseline_learning_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — RANDOMIZED SEARCH HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 2 — RANDOM SEARCH TUNING  (n_iter=50, 5-Fold StratifiedGroupKFold CV)")
print("=" * 70)

N_ITER = 50   # number of random parameter combinations to try

rng = np.random.RandomState(RANDOM_STATE)

def sample_params():
    return {
        'boosting_type'    : 'gbdt',   # dart/goss hata diye — slow aur underperforming
        'n_estimators'     : int(rng.randint(300, 2001)),
        'learning_rate'    : float(np.exp(rng.uniform(np.log(0.005), np.log(0.2)))),
        'num_leaves'       : int(rng.randint(20, 301)),
        'max_depth'        : int(rng.randint(3, 13)),
        'min_child_samples': int(rng.randint(10, 201)),
        'subsample'        : float(rng.uniform(0.5, 1.0)),
        'colsample_bytree' : float(rng.uniform(0.4, 1.0)),
        'reg_alpha'        : float(np.exp(rng.uniform(np.log(1e-4), np.log(10.0)))),
        'reg_lambda'       : float(np.exp(rng.uniform(np.log(1e-4), np.log(10.0)))),
        'min_split_gain'   : float(rng.uniform(0.0, 1.0)),
    }
# Include the Stage-1 baseline config as combination #0 (warm-start / sanity check)
candidate_list = [{
    'boosting_type': 'gbdt', 'n_estimators': 1000, 'learning_rate': 0.05,
    'num_leaves': 63, 'max_depth': 7, 'min_child_samples': 50,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'min_split_gain': 0.0,
}] + [sample_params() for _ in range(N_ITER - 1)]

search_history = []   # one row per candidate: params + mean_auc + std_auc + time

start_time = time.time()
for i, cand in enumerate(candidate_list):
    t0 = time.time()

    params = {
        'objective'       : 'binary',
        'metric'          : ['auc', 'binary_logloss'],   # auc first! see Stage-1 note above
        'scale_pos_weight': SCALE_POS_WEIGHT,
        'random_state'    : RANDOM_STATE,
        'n_jobs'          : -1,     # single model uses all cores; NOT nested with outer parallelism
        'verbose'         : -1,
        **cand,
    }

    # dart/goss don't support early stopping cleanly — cap their n_estimators
    # so a bad random draw (e.g. dart + 1900 trees) can't stall the whole run.
    use_early_stopping = (params['boosting_type'] == 'gbdt')
    if not use_early_stopping:
        params['n_estimators'] = min(params['n_estimators'], 500)

    fold_aucs = []
    for tr_idx, val_idx in cv_splits:
        X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

        model = lgb.LGBMClassifier(**params)

        if use_early_stopping:
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(stopping_rounds=50,
                                                     first_metric_only=True,
                                                     verbose=False),
                                 lgb.log_evaluation(period=-1)])
        else:
            model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_val)[:, 1]
        fold_aucs.append(roc_auc_score(y_val, y_prob))

    elapsed = time.time() - t0
    mean_auc, std_auc = np.mean(fold_aucs), np.std(fold_aucs)
    row = {'candidate': i, 'mean_test_score': mean_auc, 'std_test_score': std_auc,
           'seconds': elapsed, **{f'param_{k}': v for k, v in cand.items()}}
    search_history.append(row)

    print(f"  [{i+1:>2}/{N_ITER}] AUC={mean_auc:.4f} ± {std_auc:.4f}  "
          f"({elapsed:.1f}s)  boosting={cand['boosting_type']:<5} "
          f"n_est={cand['n_estimators']:<4} lr={cand['learning_rate']:.4f}")

tuning_time = time.time() - start_time

results_cv = pd.DataFrame(search_history)
best_row   = results_cv.loc[results_cv['mean_test_score'].idxmax()]
best_params_raw = {k.replace('param_', ''): best_row[k]
                    for k in results_cv.columns if k.startswith('param_')}

print(f"\n  Tuning complete in {tuning_time/60:.1f} min")
print(f"  Combinations tried : {N_ITER}")
print(f"  Best ROC-AUC        : {best_row['mean_test_score']:.6f}")
print(f"  Best params:")
for k, v in best_params_raw.items():
    print(f"    {k:25}: {v}")

# Save best params
best_params_path = RESULTS_DIR / 'best_lgbm_params.json'
with open(best_params_path, 'w') as f:
    json.dump(best_params_raw, f, indent=2, default=str)
print(f"\n  Best params saved → {best_params_path}")


# ── Stage 2: CV with best params (full eval, with learning curves) ───────────
print(f"\n  Running full 5-Fold CV with best params...")

BEST_PARAMS = {
    'objective'       : 'binary',
    'metric'          : ['auc', 'binary_logloss'],   # auc first — see Stage-1 note
    'n_estimators'    : int(best_params_raw.get('n_estimators', 1000)),
    'learning_rate'   : best_params_raw['learning_rate'],
    'num_leaves'      : int(best_params_raw['num_leaves']),
    'max_depth'       : int(best_params_raw['max_depth']),
    'min_child_samples': int(best_params_raw['min_child_samples']),
    'subsample'       : best_params_raw['subsample'],
    'colsample_bytree': best_params_raw['colsample_bytree'],
    'reg_alpha'       : best_params_raw['reg_alpha'],
    'reg_lambda'       : best_params_raw['reg_lambda'],
    'min_split_gain'  : best_params_raw.get('min_split_gain', 0.0),
    'boosting_type'   : best_params_raw.get('boosting_type', 'gbdt'),
    'scale_pos_weight': SCALE_POS_WEIGHT,
    'random_state'    : RANDOM_STATE,
    'n_jobs'          : -1,
    'verbose'         : -1,
}

tuned_cv_metrics     = []
tuned_train_curves   = []
tuned_val_curves     = []
tuned_best_iters     = []
tuned_oof_probs      = np.zeros(len(X_arr))   # out-of-fold probabilities
tuned_oof_trues      = np.zeros(len(X_arr))

fold_num = 0
for train_idx, val_idx in cv.split(X_arr, y_arr, g_arr):
    fold_num += 1
    X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
    y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

    model = lgb.LGBMClassifier(**BEST_PARAMS)

    use_es = (BEST_PARAMS['boosting_type'] == 'gbdt')
    if use_es:
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=-1),
            lgb.record_evaluation(evals_result := {}),
        ]
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)],
                  eval_names=['train', 'val'],
                  callbacks=callbacks)
        tuned_train_curves.append(evals_result['train']['binary_logloss'])
        tuned_val_curves.append(evals_result['val']['binary_logloss'])
        tuned_best_iters.append(model.best_iteration_)
    else:
        model.fit(X_tr, y_tr)
        tuned_best_iters.append(BEST_PARAMS['n_estimators'])

    y_prob_val = model.predict_proba(X_val)[:, 1]
    tuned_oof_probs[val_idx] = y_prob_val
    tuned_oof_trues[val_idx] = y_val

    metrics = compute_all_metrics(y_val, y_prob_val)
    metrics['best_iter'] = tuned_best_iters[-1]
    metrics['fold']      = fold_num
    tuned_cv_metrics.append(metrics)

    print(f"  Fold {fold_num}: ROC-AUC={metrics['roc_auc']:.4f} | "
          f"PR-AUC={metrics['pr_auc']:.4f} | F1={metrics['f1']:.4f}")

tuned_cv_df = pd.DataFrame(tuned_cv_metrics)
print(f"\n  ─── Tuned CV Summary ───")
for col in ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall', 'brier', 'log_loss']:
    m, s = tuned_cv_df[col].mean(), tuned_cv_df[col].std()
    print(f"  {col:12}: {m:.4f} ± {s:.4f}")

# OOF overall metrics
oof_roc = roc_auc_score(tuned_oof_trues, tuned_oof_probs)
oof_pr  = average_precision_score(tuned_oof_trues, tuned_oof_probs)
print(f"\n  OOF (full train) ROC-AUC : {oof_roc:.4f}")
print(f"  OOF (full train) PR-AUC  : {oof_pr:.4f}")

# ADDITION 2 FIX -- fit Platt calibrator on OOF train predictions only
calibrator = fit_platt_calibrator(tuned_oof_probs, tuned_oof_trues)
tuned_oof_probs_calibrated = apply_calibration(tuned_oof_probs, calibrator)
print(f"  Platt calibrator fitted on {len(tuned_oof_probs):,} OOF train predictions (no test leakage)")

# Best threshold from CALIBRATED OOF (threshold and probabilities must agree on the same scale)
best_thresh, best_oof_f1 = find_best_threshold(tuned_oof_trues, tuned_oof_probs_calibrated)
print(f"  Best threshold (OOF F1, calibrated)  : {best_thresh:.3f}  (F1={best_oof_f1:.4f})")

# ADDITION 3 -- cost-sensitive threshold, same calibrated OOF predictions,
# never the test set
cost_thresh, cost_value = find_cost_sensitive_threshold(tuned_oof_trues, tuned_oof_probs_calibrated)
print(f"  Cost-sensitive threshold (OOF, FN_cost={FN_COST}x FP_cost={FP_COST}x): "
      f"{cost_thresh:.3f}  (weighted cost={cost_value:.0f})")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 GRAPHS — Random Search + Tuned CV
# ══════════════════════════════════════════════════════════════════════════════

# ── Graph 2a: Random Search Optimization History ─────────────────────────────
trial_nums    = np.arange(len(results_cv))
trial_vals    = results_cv['mean_test_score'].values
running_best  = np.maximum.accumulate(trial_vals)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Stage 2 — Random Search Hyperparameter Tuning',
             fontsize=14, fontweight='bold', color=PALETTE['title'])

axes[0].scatter(trial_nums, trial_vals, color=PALETTE['neutral'], s=25,
                alpha=0.6, label='Combination ROC-AUC', zorder=3)
axes[0].plot(trial_nums, running_best, color=PALETTE['positive'], linewidth=2.5,
             label='Running Best', zorder=4)
axes[0].axhline(np.mean(trial_vals),
                color='#888', linewidth=1, linestyle='--', alpha=0.6, label='Mean AUC')
axes[0].set_xlabel('Combination Number', fontsize=10, color=PALETTE['label'])
axes[0].set_ylabel('ROC-AUC (5-Fold CV)', fontsize=10, color=PALETTE['label'])
axes[0].set_title('Optimization History', fontsize=11, fontweight='bold',
                  color=PALETTE['title'])
axes[0].legend(fontsize=9, frameon=False)
axes[0].spines['left'].set_color(PALETTE['spine'])
axes[0].spines['bottom'].set_color(PALETTE['spine'])
axes[0].tick_params(length=0)
axes[0].text(0.98, 0.04, f'Best: {best_row["mean_test_score"]:.4f}',
             ha='right', va='bottom', transform=axes[0].transAxes,
             fontsize=10, fontweight='bold', color=PALETTE['positive'])

# Baseline vs Tuned comparison (bar)
compare_metrics = ['roc_auc', 'pr_auc', 'f1', 'recall', 'precision']
compare_labels  = ['ROC-AUC', 'PR-AUC', 'F1', 'Recall', 'Precision']
base_vals = [baseline_cv_df[m].mean() for m in compare_metrics]
tune_vals = [tuned_cv_df[m].mean()    for m in compare_metrics]
base_stds = [baseline_cv_df[m].std()  for m in compare_metrics]
tune_stds = [tuned_cv_df[m].std()     for m in compare_metrics]

x = np.arange(len(compare_metrics))
w = 0.35
axes[1].bar(x - w/2, base_vals, w, yerr=base_stds, label='Baseline',
            color=PALETTE['neutral'], edgecolor='white', linewidth=0.8,
            error_kw=dict(ecolor='#888', capsize=4))
axes[1].bar(x + w/2, tune_vals, w, yerr=tune_stds, label='Tuned',
            color=PALETTE['lgbm'], edgecolor='white', linewidth=0.8,
            error_kw=dict(ecolor='#888', capsize=4))

for i, (bv, tv) in enumerate(zip(base_vals, tune_vals)):
    diff = tv - bv
    col  = PALETTE['lgbm'] if diff >= 0 else PALETTE['positive']
    axes[1].text(i + w/2, tv + 0.01, f'{diff:+.4f}',
                 ha='center', va='bottom', fontsize=7.5,
                 fontweight='bold', color=col)

axes[1].set_xticks(x)
axes[1].set_xticklabels(compare_labels, rotation=10, ha='right', fontsize=9)
axes[1].set_ylim(0, 1.10)
axes[1].set_ylabel('Score', fontsize=10, color=PALETTE['label'])
axes[1].set_title('Baseline vs Tuned (5-Fold CV)', fontsize=11,
                  fontweight='bold', color=PALETTE['title'])
axes[1].legend(fontsize=9, frameon=False)
axes[1].spines['left'].set_color(PALETTE['spine'])
axes[1].spines['bottom'].set_color(PALETTE['spine'])
axes[1].tick_params(length=0)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGURES_DIR / 'stage2_optuna_history.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n  Saved: stage2_optuna_history.png")

# ── Graph 2b: Tuned Learning Curves ──────────────────────────────────────────
if tuned_train_curves:
    max_len2    = max(len(c) for c in tuned_train_curves)
    tr_padded2  = np.array([np.pad(c, (0, max_len2 - len(c)), constant_values=c[-1])
                             for c in tuned_train_curves])
    val_padded2 = np.array([np.pad(c, (0, max_len2 - len(c)), constant_values=c[-1])
                             for c in tuned_val_curves])

    tr_mean2  = tr_padded2.mean(axis=0)
    val_mean2 = val_padded2.mean(axis=0)
    val_std2  = val_padded2.std(axis=0)
    iters2    = np.arange(1, max_len2 + 1)
    avg_best2 = int(np.mean(tuned_best_iters))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(iters2, tr_mean2,  color=PALETTE['lgbm'],     linewidth=2,
            label='Train Loss (avg)')
    ax.plot(iters2, val_mean2, color=PALETTE['positive'], linewidth=2,
            label='Val Loss (avg)')
    ax.fill_between(iters2, val_mean2 - val_std2, val_mean2 + val_std2,
                    color=PALETTE['positive'], alpha=0.15, label='Val ± 1 std')
    ax.axvline(avg_best2, color=PALETTE['accent'], linewidth=1.5,
               linestyle='--', label=f'Avg early stop = {avg_best2}')
    ax.set_xlabel('Boosting Rounds', fontsize=10, color=PALETTE['label'])
    ax.set_ylabel('Binary Log-Loss', fontsize=10, color=PALETTE['label'])
    ax.set_title('Stage 2 — Tuned LightGBM: Learning Curves (5-Fold Avg)',
                 fontsize=12, fontweight='bold', color=PALETTE['title'])
    ax.legend(fontsize=9, frameon=False)
    ax.spines['left'].set_color(PALETTE['spine'])
    ax.spines['bottom'].set_color(PALETTE['spine'])
    ax.tick_params(length=0)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'stage2_tuned_learning_curves.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: stage2_tuned_learning_curves.png")

# ── Graph 2c: Param Importance (correlation of each param with CV AUC) ───────
# RandomizedSearchCV has no built-in fANOVA importance like Optuna; approximate
# it with |Spearman correlation| between each numeric param's sampled value
# and the resulting mean_test_score across the N_ITER combinations tried.
try:
    from scipy.stats import spearmanr

    numeric_params = ['n_estimators', 'learning_rate', 'num_leaves', 'max_depth',
                       'min_child_samples', 'subsample', 'colsample_bytree',
                       'reg_alpha', 'reg_lambda', 'min_split_gain']
    imp_names, imp_vals = [], []
    for p in numeric_params:
        col = f'param_{p}'
        if col in results_cv.columns:
            vals = results_cv[col].astype(float).values
            score = results_cv['mean_test_score'].values
            corr, _ = spearmanr(vals, score)
            imp_names.append(p)
            imp_vals.append(abs(corr) if not np.isnan(corr) else 0.0)

    order = np.argsort(imp_vals)
    imp_names = [imp_names[i] for i in order]
    imp_vals  = [imp_vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(imp_names))
    colors_imp = plt.cm.viridis(np.linspace(0.2, 0.85, len(imp_names)))
    ax.barh(y_pos, imp_vals, color=colors_imp,
            edgecolor='white', linewidth=0.8, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_names, fontsize=9)
    ax.set_xlabel('|Spearman correlation| with CV ROC-AUC', fontsize=10, color=PALETTE['label'])
    ax.set_title('Stage 2 — Random Search: Approx. Hyperparameter Importance',
                 fontsize=12, fontweight='bold', color=PALETTE['title'])
    ax.spines['left'].set_color(PALETTE['spine'])
    ax.spines['bottom'].set_color(PALETTE['spine'])
    ax.tick_params(length=0)
    for i, val in enumerate(imp_vals):
        ax.text(val + 0.002, i, f'{val:.4f}', va='center',
                fontsize=8, fontweight='bold', color=PALETTE['title'])
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'stage2_random_search_param_importance.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: stage2_random_search_param_importance.png")
except Exception as e:
    print(f"  Param importance plot skipped: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — FINAL MODEL: Fit on full train, evaluate on TEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 3 — FINAL MODEL: Train on full X_train, evaluate on X_test")
print("=" * 70)

# Number of trees = avg best iteration from tuned CV (no leakage)
FINAL_N_ESTIMATORS = max(100, int(np.mean(tuned_best_iters)))
print(f"  Final n_estimators (from CV early stopping avg): {FINAL_N_ESTIMATORS}")

FINAL_PARAMS = {**BEST_PARAMS}
FINAL_PARAMS['n_estimators'] = FINAL_N_ESTIMATORS
# Remove metric/verbose — not needed for final fit
FINAL_PARAMS.pop('metric', None)

final_model = lgb.LGBMClassifier(**FINAL_PARAMS)
final_model.fit(X_arr, y_arr)

# Predictions
y_prob_test_raw  = final_model.predict_proba(X_test.values)[:, 1]
y_prob_train_raw = final_model.predict_proba(X_arr)[:, 1]

# ADDITION 2 FIX -- apply the SAME Platt calibrator fitted on OOF train
# predictions above. The test set is never used to fit the calibrator.
y_prob_test  = apply_calibration(y_prob_test_raw, calibrator)
y_prob_train = apply_calibration(y_prob_train_raw, calibrator)
print(f"  Calibrated test mean predicted: {y_prob_test.mean():.4f}  |  actual test positive rate: {y_test.mean():.4f}")

# Test metrics at default 0.5 and best OOF threshold (calibrated probabilities)
test_metrics_05 = compute_all_metrics(y_test, y_prob_test, threshold=0.5)
test_metrics_bt = compute_all_metrics(y_test, y_prob_test, threshold=best_thresh)
test_metrics_cost = compute_all_metrics(y_test, y_prob_test, threshold=cost_thresh)
train_metrics   = compute_all_metrics(y_arr,  y_prob_train, threshold=best_thresh)

print(f"\n  ─── Test Set Metrics (threshold=0.50) ───")
for k, v in test_metrics_05.items():
    if isinstance(v, float): print(f"  {k:12}: {v:.4f}")

print(f"\n  ─── Test Set Metrics (threshold={best_thresh:.3f}, OOF-optimal) ───")
for k, v in test_metrics_bt.items():
    if isinstance(v, float): print(f"  {k:12}: {v:.4f}")

print(f"\n  ─── Test Set Metrics (threshold={cost_thresh:.3f}, cost-sensitive, "
      f"FN {FN_COST}x FP {FP_COST}x) ───")
for k, v in test_metrics_cost.items():
    if isinstance(v, float): print(f"  {k:12}: {v:.4f}")

# Overfitting check
print(f"\n  ─── Overfitting Check (train vs test at threshold={best_thresh:.3f}) ───")
for metric in ['roc_auc', 'pr_auc', 'f1', 'recall']:
    tr_val   = train_metrics[metric]
    te_val   = test_metrics_bt[metric]
    gap      = tr_val - te_val
    flag     = "⚠️" if gap > 0.05 else "✅"
    print(f"  {flag} {metric:12}: Train={tr_val:.4f}  Test={te_val:.4f}  Gap={gap:+.4f}")

# Save final model
joblib.dump(final_model, MODELS_DIR / 'lgbm_final.pkl')
joblib.dump({
    'threshold': best_thresh,
    'cost_sensitive_threshold': cost_thresh,
    'fn_cost': FN_COST,
    'fp_cost': FP_COST,
}, MODELS_DIR / 'decision_threshold.pkl')
joblib.dump(calibrator, MODELS_DIR / 'calibrator.pkl')
print(f"\n  Model saved → {MODELS_DIR}/lgbm_final.pkl")
print(f"  Threshold saved → {MODELS_DIR}/decision_threshold.pkl")

# Save results
results_dict = {
    'baseline_cv'   : baseline_cv_df.to_dict(),
    'tuned_cv'      : tuned_cv_df.to_dict(),
    'oof_roc_auc'   : oof_roc,
    'oof_pr_auc'    : oof_pr,
    'best_threshold': best_thresh,
    'cost_sensitive_threshold': cost_thresh,
    'fn_cost': FN_COST,
    'fp_cost': FP_COST,
    'test_metrics_05' : test_metrics_05,
    'test_metrics_bt' : test_metrics_bt,
    'test_metrics_cost' : test_metrics_cost,
    'best_params'   : best_params_raw,
    'n_trials'      : N_ITER,
    'search_method' : 'RandomizedSearchCV',
    'best_n_estimators': FINAL_N_ESTIMATORS,
}
with open(RESULTS_DIR / 'lgbm_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — ALL EVALUATION GRAPHS (Test Set)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 4 — EVALUATION GRAPHS")
print("=" * 70)

# ── Graph 3: ROC + PR Curves ─────────────────────────────────────────────────
fpr, tpr, roc_thresh = roc_curve(y_test, y_prob_test)
prec, rec, pr_thresh = precision_recall_curve(y_test, y_prob_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Stage 3 — Final Model: ROC & PR Curves (Test Set)',
             fontsize=14, fontweight='bold', color=PALETTE['title'])

# ROC
roc_auc_val = test_metrics_05['roc_auc']
axes[0].plot(fpr, tpr, color=PALETTE['lgbm'], linewidth=2.5,
             label=f'LightGBM (AUC={roc_auc_val:.4f})')
axes[0].plot([0,1],[0,1], color='#AAAAAA', linewidth=1.2, linestyle='--',
             label='Random (AUC=0.50)')
# Mark best threshold
best_fpr_idx = np.argmin(np.abs(roc_thresh - best_thresh))
if best_fpr_idx < len(fpr):
    axes[0].scatter(fpr[best_fpr_idx], tpr[best_fpr_idx],
                    color=PALETTE['positive'], s=120, zorder=5,
                    label=f'Optimal thresh={best_thresh:.3f}', marker='*')
axes[0].set_xlabel('False Positive Rate', fontsize=10, color=PALETTE['label'])
axes[0].set_ylabel('True Positive Rate', fontsize=10, color=PALETTE['label'])
axes[0].set_title('ROC Curve', fontsize=11, fontweight='bold', color=PALETTE['title'])
axes[0].legend(fontsize=9, frameon=False)
axes[0].spines['left'].set_color(PALETTE['spine'])
axes[0].spines['bottom'].set_color(PALETTE['spine'])
axes[0].tick_params(length=0)

# PR
pr_auc_val  = test_metrics_05['pr_auc']
baseline_pr = y_test.mean()
axes[1].plot(rec, prec, color=PALETTE['optuna'], linewidth=2.5,
             label=f'LightGBM (AP={pr_auc_val:.4f})')
axes[1].axhline(baseline_pr, color='#AAAAAA', linewidth=1.2, linestyle='--',
                label=f'Random (AP={baseline_pr:.4f})')
axes[1].set_xlabel('Recall', fontsize=10, color=PALETTE['label'])
axes[1].set_ylabel('Precision', fontsize=10, color=PALETTE['label'])
axes[1].set_title('Precision-Recall Curve', fontsize=11,
                  fontweight='bold', color=PALETTE['title'])
axes[1].legend(fontsize=9, frameon=False)
axes[1].spines['left'].set_color(PALETTE['spine'])
axes[1].spines['bottom'].set_color(PALETTE['spine'])
axes[1].tick_params(length=0)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGURES_DIR / 'stage3_roc_pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: stage3_roc_pr_curves.png")

# ── Graph 4: Confusion Matrix (both thresholds) ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Stage 3 — Confusion Matrices (Test Set)',
             fontsize=14, fontweight='bold', color=PALETTE['title'])

for ax, thresh, m, title_suffix in zip(
    axes,
    [0.5, best_thresh],
    [test_metrics_05, test_metrics_bt],
    ['Threshold = 0.50', f'Threshold = {best_thresh:.3f} (OOF-optimal)']
):
    cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    cm_pct = cm / cm.sum() * 100

    sns.heatmap(cm, annot=False, fmt='', cmap='Blues', ax=ax,
                linewidths=2, linecolor='white',
                cbar=False, square=True,
                xticklabels=['Pred: Not <30', 'Pred: <30'],
                yticklabels=['True: Not <30', 'True: <30'])

    for i in range(2):
        for j in range(2):
            val  = cm[i, j]
            pct  = cm_pct[i, j]
            col  = 'white' if (i == 0 and j == 0) or (i == 1 and j == 1) else '#0D1B2A'
            ax.text(j + 0.5, i + 0.4, f'{val:,}',
                    ha='center', va='center',
                    fontsize=16, fontweight='bold', color=col)
            ax.text(j + 0.5, i + 0.65, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=10, color=col)

    ax.set_title(f'Confusion Matrix — {title_suffix}\n'
                 f'ROC-AUC={m["roc_auc"]:.4f} | F1={m["f1"]:.4f} | '
                 f'Recall={m["recall"]:.4f}',
                 fontsize=10, fontweight='bold', color=PALETTE['title'])
    ax.set_xlabel('Predicted Label', fontsize=10, color=PALETTE['label'])
    ax.set_ylabel('True Label', fontsize=10, color=PALETTE['label'])
    ax.tick_params(length=0, labelsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGURES_DIR / 'stage3_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: stage3_confusion_matrices.png")

# ── Graph 5: Feature Importance ───────────────────────────────────────────────
feat_names = X_train.columns.tolist()
importances_gain  = final_model.booster_.feature_importance(importance_type='gain')
importances_split = final_model.booster_.feature_importance(importance_type='split')

fi_df = pd.DataFrame({
    'feature'   : feat_names,
    'gain'      : importances_gain,
    'split'     : importances_split,
}).sort_values('gain', ascending=False).head(25)

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle('Stage 3 — Feature Importance: Final LightGBM Model',
             fontsize=14, fontweight='bold', color=PALETTE['title'])

for ax, col, label, cmap_name in zip(
    axes,
    ['gain', 'split'],
    ['Gain Importance (top 25)', 'Split Importance (top 25)'],
    ['viridis', 'plasma']
):
    fi_sorted = fi_df.sort_values(col, ascending=True)
    y_pos   = np.arange(len(fi_sorted))
    colors  = plt.get_cmap(cmap_name)(np.linspace(0.25, 0.85, len(fi_sorted)))
    ax.barh(y_pos, fi_sorted[col].values, color=colors,
            edgecolor='white', linewidth=0.5, height=0.72)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fi_sorted['feature'], fontsize=7.5)
    ax.set_xlabel(f'{col.capitalize()} Score', fontsize=10, color=PALETTE['label'])
    ax.set_title(label, fontsize=11, fontweight='bold', color=PALETTE['title'])
    ax.spines['left'].set_color(PALETTE['spine'])
    ax.spines['bottom'].set_color(PALETTE['spine'])
    ax.tick_params(length=0)
    for i, v in enumerate(fi_sorted[col].values):
        ax.text(v + fi_sorted[col].max()*0.005, i, f'{v:,.0f}',
                va='center', fontsize=6.5, color=PALETTE['title'])

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGURES_DIR / 'stage3_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: stage3_feature_importance.png")

# ── Graph 6: Threshold Analysis ───────────────────────────────────────────────
thresholds     = np.linspace(0.01, 0.99, 200)
thresh_metrics = []

for t in thresholds:
    y_pred = (y_prob_test >= t).astype(int)
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    tn = ((y_pred == 0) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    prec_t = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_t  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_t   = (2 * prec_t * rec_t / (prec_t + rec_t)) if (prec_t + rec_t) > 0 else 0
    spec_t = tn / (tn + fp) if (tn + fp) > 0 else 0
    thresh_metrics.append({
        'threshold': t, 'precision': prec_t,
        'recall': rec_t, 'f1': f1_t, 'specificity': spec_t
    })

thresh_df  = pd.DataFrame(thresh_metrics)
best_f1_t  = thresh_df.loc[thresh_df['f1'].idxmax(), 'threshold']

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(thresh_df['threshold'], thresh_df['precision'],
        color=PALETTE['neutral'], linewidth=2, label='Precision')
ax.plot(thresh_df['threshold'], thresh_df['recall'],
        color=PALETTE['positive'], linewidth=2, label='Recall')
ax.plot(thresh_df['threshold'], thresh_df['f1'],
        color=PALETTE['lgbm'], linewidth=2.5, label='F1 Score')
ax.plot(thresh_df['threshold'], thresh_df['specificity'],
        color=PALETTE['accent'], linewidth=1.5, linestyle='--', label='Specificity')
ax.axvline(best_thresh, color='#333333', linewidth=1.5, linestyle=':',
           label=f'OOF Optimal Threshold = {best_thresh:.3f}')
ax.set_xlabel('Decision Threshold', fontsize=10, color=PALETTE['label'])
ax.set_ylabel('Score', fontsize=10, color=PALETTE['label'])
ax.set_title('Stage 3 — Threshold Analysis (Test Set)',
             fontsize=12, fontweight='bold', color=PALETTE['title'])
ax.legend(fontsize=9, frameon=False)
ax.spines['left'].set_color(PALETTE['spine'])
ax.spines['bottom'].set_color(PALETTE['spine'])
ax.tick_params(length=0)
plt.tight_layout()
fig.savefig(FIGURES_DIR / 'stage3_threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: stage3_threshold_analysis.png")

# ── Graph 7: Calibration Curve ────────────────────────────────────────────────
fraction_pos, mean_pred = calibration_curve(y_test, y_prob_test, n_bins=15)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Stage 3 — Probability Calibration (Test Set)',
             fontsize=14, fontweight='bold', color=PALETTE['title'])

axes[0].plot(mean_pred, fraction_pos, 'o-', color=PALETTE['lgbm'],
             linewidth=2.5, markersize=7, label='LightGBM')
axes[0].plot([0,1],[0,1], '--', color='#AAAAAA', linewidth=1.5,
             label='Perfect calibration')
axes[0].fill_between(mean_pred,
                     fraction_pos - 0.02, fraction_pos + 0.02,
                     color=PALETTE['lgbm'], alpha=0.15)
axes[0].set_xlabel('Mean Predicted Probability', fontsize=10, color=PALETTE['label'])
axes[0].set_ylabel('Fraction of Positives', fontsize=10, color=PALETTE['label'])
axes[0].set_title('Calibration Curve', fontsize=11, fontweight='bold',
                  color=PALETTE['title'])
axes[0].legend(fontsize=9, frameon=False)
axes[0].spines['left'].set_color(PALETTE['spine'])
axes[0].spines['bottom'].set_color(PALETTE['spine'])
axes[0].tick_params(length=0)
axes[0].text(0.98, 0.04, f"Brier={test_metrics_05['brier']:.4f}",
             ha='right', va='bottom', transform=axes[0].transAxes,
             fontsize=10, fontweight='bold', color=PALETTE['lgbm'])

axes[1].hist(y_prob_test[y_test == 0], bins=50, alpha=0.6,
             color=PALETTE['neutral'], label='Not <30 (Neg)', density=True)
axes[1].hist(y_prob_test[y_test == 1], bins=50, alpha=0.6,
             color=PALETTE['positive'], label='<30 days (Pos)', density=True)
axes[1].axvline(best_thresh, color='#333333', linewidth=1.5, linestyle='--',
                label=f'Threshold={best_thresh:.3f}')
axes[1].set_xlabel('Predicted Probability', fontsize=10, color=PALETTE['label'])
axes[1].set_ylabel('Density', fontsize=10, color=PALETTE['label'])
axes[1].set_title('Score Distribution by True Class', fontsize=11,
                  fontweight='bold', color=PALETTE['title'])
axes[1].legend(fontsize=9, frameon=False)
axes[1].spines['left'].set_color(PALETTE['spine'])
axes[1].spines['bottom'].set_color(PALETTE['spine'])
axes[1].tick_params(length=0)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIGURES_DIR / 'stage3_calibration.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: stage3_calibration.png")

# ── Graph 8: Full Metrics Summary Dashboard ───────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

fig.text(0.015, 0.97,
         'Hospital Readmission (< 30 days) — LightGBM Full Evaluation Dashboard',
         fontsize=16, fontweight='bold', color=PALETTE['title'])
fig.text(0.015, 0.950,
         f"Test ROC-AUC={test_metrics_05['roc_auc']:.4f}  |  "
         f"PR-AUC={test_metrics_05['pr_auc']:.4f}  |  "
         f"F1 (thresh={best_thresh:.3f})={test_metrics_bt['f1']:.4f}  |  "
         f"Recall={test_metrics_bt['recall']:.4f}",
         fontsize=10, color='#5A6A7A', style='italic')

# Panel A: ROC
ax_roc = fig.add_subplot(gs[0, 0:2])
ax_roc.plot(fpr, tpr, color=PALETTE['lgbm'], linewidth=2.5,
            label=f"AUC={test_metrics_05['roc_auc']:.4f}")
ax_roc.plot([0,1],[0,1], '--', color='#AAAAAA', linewidth=1)
ax_roc.set_xlabel('FPR', fontsize=9)
ax_roc.set_ylabel('TPR', fontsize=9)
ax_roc.set_title('ROC Curve', fontsize=10, fontweight='bold')
ax_roc.legend(fontsize=8, frameon=False)
ax_roc.tick_params(length=0, labelsize=8)
ax_roc.spines['left'].set_color(PALETTE['spine'])
ax_roc.spines['bottom'].set_color(PALETTE['spine'])

# Panel B: PR
ax_pr = fig.add_subplot(gs[0, 2:4])
ax_pr.plot(rec, prec, color=PALETTE['optuna'], linewidth=2.5,
           label=f"AP={test_metrics_05['pr_auc']:.4f}")
ax_pr.axhline(y_test.mean(), color='#AAAAAA', linestyle='--', linewidth=1)
ax_pr.set_xlabel('Recall', fontsize=9)
ax_pr.set_ylabel('Precision', fontsize=9)
ax_pr.set_title('Precision-Recall Curve', fontsize=10, fontweight='bold')
ax_pr.legend(fontsize=8, frameon=False)
ax_pr.tick_params(length=0, labelsize=8)
ax_pr.spines['left'].set_color(PALETTE['spine'])
ax_pr.spines['bottom'].set_color(PALETTE['spine'])

# Panel C: CV comparison
ax_cv = fig.add_subplot(gs[1, 0:2])
cv_compare = {
    'Metric'  : compare_labels,
    'Baseline': base_vals,
    'Tuned'   : tune_vals,
}
x2 = np.arange(len(compare_labels))
ax_cv.bar(x2 - 0.2, base_vals, 0.35, label='Baseline',
          color=PALETTE['neutral'], edgecolor='white', alpha=0.85)
ax_cv.bar(x2 + 0.2, tune_vals, 0.35, label='Tuned',
          color=PALETTE['lgbm'], edgecolor='white')
ax_cv.set_xticks(x2)
ax_cv.set_xticklabels(compare_labels, rotation=10, ha='right', fontsize=8)
ax_cv.set_ylim(0, 1.0)
ax_cv.set_title('Baseline vs Tuned CV', fontsize=10, fontweight='bold')
ax_cv.legend(fontsize=7, frameon=False)
ax_cv.tick_params(length=0, labelsize=8)
ax_cv.spines['left'].set_color(PALETTE['spine'])
ax_cv.spines['bottom'].set_color(PALETTE['spine'])

# Panel D: Threshold curve
ax_thresh = fig.add_subplot(gs[1, 2:4])
ax_thresh.plot(thresh_df['threshold'], thresh_df['f1'],
               color=PALETTE['lgbm'], linewidth=2, label='F1')
ax_thresh.plot(thresh_df['threshold'], thresh_df['recall'],
               color=PALETTE['positive'], linewidth=1.5, label='Recall')
ax_thresh.plot(thresh_df['threshold'], thresh_df['precision'],
               color=PALETTE['neutral'], linewidth=1.5, label='Precision')
ax_thresh.axvline(best_thresh, color='#333', linewidth=1.3,
                  linestyle=':', label=f'Optimal={best_thresh:.3f}')
ax_thresh.set_xlabel('Threshold', fontsize=9)
ax_thresh.set_title('Threshold Analysis', fontsize=10, fontweight='bold')
ax_thresh.legend(fontsize=7, frameon=False)
ax_thresh.tick_params(length=0, labelsize=8)
ax_thresh.spines['left'].set_color(PALETTE['spine'])
ax_thresh.spines['bottom'].set_color(PALETTE['spine'])

# Panel E: Confusion Matrix
ax_cm = fig.add_subplot(gs[2, 0:2])
m = test_metrics_bt
cm_vals = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
sns.heatmap(cm_vals, annot=True, fmt=',d', cmap='Blues', ax=ax_cm,
            linewidths=2, linecolor='white', cbar=False, square=True,
            xticklabels=['Pred: Neg', 'Pred: Pos'],
            yticklabels=['True: Neg', 'True: Pos'],
            annot_kws={'size': 12, 'weight': 'bold'})
ax_cm.set_title(f'Confusion Matrix (thresh={best_thresh:.3f})',
                fontsize=10, fontweight='bold')
ax_cm.tick_params(length=0, labelsize=8)

# Panel F: Score distribution
ax_dist = fig.add_subplot(gs[2, 2:4])
ax_dist.hist(y_prob_test[y_test == 0], bins=40, alpha=0.55,
             color=PALETTE['neutral'], label='Negative', density=True)
ax_dist.hist(y_prob_test[y_test == 1], bins=40, alpha=0.55,
             color=PALETTE['positive'], label='Positive (<30)', density=True)
ax_dist.axvline(best_thresh, color='#333', linewidth=1.5, linestyle='--',
                label=f'Thresh={best_thresh:.3f}')
ax_dist.set_xlabel('Predicted Score', fontsize=9)
ax_dist.set_title('Score Distribution', fontsize=10, fontweight='bold')
ax_dist.legend(fontsize=7, frameon=False)
ax_dist.tick_params(length=0, labelsize=8)
ax_dist.spines['left'].set_color(PALETTE['spine'])
ax_dist.spines['bottom'].set_color(PALETTE['spine'])

fig.savefig(FIGURES_DIR / 'stage3_full_dashboard.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: stage3_full_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\n  BASELINE LightGBM (5-Fold CV):")
print(f"    ROC-AUC : {baseline_cv_df['roc_auc'].mean():.4f} ± {baseline_cv_df['roc_auc'].std():.4f}")
print(f"    PR-AUC  : {baseline_cv_df['pr_auc'].mean():.4f}  ± {baseline_cv_df['pr_auc'].std():.4f}")
print(f"    F1      : {baseline_cv_df['f1'].mean():.4f}  ± {baseline_cv_df['f1'].std():.4f}")

print(f"\n  TUNED LightGBM (5-Fold CV, RandomizedSearchCV {N_ITER} combinations):")
print(f"    ROC-AUC : {tuned_cv_df['roc_auc'].mean():.4f} ± {tuned_cv_df['roc_auc'].std():.4f}")
print(f"    PR-AUC  : {tuned_cv_df['pr_auc'].mean():.4f}  ± {tuned_cv_df['pr_auc'].std():.4f}")
print(f"    F1      : {tuned_cv_df['f1'].mean():.4f}  ± {tuned_cv_df['f1'].std():.4f}")
print(f"    OOF ROC-AUC: {oof_roc:.4f}  |  OOF PR-AUC: {oof_pr:.4f}")

print(f"\n  FINAL MODEL (Test Set):")
print(f"    ROC-AUC  : {test_metrics_05['roc_auc']:.4f}")
print(f"    PR-AUC   : {test_metrics_05['pr_auc']:.4f}")
print(f"    Threshold: {best_thresh:.3f} (OOF-optimised)")
print(f"    F1       : {test_metrics_bt['f1']:.4f}")
print(f"    Precision: {test_metrics_bt['precision']:.4f}")
print(f"    Recall   : {test_metrics_bt['recall']:.4f}")
print(f"    Specificity: {test_metrics_bt['specificity']:.4f}")
print(f"    Brier    : {test_metrics_05['brier']:.4f}")
print(f"    Log-Loss : {test_metrics_05['log_loss']:.4f}")
print(f"    n_estimators: {FINAL_N_ESTIMATORS}")

print(f"\n  Graphs saved to: {FIGURES_DIR}")
print(f"    ✅ stage1_baseline_cv_metrics.png")
print(f"    ✅ stage1_baseline_learning_curves.png")
print(f"    ✅ stage2_optuna_history.png  (now: Random Search history)")
print(f"    ✅ stage2_tuned_learning_curves.png")
print(f"    ✅ stage2_random_search_param_importance.png")
print(f"    ✅ stage3_roc_pr_curves.png")
print(f"    ✅ stage3_confusion_matrices.png")
print(f"    ✅ stage3_feature_importance.png")
print(f"    ✅ stage3_threshold_analysis.png")
print(f"    ✅ stage3_calibration.png")
print(f"    ✅ stage3_full_dashboard.png")

print(f"\n  Models saved to: {MODELS_DIR}")
print(f"    ✅ lgbm_final.pkl")
print(f"    ✅ decision_threshold.pkl")
print(f"\n  Results saved to: {RESULTS_DIR}")
print(f"    ✅ lgbm_results.json")
print(f"    ✅ best_lgbm_params.json")

print(f"\n  NEXT → Code 5: Compare with XGBoost / RandomForest / LogisticRegression")
print("=" * 70)