"""
Model Evaluation — Phase 2
==================================================================
Loads the LightGBM model saved by model_training.py (models/lgbm_final.pkl)
and evaluates it on the held-out test set.

IMPORTANT — NO DATA LEAKAGE:
The decision threshold is NOT re-searched on the test set. It is loaded
from models/decision_threshold.pkl, which was chosen during training using
only out-of-fold (OOF) predictions on the TRAIN set. The test set here is
used only to REPORT final, unbiased performance — never to pick a
threshold or any other setting. The threshold curve (Graph 7) still sweeps
every threshold on the test set for visualization, but that sweep plays no
role in choosing the operating point.

Overfitting/underfitting was already checked in model_training.py
(train-vs-test gap at the OOF threshold). This script does not repeat
that check — it only reports test-set performance at the fixed threshold.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import joblib
import json

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR    = BASE_DIR / 'models'
FIGURES_DIR   = BASE_DIR / 'reports' / 'figures'
RESULTS_DIR   = BASE_DIR / 'reports' / 'results'

for p in [FIGURES_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ── Professional, decent design tokens (grid OFF everywhere) ─────────────────
BG        = '#FFFFFF'
PANEL     = '#FFFFFF'
SPINE     = '#D7DEE5'
TITLE_CLR = '#1B2A38'
LABEL_CLR = '#3A4652'
TICK_CLR  = '#5C6B78'

PALETTE = {
    'primary'   : '#2C5F8A',   # steel blue — main model line
    'secondary' : '#3F7D5C',   # muted green — secondary series
    'accent'    : '#B5762B',   # muted amber — highlight/threshold marker
    'negative'  : '#8892A0',   # slate grey — reference/random lines
    'fp'        : '#B85C4A',   # muted terracotta — false positives
    'fn'        : '#4A5FA0',   # muted indigo — false negatives
    'tp'        : '#3F7D5C',   # muted green — true positives
    'pos_class' : '#B5762B',
    'neg_class' : '#5C7A96',
}

FEATURE_CMAP = LinearSegmentedColormap.from_list(
    'slate_teal', ['#1F3A4D', '#2C6E7F', '#5FA8A3', '#A8D3CB'], N=256
)
SHAP_CMAP = LinearSegmentedColormap.from_list(
    'slate_amber', ['#2C5F8A', '#6E8FA8', '#B5A07A', '#B5762B'], N=256
)

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'figure.facecolor' : BG,
    'axes.facecolor'   : PANEL,
    'axes.grid'        : False,     # grid OFF, project-wide
    'savefig.dpi'      : 300,
})

def _style_ax(ax):
    """Consistent, clean axis styling — no grid, soft spines."""
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(SPINE)
    ax.spines['bottom'].set_color(SPINE)
    ax.tick_params(colors=TICK_CLR, length=0)
    ax.grid(False)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA, MODEL & THRESHOLD (threshold comes from TRAINING, not test)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("MODEL EVALUATION — HELD-OUT TEST SET")
print("=" * 65)

print("\n[1] Loading data, model and OOF-chosen threshold...")
X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv').squeeze()   # column-name agnostic

model_path = MODELS_DIR / 'lgbm_final.pkl'
if not model_path.exists():
    raise FileNotFoundError(
        f"{model_path} not found. Run model_training.py first."
    )
model = joblib.load(model_path)

threshold_path = MODELS_DIR / 'decision_threshold.pkl'
if not threshold_path.exists():
    raise FileNotFoundError(
        f"{threshold_path} not found. This file is written by "
        f"model_training.py from OOF predictions — evaluation must not "
        f"invent its own threshold from the test set."
    )
THRESHOLD = joblib.load(threshold_path)['threshold']

MODEL_NAME = 'LightGBM'
print(f"   Model            : {MODEL_NAME}")
print(f"   Test set size    : {len(y_test):,}")
print(f"   Positive class   : {y_test.mean()*100:.2f}%")
print(f"   Decision threshold (from training OOF, no leakage): {THRESHOLD:.3f}")

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred       = (y_pred_proba >= THRESHOLD).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# 2. METRICS AT THE FIXED (OOF) THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Metrics at fixed threshold...")

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_pred_proba)
ap   = average_precision_score(y_test, y_pred_proba)
tn_v, fp_v, fn_v, tp_v = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
specificity = tn_v / (tn_v + fp_v) if (tn_v + fp_v) > 0 else 0

print(f"\n  Accuracy    : {acc:.4f}")
print(f"  Precision   : {prec:.4f}")
print(f"  Recall      : {rec:.4f}")
print(f"  Specificity : {specificity:.4f}")
print(f"  F1-Score    : {f1:.4f}")
print(f"  ROC-AUC     : {auc:.4f}")
print(f"  PR-AUC      : {ap:.4f}")
print(f"\n  TP={tp_v}  FP={fp_v}  TN={tn_v}  FN={fn_v}")
print("\n" + classification_report(y_test, y_pred, target_names=['Not <30', 'Readmitted <30']))

# ══════════════════════════════════════════════════════════════════════════════
# G1 — ROC & PR CURVES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] G1 — ROC & PR Curves...")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
rec_curve, prec_curve, _ = precision_recall_curve(y_test, y_pred_proba)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)
for ax in axes:
    ax.set_facecolor(PANEL)

axes[0].plot(fpr, tpr, color=PALETTE['primary'], linewidth=2.5,
             label=f'{MODEL_NAME} (AUC={auc:.4f})')
axes[0].plot([0, 1], [0, 1], '--', color=PALETTE['negative'], linewidth=1.2, label='Random')
axes[0].set_xlabel('False Positive Rate', color=LABEL_CLR)
axes[0].set_ylabel('True Positive Rate', color=LABEL_CLR)
axes[0].set_title('ROC Curve', fontsize=12, fontweight='bold', color=TITLE_CLR)
axes[0].legend(fontsize=9, frameon=False)
_style_ax(axes[0])

axes[1].plot(rec_curve, prec_curve, color=PALETTE['secondary'], linewidth=2.5,
             label=f'{MODEL_NAME} (AP={ap:.4f})')
axes[1].axhline(y_test.mean(), color=PALETTE['negative'], linestyle='--', linewidth=1.2,
                 label='Random (prevalence)')
axes[1].set_xlabel('Recall', color=LABEL_CLR)
axes[1].set_ylabel('Precision', color=LABEL_CLR)
axes[1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold', color=TITLE_CLR)
axes[1].legend(fontsize=9, frameon=False)
_style_ax(axes[1])

fig.suptitle(f'Graph 1  |  ROC & PR Curves — {MODEL_NAME}', fontsize=13, fontweight='bold', color=TITLE_CLR)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIGURES_DIR / 'G1_roc_pr_curves.png', dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("   Saved: G1_roc_pr_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
# G2 — CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] G2 — Confusion Matrix...")
fig, ax = plt.subplots(figsize=(6.5, 6), facecolor=BG)
ax.set_facecolor(PANEL)
cm = np.array([[tn_v, fp_v], [fn_v, tp_v]])
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax, linewidths=2,
            linecolor='white', cbar=False, square=True,
            xticklabels=['Pred: Not <30', 'Pred: <30'],
            yticklabels=['True: Not <30', 'True: <30'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title(f'Graph 2  |  Confusion Matrix (threshold={THRESHOLD:.3f})',
             fontsize=13, fontweight='bold', color=TITLE_CLR)
fig.savefig(FIGURES_DIR / 'G2_confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("   Saved: G2_confusion_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# G7 — THRESHOLD CURVE (visualization only — the marked point is the
#      threshold chosen during TRAINING, this sweep does not select it)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] G7 — Threshold Curve (visual reference only)...")

thresholds = np.linspace(0, 1, 200)
precisions, recalls, f1_scores = [], [], []
for t in thresholds:
    yp = (y_pred_proba >= t).astype(int)
    precisions.append(precision_score(y_test, yp, zero_division=0))
    recalls.append(recall_score(y_test, yp, zero_division=0))
    f1_scores.append(f1_score(y_test, yp, zero_division=0))

idx_at_threshold = np.argmin(np.abs(thresholds - THRESHOLD))
f1_at, pr_at, rc_at = f1_scores[idx_at_threshold], precisions[idx_at_threshold], recalls[idx_at_threshold]

fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
fig.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.13)
ax.set_facecolor(PANEL)

ax.plot(thresholds, precisions, lw=2.0, color=PALETTE['primary'], label='Precision', alpha=0.9)
ax.plot(thresholds, recalls, lw=2.0, color=PALETTE['fp'], label='Recall', alpha=0.9)
ax.plot(thresholds, f1_scores, lw=2.5, color=PALETTE['secondary'], label='F1-Score', alpha=1.0)
ax.axvline(THRESHOLD, color=PALETTE['accent'], lw=2.2, ls='--',
           label=f'Used Threshold = {THRESHOLD:.3f}  (from training OOF)', zorder=4)
ax.axvline(0.5, color=PALETTE['negative'], lw=1.1, ls=':', label='Default (0.5)', alpha=0.7)

for val, clr, lbl in [
    (f1_at, PALETTE['secondary'], f'F1={f1_at:.3f}'),
    (pr_at, PALETTE['primary'], f'P={pr_at:.3f}'),
    (rc_at, PALETTE['fp'], f'R={rc_at:.3f}'),
]:
    ax.scatter(THRESHOLD, val, color=clr, s=90, zorder=6, edgecolors='white', linewidths=1.5)
    ax.text(THRESHOLD + 0.015, val + 0.012, lbl, fontsize=7.5, color=clr, fontweight='bold')

ax.set_xlabel('Threshold', fontsize=11, color=LABEL_CLR, labelpad=6)
ax.set_ylabel('Score', fontsize=11, color=LABEL_CLR, labelpad=6)
ax.set_title(f'Graph 7  |  Threshold Curve — operating point fixed at {THRESHOLD:.3f} (chosen during training)',
             fontsize=12, fontweight='bold', color=TITLE_CLR, pad=10)
ax.legend(fontsize=8.5, frameon=True, facecolor=BG, edgecolor=SPINE, loc='upper right')
_style_ax(ax)

fig.savefig(FIGURES_DIR / 'G7_threshold_curve.png', dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("   Saved: G7_threshold_curve.png")

# ══════════════════════════════════════════════════════════════════════════════
# G8 — FEATURE IMPORTANCE (gain)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] G8 — Feature Importance...")

imp_df = pd.DataFrame({
    'Feature'   : X_test.columns,
    'Importance': model.booster_.feature_importance(importance_type='gain'),
}).sort_values('Importance', ascending=False).reset_index(drop=True)
top15 = imp_df.head(15).reset_index(drop=True)

bar_colors = FEATURE_CMAP(np.linspace(0.15, 0.90, len(top15)))

fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
fig.subplots_adjust(left=0.34, right=0.93, top=0.87, bottom=0.10)
ax.set_facecolor(PANEL)

bars = ax.barh(top15['Feature'], top15['Importance'], color=bar_colors, edgecolor='white', height=0.62)
for bar, val in zip(bars, top15['Importance']):
    ax.text(val + top15['Importance'].max() * 0.012, bar.get_y() + bar.get_height() / 2,
            f'{val:,.0f}', va='center', ha='left', fontsize=8.5, fontweight='bold', color=TITLE_CLR)

ax.invert_yaxis()
ax.set_xlabel('Importance (Gain)', fontsize=11, color=LABEL_CLR, labelpad=6)
ax.set_title(f'Graph 8  |  Top 15 Feature Importances — {MODEL_NAME}',
             fontsize=13, fontweight='bold', color=TITLE_CLR, pad=10)
ax.tick_params(axis='y', labelsize=9, colors=TICK_CLR)
_style_ax(ax)

fig.savefig(FIGURES_DIR / 'G8_feature_importance.png', dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("   Saved: G8_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# G9 / G10 — SHAP ANALYSIS (bar importance + beeswarm summary)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] G9/G10 — SHAP Analysis...")
try:
    import shap

    N_SHAP_SAMPLE = min(1000, len(X_test))
    X_sample = X_test.sample(N_SHAP_SAMPLE, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    # Some SHAP/model combinations return a list of arrays (one per class)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

    # ── G9: SHAP beeswarm summary plot (native shap style, muted look) ──────
    plt.figure(figsize=(10, 8), facecolor=BG)
    shap.summary_plot(
        shap_vals, X_sample, show=False, plot_size=None,
        cmap=SHAP_CMAP,
    )
    fig9 = plt.gcf()
    fig9.set_facecolor(BG)
    ax9 = plt.gca()
    ax9.set_facecolor(PANEL)
    ax9.set_title(f'Graph 9  |  SHAP Summary (Beeswarm) — {MODEL_NAME}',
                  fontsize=13, fontweight='bold', color=TITLE_CLR, pad=12)
    for s in ['top', 'right']:
        ax9.spines[s].set_visible(False)
    ax9.grid(False)
    plt.tight_layout()
    fig9.savefig(FIGURES_DIR / 'G9_shap_summary_beeswarm.png', dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("   Saved: G9_shap_summary_beeswarm.png")

    # ── G10: SHAP mean |value| bar chart (matches project styling) ──────────
    mean_shap = np.abs(shap_vals).mean(axis=0)
    shap_bar_df = pd.DataFrame({
        'Feature': X_sample.columns, 'Mean_SHAP': mean_shap,
    }).sort_values('Mean_SHAP', ascending=False).head(15).reset_index(drop=True)

    bar_colors_g10 = SHAP_CMAP(np.linspace(0.10, 0.95, len(shap_bar_df)))

    fig10, ax10 = plt.subplots(figsize=(11, 7), facecolor=BG)
    fig10.subplots_adjust(left=0.34, right=0.93, top=0.87, bottom=0.10)
    ax10.set_facecolor(PANEL)

    bars10 = ax10.barh(shap_bar_df['Feature'], shap_bar_df['Mean_SHAP'],
                        color=bar_colors_g10, edgecolor='white', height=0.62)
    for bar, val in zip(bars10, shap_bar_df['Mean_SHAP']):
        ax10.text(val + shap_bar_df['Mean_SHAP'].max() * 0.012,
                   bar.get_y() + bar.get_height() / 2,
                   f'{val:.4f}', va='center', ha='left',
                   fontsize=8.5, fontweight='bold', color=TITLE_CLR)

    ax10.invert_yaxis()
    ax10.set_xlabel('Mean |SHAP Value|', fontsize=11, color=LABEL_CLR, labelpad=6)
    ax10.set_title(f'Graph 10  |  SHAP Feature Importance — {MODEL_NAME}',
                   fontsize=13, fontweight='bold', color=TITLE_CLR, pad=10)
    ax10.tick_params(axis='y', labelsize=9, colors=TICK_CLR)
    _style_ax(ax10)

    fig10.savefig(FIGURES_DIR / 'G10_shap_bar.png', dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("   Saved: G10_shap_bar.png")

    shap_bar_df.to_csv(RESULTS_DIR / 'shap_importance.csv', index=False)
    print("   Saved: shap_importance.csv")

except ImportError:
    print("   SHAP not installed — run: pip install shap")
except Exception as e:
    print(f"   SHAP error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# G11 — CALIBRATION CURVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] G11 — Calibration Curve...")

prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
fig.subplots_adjust(left=0.12, right=0.97, top=0.87, bottom=0.12)
ax.set_facecolor(PANEL)

ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.15, color=PALETTE['secondary'], zorder=2)
ax.plot([0, 1], [0, 1], lw=1.6, ls='--', color=PALETTE['negative'], label='Perfectly Calibrated', zorder=3)
ax.plot(prob_pred, prob_true, lw=2.5, color=PALETTE['secondary'], label=MODEL_NAME, zorder=4)
ax.scatter(prob_pred, prob_true, s=75, color=PALETTE['secondary'], edgecolors='white', linewidths=1.2, zorder=5)

mid = len(prob_pred) // 2
gap = abs(prob_true[mid] - prob_pred[mid])
ax.annotate(f'Gap ≈ {gap:.3f}',
            xy=(prob_pred[mid], (prob_true[mid] + prob_pred[mid]) / 2),
            fontsize=8.5, color=PALETTE['secondary'], fontstyle='italic',
            xytext=(prob_pred[mid] + 0.08, (prob_true[mid] + prob_pred[mid]) / 2),
            arrowprops=dict(arrowstyle='->', color=PALETTE['secondary'], lw=1.0))

ax.set_xlabel('Mean Predicted Probability', fontsize=11, color=LABEL_CLR, labelpad=6)
ax.set_ylabel('Fraction of Positives', fontsize=11, color=LABEL_CLR, labelpad=6)
ax.set_title(f'Graph 11  |  Calibration Curve — {MODEL_NAME}',
             fontsize=13, fontweight='bold', color=TITLE_CLR, pad=10)
ax.legend(fontsize=9.5, frameon=True, facecolor=BG, edgecolor=SPINE, loc='upper left')
_style_ax(ax)

fig.savefig(FIGURES_DIR / 'G11_calibration_curve.png', dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("   Saved: G11_calibration_curve.png")

# ══════════════════════════════════════════════════════════════════════════════
# G12 — ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] G12 — Error Analysis...")

results_df = X_test.copy()
results_df['actual'] = y_test.values
results_df['predicted'] = y_pred
results_df['proba'] = y_pred_proba

fp_df = results_df[(results_df['actual'] == 0) & (results_df['predicted'] == 1)]
fn_df = results_df[(results_df['actual'] == 1) & (results_df['predicted'] == 0)]
tp_df = results_df[(results_df['actual'] == 1) & (results_df['predicted'] == 1)]
tn_df = results_df[(results_df['actual'] == 0) & (results_df['predicted'] == 0)]

print(f"   TP={len(tp_df)}  FP={len(fp_df)}  FN={len(fn_df)}  TN={len(tn_df)}")

key_feats = [f for f in [
    'time_in_hospital', 'num_medications', 'num_procedures', 'number_diagnoses'
] if f in results_df.columns]

if key_feats:
    err_cmp = pd.DataFrame({
        'Feature': key_feats,
        'FP_mean': [fp_df[f].mean() if len(fp_df) > 0 else 0 for f in key_feats],
        'FN_mean': [fn_df[f].mean() if len(fn_df) > 0 else 0 for f in key_feats],
        'TP_mean': [tp_df[f].mean() if len(tp_df) > 0 else 0 for f in key_feats],
    })

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.22)
    ax.set_facecolor(PANEL)

    x = np.arange(len(key_feats))
    w = 0.26

    b1 = ax.bar(x - w, err_cmp['FP_mean'], w, label=f'False Positives  (n={len(fp_df):,})',
                color=PALETTE['fp'], edgecolor='white', linewidth=0.8, zorder=3)
    b2 = ax.bar(x, err_cmp['FN_mean'], w, label=f'False Negatives  (n={len(fn_df):,})',
                color=PALETTE['fn'], edgecolor='white', linewidth=0.8, zorder=3)
    b3 = ax.bar(x + w, err_cmp['TP_mean'], w, label=f'True Positives   (n={len(tp_df):,})',
                color=PALETTE['tp'], edgecolor='white', linewidth=0.8, zorder=3)

    for bars_grp in [b1, b2, b3]:
        for bar in bars_grp:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + ax.get_ylim()[1] * 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                    color=bar.get_facecolor())

    readable = {
        'time_in_hospital': 'Time in\nHospital', 'num_medications': 'Medications',
        'num_procedures': 'Procedures', 'number_diagnoses': 'Diagnoses\nCount',
    }
    ax.set_xticks(x)
    ax.set_xticklabels([readable.get(f, f) for f in key_feats], fontsize=10, color=TICK_CLR)
    ax.set_ylabel('Mean Value (scaled features)', fontsize=11, color=LABEL_CLR, labelpad=6)
    ax.set_title('Graph 12  |  Error Analysis — Feature Profiles (FP · FN · TP)',
                 fontsize=13, fontweight='bold', color=TITLE_CLR, pad=10)

    handles = [
        mpatches.Patch(facecolor=PALETTE['fp'], edgecolor='white', label=f'False Positives  (n={len(fp_df):,})'),
        mpatches.Patch(facecolor=PALETTE['fn'], edgecolor='white', label=f'False Negatives  (n={len(fn_df):,})'),
        mpatches.Patch(facecolor=PALETTE['tp'], edgecolor='white', label=f'True Positives   (n={len(tp_df):,})'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=9,
               frameon=True, facecolor=BG, edgecolor=SPINE, framealpha=1.0,
               handlelength=1.4, handleheight=0.9, columnspacing=2.0, borderpad=0.8)
    _style_ax(ax)

    fig.savefig(FIGURES_DIR / 'G12_error_analysis.png', dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("   Saved: G12_error_analysis.png")

    err_cmp.to_csv(RESULTS_DIR / 'error_analysis_comparison.csv', index=False)
    print("   Saved: error_analysis_comparison.csv")
else:
    print("   Key features not found — skipping error plot.")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE METRICS & PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Saving artifacts...")

metrics_row = {
    'Model': MODEL_NAME,
    'Threshold': THRESHOLD,
    'Threshold_source': 'training_OOF (decision_threshold.pkl) — no test leakage',
    'Accuracy': round(acc, 4),
    'Precision': round(prec, 4),
    'Recall': round(rec, 4),
    'Specificity': round(specificity, 4),
    'F1': round(f1, 4),
    'ROC_AUC': round(auc, 4),
    'PR_AUC': round(ap, 4),
    'TP': tp_v, 'FP': fp_v, 'TN': tn_v, 'FN': fn_v,
}
pd.DataFrame([metrics_row]).to_csv(RESULTS_DIR / 'test_metrics_final.csv', index=False)
with open(RESULTS_DIR / 'test_metrics_final.json', 'w') as f:
    json.dump(metrics_row, f, indent=2, default=str)

preds_out = X_test.copy()
preds_out['actual'] = y_test.values
preds_out[f'predicted_{THRESHOLD:.3f}'] = y_pred
preds_out['proba'] = y_pred_proba
preds_out.to_csv(RESULTS_DIR / f'test_predictions_threshold_{THRESHOLD:.3f}.csv', index=False)

print("   Saved: test_metrics_final.csv / .json")
print(f"   Saved: test_predictions_threshold_{THRESHOLD:.3f}.csv")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"EVALUATION COMPLETE — MODEL = {MODEL_NAME.upper()} | THRESHOLD = {THRESHOLD:.3f} (from training OOF)")
print("=" * 65)
print(f"\nFinal Test Performance:")
print(f"  Recall={rec:.4f}  Precision={prec:.4f}  F1={f1:.4f}  "
      f"Specificity={specificity:.4f}  ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}")
print("\nGraphs saved to:", FIGURES_DIR)
print("  G1  ROC & PR curves")
print("  G2  Confusion matrix")
print("  G7  Threshold curve (visual reference — operating point fixed from training)")
print("  G8  Feature importance (gain)")
print("  G9  SHAP summary (beeswarm)")
print("  G10 SHAP mean |value| bar")
print("  G11 Calibration curve")
print("  G12 Error analysis (FP/FN/TP feature profiles)")
print("=" * 65)