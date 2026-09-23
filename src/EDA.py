import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from scipy.stats import chi2_contingency, f_oneway, kruskal

BASE_DIR = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE SAVE PATH
# ═══════════════════════════════════════════════════════════════════════════
OUT = BASE_DIR / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────
# 🔴 FIX: Correct file name — loads cleaned output from Code 1
df = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'diabetic_data_cleaned.csv')

print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── AGE ORDERING ──────────────────────────────────────────────────────────
age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
             '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
df['age'] = pd.Categorical(df['age'], categories=age_order, ordered=True)

# ── GROUPED COLUMNS ───────────────────────────────────────────────────────
for col in ['medical_specialty', 'payer_code']:
    top_cats = df[col].value_counts().head(10).index
    df[col + '_grouped'] = df[col].apply(lambda x: x if x in top_cats else 'Other')

# ── TARGET BINARY (created once, used throughout) ─────────────────────────
# 1 = readmitted within 30 days (high risk), 0 = otherwise
# Created here so all steps can use it — NO data leakage because this
# is only used for VISUALIZATION, not for fitting any model/transform.
df['target_binary'] = (df['readmitted'] == '<30').astype(int)

print(f"target_binary: {df['target_binary'].sum():,} positive ({df['target_binary'].mean()*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════
CLASS_ORDER = ['<30', '>30', 'NO']

VIOLIN_COLORS = {
    '<30': '#0D4A45',
    '>30': '#1A8A7A',
    'NO':  '#7EC8BE',
}

CLASS_COLORS_8 = {
    '<30': '#D7462C',
    '>30': '#7B3F6E',
    'NO':  '#2E8B57',
}
CLASS_LABELS = {
    '<30': 'Readmitted < 30 days',
    '>30': 'Readmitted > 30 days',
    'NO':  'Not Readmitted',
}

# 🔴 FIX: examide & citoglipton removed — both dropped in Code 1 (zero variance)
MED_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]

MED_BAR_COLORS = [
    '#D7462C', '#7B3F6E', '#2E8B57', '#1B6B7B', '#D4860A',
    '#4A7A4A', '#7B1F3A', '#1A2E5A', '#5C3D2A', '#2A4A3A',
]

MED_CLASS_COLORS = {
    '<30': '#4E79A7',
    '>30': '#F28E2B',
    'NO':  '#59A14F',
}
MED_CLASS_LABELS = {
    '<30': '<30 days (High Risk)',
    '>30': '>30 days',
    'NO':  'Not Readmitted',
}

BG        = '#FFFFFF'
PANEL_BG  = '#F7FAFC'
SPINE_CLR = '#D1DCE5'
TITLE_CLR = '#0D1B2A'
LABEL_CLR = '#2C3E50'
TICK_CLR  = '#4A5568'
ANNOT_THR = 7

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          False,
})


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — Numerical Features — TEAL Violin Plots
# ═══════════════════════════════════════════════════════════════════════════
num_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'number_diagnoses']
num_titles   = ['Length of Stay (days)', 'Laboratory Procedures',
                'Procedures Performed', 'Medications Prescribed',
                'Outpatient Visits (past year)', 'Emergency Visits (past year)',
                'Inpatient Admissions (past year)', 'Diagnoses Count']

def plot_violin(ax, col, title):
    data, positions, colors = [], [], []
    for i, cls in enumerate(CLASS_ORDER):
        vals = df[df['readmitted'] == cls][col].dropna()
        if len(vals) == 0:
            continue
        data.append(vals)
        positions.append(i)
        colors.append(VIOLIN_COLORS[cls])
    parts = ax.violinplot(data, positions=positions, widths=0.6,
                          showmeans=False, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.80)
        pc.set_edgecolor('white')
        pc.set_linewidth(0.8)
    parts['cmedians'].set_color('white')
    parts['cmedians'].set_linewidth(2)
    for i, vals in enumerate(data):
        median = np.median(vals)
        ax.plot([i - 0.18, i + 0.18], [median, median],
                color='white', linewidth=2, zorder=4)
        ax.text(i + 0.22, median, f'{median:.0f}',
                ha='left', va='center',
                fontsize=8, fontweight='bold', color=TITLE_CLR, zorder=5)
    ax.set_xticks(positions)
    ax.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER],
                       rotation=15, ha='right', fontsize=7)
    ax.set_ylabel('Value', fontsize=8, color=LABEL_CLR)
    ax.set_title(title, fontsize=9, fontweight='bold', color=TITLE_CLR, loc='left')
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(SPINE_CLR)
    ax.spines['bottom'].set_color(SPINE_CLR)
    ax.tick_params(length=0)

fig_num, axes_num = plt.subplots(2, 4, figsize=(20, 10))
fig_num.patch.set_facecolor(BG)
fig_num.subplots_adjust(top=0.88, bottom=0.18, left=0.06,
                        right=0.97, wspace=0.42, hspace=0.65)

for ax, col, title in zip(axes_num.flatten(), num_features, num_titles):
    plot_violin(ax, col, title)

fig_num.text(0.015, 0.96,
             'Step 7  |  Hospital Readmission — Numerical Features',
             fontsize=13, fontweight='bold', color=TITLE_CLR)
fig_num.text(0.015, 0.935,
             'Violin plots — distribution across readmission classes  (white line = median)',
             fontsize=8.5, color='#5A6A7A', style='italic')

handles_num = [
    mpatches.Patch(facecolor=VIOLIN_COLORS[c], edgecolor='white',
                   linewidth=0.5, label=CLASS_LABELS[c])
    for c in CLASS_ORDER
]
fig_num.legend(handles=handles_num,
               title='Readmission Status', title_fontsize=8.5, fontsize=9,
               loc='lower center', ncol=3,
               frameon=True, framealpha=1.0,
               edgecolor=SPINE_CLR, facecolor=BG,
               bbox_to_anchor=(0.5, 0.04),
               handlelength=1.4, handleheight=0.9, labelcolor=LABEL_CLR,
               columnspacing=2.0, borderpad=0.8)

fig_num.savefig(OUT / 'step7_violin_readmission.png',
                dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: step7_violin_readmission.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — Categorical Features — Grouped Bar (Retro Metro palette)
# ═══════════════════════════════════════════════════════════════════════════
def plot_grouped_bar(ax, col, title, class_colors, class_labels_dict):
    ct = pd.crosstab(df[col], df['readmitted'], normalize='index') * 100
    classes    = [c for c in CLASS_ORDER if c in ct.columns]
    categories = ct.index.tolist()
    n_cats     = len(categories)
    n_cls      = len(classes)

    group_w = 0.72
    bar_w   = group_w / n_cls
    offsets = np.linspace(-group_w/2 + bar_w/2, group_w/2 - bar_w/2, n_cls)
    x       = np.arange(n_cats)

    ax.set_facecolor(PANEL_BG)

    for i, cls in enumerate(classes):
        vals = ct[cls].values
        xpos = x + offsets[i]
        bars = ax.bar(xpos, vals,
                      width=bar_w - 0.03,
                      color=class_colors[cls],
                      edgecolor='white',
                      linewidth=0.9,
                      zorder=3)
        for bar, v in zip(bars, vals):
            if v >= ANNOT_THR:
                txt = ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f'{v:.1f}%',
                    ha='center', va='bottom',
                    fontsize=6.0, fontweight='bold',
                    color=class_colors[cls], zorder=5)
                txt.set_path_effects(
                    [pe.withStroke(linewidth=1.0, foreground='#FFFFFF80')])

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=40, ha='right',
                       fontsize=7.8, color=TICK_CLR)
    ax.set_ylim(0, ct.values.max() * 1.35)
    ax.set_ylabel('Patients (%)', fontsize=8.5, color=LABEL_CLR, labelpad=5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v)}%'))
    ax.tick_params(axis='y', colors=TICK_CLR, labelsize=8, length=0)
    ax.tick_params(axis='x', length=0)
    ax.set_title(title, fontsize=10, fontweight='bold',
                 color=TITLE_CLR, pad=8, loc='left')
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(SPINE_CLR)
    ax.spines['bottom'].set_linewidth(0.8)

def shared_legend_grouped(fig, class_colors, class_labels_dict):
    handles = [
        mpatches.Patch(facecolor=class_colors[c], edgecolor='white',
                       linewidth=0.5, label=class_labels_dict[c])
        for c in CLASS_ORDER
    ]
    fig.legend(handles=handles,
               title='Readmission Status', title_fontsize=8.5, fontsize=8,
               loc='lower center', ncol=3,
               frameon=True, framealpha=1.0, edgecolor=SPINE_CLR,
               facecolor=BG, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.4, handleheight=0.9, labelcolor=LABEL_CLR)

# Part 1
cat1    = ['age', 'gender', 'race', 'admission_type_id',
           'discharge_disposition_id', 'change']
titles1 = ['Readmission by Age Group', 'Readmission by Gender',
           'Readmission by Race / Ethnicity', 'Readmission by Admission Type',
           'Readmission by Discharge Disposition',
           'Readmission by Medication Change']

fig1, axes1 = plt.subplots(2, 3, figsize=(21, 12))
fig1.patch.set_facecolor(BG)
for ax, col, title in zip(axes1.flatten(), cat1, titles1):
    plot_grouped_bar(ax, col, title, CLASS_COLORS_8, CLASS_LABELS)
fig1.text(0.015, 1.005,
          'Step 8  |  Hospital Readmission — Categorical Features (Part 1)',
          fontsize=13, fontweight='bold', color=TITLE_CLR, va='bottom')
fig1.text(0.015, 0.998,
          'Grouped bars show % of patients per readmission class within each category',
          fontsize=8.5, color='#5A6A7A', va='top', style='italic')
shared_legend_grouped(fig1, CLASS_COLORS_8, CLASS_LABELS)
plt.tight_layout(pad=2.5, h_pad=3.5, w_pad=3.0)
fig1.savefig(OUT / 'step8_categorical_part1_main.png',
             dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: step8_categorical_part1_main.png")

# Part 2
cat2    = ['diabetesMed', 'payer_code_grouped', 'medical_specialty_grouped']
titles2 = ['Readmission by Diabetes Medication',
           'Readmission by Payer Code  (Top 10 + Other)',
           'Readmission by Medical Specialty  (Top 10 + Other)']

fig2, axes2 = plt.subplots(1, 3, figsize=(21, 7))
fig2.patch.set_facecolor(BG)
for ax, col, title in zip(axes2, cat2, titles2):
    plot_grouped_bar(ax, col, title, CLASS_COLORS_8, CLASS_LABELS)
fig2.text(0.015, 1.01,
          'Step 8  |  Hospital Readmission — Categorical Features (Part 2)',
          fontsize=13, fontweight='bold', color=TITLE_CLR, va='bottom')
fig2.text(0.015, 1.003,
          'Payer code and medical specialty grouped; top 10 categories shown individually',
          fontsize=8.5, color='#5A6A7A', va='top', style='italic')
shared_legend_grouped(fig2, CLASS_COLORS_8, CLASS_LABELS)
plt.tight_layout(pad=2.5, w_pad=3.5)
fig2.savefig(OUT / 'step8_categorical_part2_payer_specialty.png',
             dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: step8_categorical_part2_payer_specialty.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 9a — Top 10 Prescription Rate
# 🔴 FIX 1: examide & citoglipton removed from MED_COLS (defined at top)
# 🟡 FIX 2: hardcoded n=101,766 → dynamic len(df)
# ═══════════════════════════════════════════════════════════════════════════
prescribed_pct = {}
for med in MED_COLS:
    if med in df.columns:
        prescribed_pct[med] = (df[med] != 'No').mean() * 100

top_meds = pd.Series(prescribed_pct).sort_values(ascending=False).head(10)

fig9a, ax9a = plt.subplots(figsize=(12, 7), facecolor='white')
fig9a.subplots_adjust(left=0.22, right=0.90, top=0.88, bottom=0.12)

bars = ax9a.barh(
    top_meds.index[::-1],
    top_meds.values[::-1],
    color=MED_BAR_COLORS,
    height=0.58,
    edgecolor='white',
    linewidth=0.8
)

for bar, val, color in zip(bars, top_meds.values[::-1], MED_BAR_COLORS):
    ax9a.text(
        val + 0.4,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.1f}%',
        va='center', ha='left',
        fontsize=9, fontweight='bold', color='#1A1A2E'
    )

ax9a.set_xlim(0, top_meds.max() * 1.20)
ax9a.set_xlabel('Patients Prescribed (%)', fontsize=10, color='#444', labelpad=6)
ax9a.set_title('Step 9  |  Top 10 Most Prescribed Diabetes Medications',
               fontsize=13, fontweight='bold', color='#1A1A2E', pad=12)
ax9a.spines['left'].set_color('#C5D0DC')
ax9a.spines['bottom'].set_color('#C5D0DC')
ax9a.tick_params(length=0, labelsize=9)

med_legend_handles = [
    mpatches.Patch(facecolor=MED_BAR_COLORS[i],
                   label=f'{top_meds.index[::-1][i]}  ({top_meds.values[::-1][i]:.1f}%)')
    for i in range(len(top_meds))
]
ax9a.legend(handles=med_legend_handles,
            loc='lower right', fontsize=7.5,
            frameon=True, facecolor='white',
            edgecolor='#DDDDDD', framealpha=1.0,
            handlelength=1.2, handleheight=0.9,
            ncol=1, borderpad=0.8)

# 🟡 FIX: dynamic n instead of hardcoded 101,766
fig9a.text(0.57, 0.01,
           f'Prescribed = any value other than "No"  |  n = {len(df):,} patients',
           ha='center', fontsize=8, color='#888', style='italic')

fig9a.savefig(OUT / 'step9_medication_prescription_rate.png',
              dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: step9_medication_prescription_rate.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 9b — Key Medication Readmission — Grouped Bar (Tableau Classic)
# ═══════════════════════════════════════════════════════════════════════════
key_meds = ['insulin', 'metformin', 'glipizide']

fig9b, axes9b = plt.subplots(1, 3, figsize=(18, 7), facecolor='white')
fig9b.subplots_adjust(left=0.06, right=0.97,
                      top=0.84, bottom=0.20, wspace=0.38)

for ax, med in zip(axes9b, key_meds):
    ct      = pd.crosstab(df[med], df['readmitted'], normalize='index') * 100
    ct_plot = ct[CLASS_ORDER]

    categories_med = ct_plot.index.tolist()
    n_cls_med      = len(CLASS_ORDER)

    group_w_med = 0.70
    bar_w_med   = group_w_med / n_cls_med
    offsets_med = np.linspace(-group_w_med/2 + bar_w_med/2,
                               group_w_med/2 - bar_w_med/2, n_cls_med)
    x_med       = np.arange(len(categories_med))

    for i, col_name in enumerate(CLASS_ORDER):
        color = MED_CLASS_COLORS[col_name]
        vals  = ct_plot[col_name].values
        xpos  = x_med + offsets_med[i]
        bars  = ax.bar(xpos, vals,
                       width=bar_w_med - 0.03,
                       color=color,
                       edgecolor='white',
                       linewidth=0.9)
        for bar, val in zip(bars, vals):
            if val >= 8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.6,
                    f'{val:.1f}%',
                    ha='center', va='bottom',
                    fontsize=8.0, fontweight='bold',
                    color=color
                )

    ax.set_xticks(x_med)
    ax.set_xticklabels(categories_med, rotation=0, fontsize=9)
    ax.set_ylim(0, ct_plot.values.max() * 1.30)
    ax.set_title(f'Readmission by\n{med.capitalize()} Status',
                 fontsize=11, fontweight='bold', color='#1A1A2E', pad=10)
    ax.set_ylabel('Percentage of Patients (%)', fontsize=9,
                  color='#444', labelpad=5)
    ax.set_xlabel(f'{med.capitalize()} Dosage Status', fontsize=9,
                  color='#444', labelpad=5)
    ax.tick_params(axis='x', rotation=0, labelsize=9, length=0)
    ax.tick_params(axis='y', labelsize=8, length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#C5D0DC')
    ax.spines['bottom'].set_color('#C5D0DC')
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    for j, cat in enumerate(categories_med):
        n = (df[med] == cat).sum()
        ax.text(j, ct_plot.values.max() * 1.22, f'n={n:,}',
                ha='center', va='bottom', fontsize=7.5, color='#666')

fig9b.suptitle('Step 9  |  Readmission Rate by Key Medication Status',
               fontsize=13, fontweight='bold', color='#1A1A2E', y=0.97)

med_handles = [
    mpatches.Patch(facecolor=MED_CLASS_COLORS[c],
                   edgecolor='white', label=MED_CLASS_LABELS[c])
    for c in CLASS_ORDER
]
fig9b.legend(handles=med_handles,
             loc='lower center', bbox_to_anchor=(0.5, 0.04),
             ncol=3, fontsize=10,
             frameon=True, facecolor='white',
             edgecolor='#DDDDDD', framealpha=1.0,
             handlelength=1.4, handleheight=1.0,
             columnspacing=2.0, borderpad=0.8)

fig9b.savefig(OUT / 'step9_key_medications_readmission.png',
              dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: step9_key_medications_readmission.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 10 — Correlation Heatmap — Numerical Features
# ═══════════════════════════════════════════════════════════════════════════
num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses']

corr_matrix = df[num_cols].corr()

cmap = LinearSegmentedColormap.from_list(
    'elegant_div',
    ['#0D4A45', '#3A8A7A', '#F5F7FA', '#B05A5A', '#6B1A1A'],
    N=256
)

readable = {
    'time_in_hospital':   'Time in\nHospital',
    'num_lab_procedures': 'Lab\nProcedures',
    'num_procedures':     'Procedures',
    'num_medications':    'Medications',
    'number_outpatient':  'Outpatient\nVisits',
    'number_emergency':   'Emergency\nVisits',
    'number_inpatient':   'Inpatient\nVisits',
    'number_diagnoses':   'Diagnoses\nCount',
}
labels = [readable[c] for c in num_cols]

fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
fig.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.14)

sns.heatmap(
    corr_matrix,
    annot=True, fmt='.2f', cmap=cmap,
    vmin=-1, vmax=1, square=True,
    linewidths=1.2, linecolor='white',
    annot_kws={'size': 9, 'weight': 'bold', 'color': '#0D1B2A'},
    cbar_kws={'shrink': 0.78, 'pad': 0.03},
    ax=ax, xticklabels=labels, yticklabels=labels,
)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=8, length=0)
cbar.set_label('Pearson Correlation', fontsize=8.5, color='#2C3E50', labelpad=8)
cbar.outline.set_edgecolor('#D1DCE5')

ax.set_xticklabels(labels, fontsize=8.5, color='#2C3E50', rotation=0)
ax.set_yticklabels(labels, fontsize=8.5, color='#2C3E50', rotation=0)
ax.tick_params(length=0)
for side in ['top', 'right', 'left', 'bottom']:
    ax.spines[side].set_visible(False)

fig.text(0.015, 0.96,
         'Step 10  |  Correlation Heatmap — Numerical Features',
         fontsize=13, fontweight='bold', color='#0D1B2A')

neg_patch  = mpatches.Patch(facecolor='#0D4A45', edgecolor='white',
                             linewidth=0.5, label='Strong Negative  (–1.0)')
zero_patch = mpatches.Patch(facecolor='#F5F7FA', edgecolor='#CCCCCC',
                             linewidth=0.5, label='No Correlation  (0.0)')
pos_patch  = mpatches.Patch(facecolor='#6B1A1A', edgecolor='white',
                             linewidth=0.5, label='Strong Positive  (+1.0)')

fig.legend(
    handles=[neg_patch, zero_patch, pos_patch],
    loc='lower center', bbox_to_anchor=(0.48, 0.01),
    ncol=3, fontsize=8.5,
    frameon=True, facecolor='white', edgecolor='#D1DCE5',
    framealpha=1.0, handlelength=1.8, handleheight=1.0,
    columnspacing=1.8, borderpad=0.8,
)

fig.savefig(OUT / 'step10_correlation_heatmap.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: step10_correlation_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 12 — Age Analysis (Grouped Bar + Lollipop)
# ═══════════════════════════════════════════════════════════════════════════
SPINE_CLR = '#DDE4EA'
TITLE_CLR = '#1A2535'
LABEL_CLR = '#3A4A5A'
TICK_CLR  = '#5A6A7A'

plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9})

CLASS_ORDER_12  = ['<30', '>30', 'NO']
CLASS_COLORS_12 = {
    '<30': '#C0392B',
    '>30': '#2E6EA6',
    'NO':  '#7A9E6E',
}
CLASS_LABELS_12 = {
    '<30': 'Readmitted < 30 days  (High Risk)',
    '>30': 'Readmitted > 30 days',
    'NO':  'Not Readmitted',
}

age_readmit = pd.crosstab(df['age'], df['readmitted'],
                           normalize='index') * 100
age_readmit = age_readmit.reindex(age_order)

fig1, ax1 = plt.subplots(figsize=(14, 7), facecolor=BG)
fig1.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.24)

n_ages    = len(age_order)
n_cls12   = len(CLASS_ORDER_12)
group_w12 = 0.72
bar_w12   = group_w12 / n_cls12
offsets12 = np.linspace(-group_w12/2 + bar_w12/2,
                         group_w12/2 - bar_w12/2, n_cls12)
x12 = np.arange(n_ages)

for i, cls in enumerate(CLASS_ORDER_12):
    if cls not in age_readmit.columns:
        continue
    vals = age_readmit[cls].values
    xpos = x12 + offsets12[i]
    bars = ax1.bar(xpos, vals,
                   width=bar_w12 - 0.03,
                   color=CLASS_COLORS_12[cls],
                   edgecolor='white', linewidth=1.0, zorder=3,
                   label=CLASS_LABELS_12[cls])
    for bar, v in zip(bars, vals):
        if v >= 7:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{v:.1f}%',
                ha='center', va='bottom',
                fontsize=6.5, fontweight='bold',
                color=CLASS_COLORS_12[cls], zorder=5
            )

ax1.set_xticks(x12)
ax1.set_xticklabels(age_order, rotation=40, ha='right',
                    fontsize=9, color=TICK_CLR)
ax1.set_ylim(0, age_readmit.values.max() * 1.30)
ax1.set_ylabel('Percentage of Patients (%)', fontsize=10,
               color=LABEL_CLR, labelpad=6)
ax1.set_xlabel('Age Group', fontsize=10, color=LABEL_CLR, labelpad=6)
ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{int(v)}%'))
ax1.tick_params(axis='both', length=0)
ax1.spines['left'].set_color(SPINE_CLR)
ax1.spines['bottom'].set_color(SPINE_CLR)

for j, age in enumerate(age_order):
    n = (df['age'] == age).sum()
    ax1.text(j, age_readmit.values.max() * 1.22,
             f'n={n:,}', ha='center', va='bottom',
             fontsize=7, color='#888')

fig1.text(0.015, 0.94,
          'Step 12  |  Readmission Rate by Age Group',
          fontsize=13, fontweight='bold', color=TITLE_CLR)
fig1.text(0.015, 0.905,
          'Grouped bars — each cluster = one age group  |  bars show readmission class breakdown',
          fontsize=8.5, color='#5A6A7A', style='italic')

handles12 = [
    mpatches.Patch(facecolor=CLASS_COLORS_12[c], edgecolor='white',
                   linewidth=0.5, label=CLASS_LABELS_12[c])
    for c in CLASS_ORDER_12
]
fig1.legend(handles=handles12,
            loc='lower center', bbox_to_anchor=(0.5, 0.02),
            ncol=3, fontsize=9,
            frameon=True, facecolor='white',
            edgecolor=SPINE_CLR, framealpha=1.0,
            handlelength=1.6, handleheight=1.0,
            columnspacing=2.0, borderpad=0.8)

fig1.savefig(OUT / 'step12_age_readmission.png',
             dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: step12_age_readmission.png")

# Lollipop
medians = (df.groupby('age', observed=True)['time_in_hospital']
             .median().reindex(age_order))
means   = (df.groupby('age', observed=True)['time_in_hospital']
             .mean().reindex(age_order))

LOLLIPOP_COLOR = '#5B2D8E'
MEAN_COLOR     = '#D4527A'
STEM_COLOR     = '#A07BC0'

fig2, ax2 = plt.subplots(figsize=(13, 7), facecolor=BG)
fig2.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.22)

x = np.arange(len(age_order))

ax2.vlines(x, 0, medians.values,
           color=STEM_COLOR, linewidth=2.0, alpha=0.6, zorder=2)
ax2.scatter(x, medians.values,
            color=LOLLIPOP_COLOR, s=120, zorder=4,
            edgecolors='white', linewidths=1.5, label='Median stay')
ax2.scatter(x, means.values,
            color=MEAN_COLOR, s=60, marker='D', zorder=5,
            edgecolors='white', linewidths=1.0, label='Mean stay')

for j, (med, mn) in enumerate(zip(medians.values, means.values)):
    ax2.text(j, med + 0.15, f'{med:.1f}d',
             ha='center', va='bottom',
             fontsize=8.5, fontweight='bold', color=LOLLIPOP_COLOR)
    ax2.text(j, mn - 0.25, f'{mn:.1f}',
             ha='center', va='top',
             fontsize=7, color=MEAN_COLOR)

overall_med = df['time_in_hospital'].median()
ax2.axhline(overall_med, color='#AAAAAA', linewidth=1.2, linestyle='--', zorder=1)
ax2.text(len(age_order) - 0.3, overall_med + 0.1,
         f'Overall median = {overall_med:.0f}d',
         ha='right', va='bottom', fontsize=8, color='#888', style='italic')

ax2.set_xticks(x)
ax2.set_xticklabels(age_order, rotation=40, ha='right',
                    fontsize=9, color=TICK_CLR)
ax2.set_ylabel('Time in Hospital (days)', fontsize=10,
               color=LABEL_CLR, labelpad=6)
ax2.set_xlabel('Age Group', fontsize=10, color=LABEL_CLR, labelpad=6)
ax2.set_ylim(0, medians.max() + 1.5)
ax2.tick_params(axis='both', length=0)
ax2.spines['left'].set_color(SPINE_CLR)
ax2.spines['bottom'].set_color(SPINE_CLR)

ax2.legend(fontsize=9, frameon=False, loc='upper left',
           handlelength=0.8, labelcolor=LABEL_CLR)

fig2.text(0.015, 0.94,
          'Step 12  |  Length of Hospital Stay by Age Group',
          fontsize=13, fontweight='bold', color=TITLE_CLR)
fig2.text(0.015, 0.905,
          'Lollipop chart — violet dot = median  ·  rose diamond = mean  ·  dashed = overall median',
          fontsize=8.5, color='#5A6A7A', style='italic')

fig2.savefig(OUT / 'step12_age_hospital_stay.png',
             dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: step12_age_hospital_stay.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 14 — Patient Encounter Frequency
# 🟡 FIX: Added caveat note — analysis on full dataset, split in Code 2
# ═══════════════════════════════════════════════════════════════════════════
patient_counts = df['patient_nbr'].value_counts()
df['encounter_count']    = df['patient_nbr'].map(patient_counts)
df['multiple_encounters'] = (df['encounter_count'] >= 2).astype(int)

multiple_pct = (df['multiple_encounters'].sum() / len(df)) * 100
print(f"\nPatients with multiple encounters: {multiple_pct:.2f}%")

readmit_by_encounters = df.groupby('multiple_encounters')['target_binary'].mean() * 100
print(readmit_by_encounters)

CLASS_COLORS_14 = {
    '>30': '#4A6FA5',
    '<30': '#D85A30',
}

fig, ax = plt.subplots(figsize=(7, 5))
x_labels = ['Single Encounter', 'Multiple Encounters']
colors   = [CLASS_COLORS_14['>30'], CLASS_COLORS_14['<30']]
bars     = ax.bar(x_labels, readmit_by_encounters.values,
                  color=colors, edgecolor='white', linewidth=0.8,
                  width=0.55, zorder=3)

for bar, val in zip(bars, readmit_by_encounters.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f'{val:.1f}%', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#2C2C2A')

ax.set_title('Step 14 | 30-Day Readmission by Encounter Frequency',
             fontsize=13, fontweight='bold', pad=14, color='#2C2C2A')
ax.set_ylabel('Readmission <30 (%)', fontsize=11, labelpad=8, color='#444441')
ax.set_xlabel('Encounter Type', fontsize=11, labelpad=8, color='#444441')
ax.tick_params(labelsize=10, colors='#5F5E5A')
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='#B4B2A9')
ax.set_axisbelow(True)
ax.grid(axis='x', visible=False)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(0.6)
ax.spines[['left', 'bottom']].set_color('#B4B2A9')

handles14 = [
    mpatches.Patch(color=CLASS_COLORS_14['>30'], label='Single Encounter'),
    mpatches.Patch(color=CLASS_COLORS_14['<30'], label='Multiple Encounters'),
]
ax.legend(handles=handles14, loc='upper left', frameon=True,
          framealpha=0.9, edgecolor='#D3D1C7', fontsize=9)

# 🟡 FIX: Caveat note — unsplit data, patient-level split in Code 2
fig.text(0.5, -0.02,
         'Note: Analysis on full dataset — patient-level GroupShuffleSplit applied in Code 2',
         ha='center', fontsize=7.5, color='#888', style='italic')

plt.tight_layout()
plt.grid(False)
plt.savefig(OUT / 'step14_duplicate_patient_analysis.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: step14_duplicate_patient_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 16 — Medical Specialty Readmission
# ═══════════════════════════════════════════════════════════════════════════
CLASS_ORDER_16  = ['<30', '>30', 'NO']
CLASS_COLORS_16 = {
    '<30': '#6A4C9C',
    '>30': '#2196C4',
    'NO':  '#3DAA6E',
}
CLASS_LABELS_16 = {
    '<30': 'Readmitted < 30 days  (High Risk)',
    '>30': 'Readmitted > 30 days',
    'NO':  'Not Readmitted',
}

BG_16    = '#FFFFFF'
SPINE_16 = '#DDE4EA'
TITLE_16 = '#0A2342'
LABEL_16 = '#2C3E50'
TICK_16  = '#4A5568'

specialty_readmit = pd.crosstab(
    df['medical_specialty_grouped'],
    df['readmitted'],
    normalize='index'
) * 100
specialty_readmit = specialty_readmit.sort_values('<30', ascending=False)

categories = specialty_readmit.index.tolist()
n_cats     = len(categories)
n_cls      = len(CLASS_ORDER_16)

group_w16 = 0.70
bar_w16   = group_w16 / n_cls
offsets16 = np.linspace(-group_w16/2 + bar_w16/2,
                         group_w16/2 - bar_w16/2, n_cls)
x16       = np.arange(n_cats)

fig16, ax16 = plt.subplots(figsize=(16, 7), facecolor=BG_16)
fig16.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.28)

ax16.set_facecolor(BG_16)
ax16.yaxis.grid(False)
ax16.xaxis.grid(False)

for i, cls in enumerate(CLASS_ORDER_16):
    if cls not in specialty_readmit.columns:
        continue
    vals = specialty_readmit[cls].values
    xpos = x16 + offsets16[i]
    bars = ax16.bar(xpos, vals,
                    width=bar_w16 - 0.03,
                    color=CLASS_COLORS_16[cls],
                    edgecolor='white', linewidth=0.8,
                    zorder=3, label=CLASS_LABELS_16[cls])
    for bar, v in zip(bars, vals):
        if v >= 5:
            ax16.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f'{v:.1f}%',
                ha='center', va='bottom',
                fontsize=6.5, fontweight='bold',
                color=CLASS_COLORS_16[cls], zorder=5
            )

for j, cat in enumerate(categories):
    n = (df['medical_specialty_grouped'] == cat).sum()
    ax16.text(j, -8, f'n={n:,}',
              ha='center', va='top', fontsize=6.8, color='#888888')

ax16.set_xticks(x16)
ax16.set_xticklabels(categories, rotation=40, ha='right',
                     fontsize=8.5, color=TICK_16)
ax16.set_ylim(0, specialty_readmit.values.max() * 1.28)
ax16.set_ylabel('Percentage of Patients (%)', fontsize=10,
                color=LABEL_16, labelpad=6)
ax16.set_xlabel('Medical Specialty', fontsize=10, color=LABEL_16, labelpad=8)
ax16.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v)}%'))
ax16.tick_params(axis='both', length=0)
for side in ['top', 'right', 'left']:
    ax16.spines[side].set_visible(False)
ax16.spines['bottom'].set_color(SPINE_16)
ax16.spines['bottom'].set_linewidth(0.8)

fig16.text(0.015, 0.94,
           'Step 16  |  Readmission Rate by Medical Specialty',
           fontsize=13, fontweight='bold', color=TITLE_16)
fig16.text(0.015, 0.906,
           'Grouped bars — Top 10 specialties + Other  |  sorted by highest early readmission',
           fontsize=8.5, color='#5A6A7A', style='italic')

handles16 = [
    mpatches.Patch(facecolor=CLASS_COLORS_16[c], edgecolor='white',
                   linewidth=0.5, label=CLASS_LABELS_16[c])
    for c in CLASS_ORDER_16
]
fig16.legend(handles=handles16,
             loc='lower center', bbox_to_anchor=(0.5, 0.01),
             ncol=3, fontsize=9,
             frameon=True, facecolor='white', edgecolor=SPINE_16,
             framealpha=1.0, handlelength=1.6, handleheight=1.0,
             columnspacing=2.0, borderpad=0.8)

fig16.savefig(OUT / 'step16_medical_specialty_readmission.png',
              dpi=300, bbox_inches='tight', facecolor=BG_16)
plt.close()
print("Saved: step16_medical_specialty_readmission.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 20 — Multivariate Heatmaps
# ═══════════════════════════════════════════════════════════════════════════
BG_20    = '#FFFFFF'
TITLE_20 = '#1A2535'
LABEL_20 = '#2C3E50'
TICK_20  = '#4A5568'
SPINE_20 = '#DDE4EA'

df['time_cat'] = pd.cut(
    df['time_in_hospital'],
    bins=[0, 3, 6, 9, 12, 100],
    labels=['0–3 days', '4–6 days', '7–9 days', '10–12 days', '13+ days']
)

pivot1 = pd.crosstab(
    df['age'], df['time_cat'],
    values=df['readmitted'].apply(lambda x: 1 if x == '<30' else 0),
    aggfunc='mean'
) * 100
pivot1 = pivot1.fillna(0)

fig1, ax1 = plt.subplots(figsize=(12, 8), facecolor=BG_20)
fig1.subplots_adjust(left=0.12, right=0.92, top=0.88, bottom=0.14)

sns.heatmap(
    pivot1, annot=True, fmt='.1f', cmap='YlOrRd',
    vmin=0, vmax=pivot1.values.max(),
    linewidths=1.5, linecolor='white',
    annot_kws={'size': 10, 'weight': 'bold', 'color': '#2C1A00'},
    cbar_kws={'shrink': 0.75, 'pad': 0.03}, ax=ax1
)

cbar1 = ax1.collections[0].colorbar
cbar1.set_label('30-Day Readmission Rate (%)', fontsize=9, color=LABEL_20, labelpad=8)
cbar1.ax.tick_params(labelsize=8, length=0)
cbar1.outline.set_edgecolor(SPINE_20)

ax1.set_xlabel('Length of Hospital Stay', fontsize=11, color=LABEL_20, labelpad=8)
ax1.set_ylabel('Age Group', fontsize=11, color=LABEL_20, labelpad=8)
ax1.tick_params(axis='both', length=0, labelsize=9)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, color=TICK_20)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, color=TICK_20)
for side in ['top', 'right', 'left', 'bottom']:
    ax1.spines[side].set_visible(False)

fig1.text(0.015, 0.94,
          'Step 20  |  Age × Length of Stay → 30-Day Readmission Rate',
          fontsize=13, fontweight='bold', color=TITLE_20)
fig1.text(0.015, 0.906,
          'Each cell = % of patients readmitted within 30 days  |  darker = higher readmission risk',
          fontsize=8.5, color='#5A6A7A', style='italic')

fig1.savefig(OUT / 'step20_multivariate_age_time_heatmap.png',
             dpi=300, bbox_inches='tight', facecolor=BG_20)
plt.close()
print("Saved: step20_multivariate_age_time_heatmap.png")

pivot2 = pd.crosstab(
    df['admission_type_id'], df['age'],
    values=df['readmitted'].apply(lambda x: 1 if x == '<30' else 0),
    aggfunc='mean'
) * 100
pivot2 = pivot2.fillna(0)

fig2, ax2 = plt.subplots(figsize=(14, 7), facecolor=BG_20)
fig2.subplots_adjust(left=0.10, right=0.93, top=0.88, bottom=0.14)

sns.heatmap(
    pivot2, annot=True, fmt='.1f', cmap='YlGnBu',
    vmin=0, vmax=pivot2.values.max(),
    linewidths=1.5, linecolor='white',
    annot_kws={'size': 10, 'weight': 'bold', 'color': '#0A2535'},
    cbar_kws={'shrink': 0.75, 'pad': 0.03}, ax=ax2
)

cbar2 = ax2.collections[0].colorbar
cbar2.set_label('30-Day Readmission Rate (%)', fontsize=9, color=LABEL_20, labelpad=8)
cbar2.ax.tick_params(labelsize=8, length=0)
cbar2.outline.set_edgecolor(SPINE_20)

ax2.set_xlabel('Age Group', fontsize=11, color=LABEL_20, labelpad=8)
ax2.set_ylabel('Admission Type ID', fontsize=11, color=LABEL_20, labelpad=8)
ax2.tick_params(axis='both', length=0, labelsize=9)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right', color=TICK_20)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, color=TICK_20)
for side in ['top', 'right', 'left', 'bottom']:
    ax2.spines[side].set_visible(False)

fig2.text(0.015, 0.94,
          'Step 20  |  Admission Type × Age → 30-Day Readmission Rate',
          fontsize=13, fontweight='bold', color=TITLE_20)
fig2.text(0.015, 0.906,
          'Each cell = % of patients readmitted within 30 days  |  darker = higher readmission risk',
          fontsize=8.5, color='#5A6A7A', style='italic')

fig2.savefig(OUT / 'step20_multivariate_admit_age_heatmap.png',
             dpi=300, bbox_inches='tight', facecolor=BG_20)
plt.close()
print("Saved: step20_multivariate_admit_age_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 21 — Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 50)
print("STEP 21: Statistical Tests")
print("=" * 50)

cat_features = ['gender', 'race', 'age', 'admission_type_id',
                'discharge_disposition_id', 'change', 'diabetesMed']

print("\n--- Chi-square Test (Categorical vs Readmitted) ---")
for col in cat_features:
    contingency = pd.crosstab(df[col], df['readmitted'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"{col:25} | chi2={chi2:.2f} | p-value={p:.4f} | "
          f"{'Significant' if p < 0.05 else 'Not significant'}")

num_features_stat = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                     'num_medications', 'number_outpatient', 'number_emergency',
                     'number_inpatient', 'number_diagnoses']
X_LABELS = ['Time in\nHospital', 'Lab\nProcedures', 'Procedures', 'Medications',
            'Outpatient\nVisits', 'Emergency\nVisits', 'Inpatient\nAdmissions', 'Diagnoses\nCount']

print("\n--- ANOVA & Kruskal-Wallis (Numerical vs Binary Target) ---")
p_values, f_stats, h_stats = [], [], []

for col in num_features_stat:
    g0 = df[df['target_binary'] == 0][col]
    g1 = df[df['target_binary'] == 1][col]

    try:
        f_stat, p_anova = f_oneway(g0, g1)
        if np.isnan(f_stat) or np.isinf(f_stat):
            f_stat, p_anova = 0.0, 1.0
    except Exception:
        f_stat, p_anova = 0.0, 1.0

    try:
        if g0.nunique() <= 1 and g1.nunique() <= 1:
            raise ValueError("Constant input")
        h_stat, p_kw = kruskal(g0, g1)
    except Exception:
        h_stat, p_kw = 0.0, 1.0
        print(f"{col:25} | SKIPPED — all values identical")
        p_values.append(p_kw)
        f_stats.append(f_stat)
        h_stats.append(h_stat)
        continue

    p_values.append(p_kw)
    f_stats.append(f_stat)
    h_stats.append(h_stat)
    print(f"{col:25} | ANOVA p={p_anova:.4f} | Kruskal p={p_kw:.4f} | "
          f"{'Significant' if p_kw < 0.05 else 'Not'}")

BG_21    = '#FFFFFF'
SPINE_21 = '#E2E8EF'
TITLE_21 = '#2D2D2D'
LABEL_21 = '#3D3D3D'
TICK_21  = '#5A5A6A'
BAR_COLORS = ['#C0392B' if p < 0.05 else '#A0A8B0' for p in p_values]

fig1, ax1 = plt.subplots(figsize=(13, 6), facecolor=BG_21)
fig1.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.18)

x1   = np.arange(len(num_features_stat))
bars = ax1.bar(x1, [max(p, 1e-10) for p in p_values],
               width=0.52, color=BAR_COLORS,
               edgecolor='white', linewidth=1.0, zorder=3)
ax1.axhline(y=0.05, color='#E67E22', linewidth=1.8, linestyle='--', zorder=4)

for bar, p in zip(bars, p_values):
    label = '<0.0001' if p < 0.0001 else ('N/A' if p == 1.0 else f'{p:.4f}')
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() * 1.5, label,
             ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#2D2D2D')

ax1.set_yscale('log')
ax1.set_xticks(x1)
ax1.set_xticklabels(X_LABELS, fontsize=9, color=TICK_21)
ax1.set_ylabel('p-value  (log scale)', fontsize=10, color=LABEL_21, labelpad=6)
ax1.tick_params(axis='both', length=0, labelsize=9)
ax1.yaxis.grid(False)
ax1.xaxis.grid(False)
for side in ['top', 'right', 'left']:
    ax1.spines[side].set_visible(False)
ax1.spines['bottom'].set_color(SPINE_21)
ax1.spines['bottom'].set_linewidth(0.8)

sig_patch   = mpatches.Patch(facecolor='#C0392B', edgecolor='white',
                              label='Significant  (p < 0.05)')
nosig_patch = mpatches.Patch(facecolor='#A0A8B0', edgecolor='white',
                              label='Not significant / Skipped')
thresh_line = plt.Line2D([0], [0], color='#E67E22', linewidth=1.8,
                         linestyle='--', label='Threshold  (p = 0.05)')
fig1.legend(handles=[sig_patch, nosig_patch, thresh_line],
            loc='lower center', bbox_to_anchor=(0.5, 0.01),
            ncol=3, fontsize=9, frameon=True, facecolor='white',
            edgecolor=SPINE_21, framealpha=1.0,
            handlelength=1.6, handleheight=1.0,
            columnspacing=2.0, borderpad=0.8)
fig1.text(0.015, 0.94,
          'Step 21  |  Statistical Significance — Kruskal-Wallis Test',
          fontsize=13, fontweight='bold', color=TITLE_21)
fig1.text(0.015, 0.906,
          'p-values on log scale  |  red = significant  |  grey = not significant  |  orange = threshold',
          fontsize=8.5, color='#7A7A8A', style='italic')
fig1.savefig(OUT / 'step21_statistical_pvalues.png',
             dpi=300, bbox_inches='tight', facecolor=BG_21)
plt.close()
print("Saved: step21_statistical_pvalues.png")

COLOR_ANOVA   = '#5B8DB8'
COLOR_KRUSKAL = '#B85B5B'

fig2, ax2 = plt.subplots(figsize=(13, 6), facecolor=BG_21)
fig2.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.18)

x2    = np.arange(len(num_features_stat))
width = 0.38

bars_f = ax2.bar(x2 - width/2, f_stats, width,
                 color=COLOR_ANOVA, edgecolor='white', linewidth=1.0, zorder=3)
bars_h = ax2.bar(x2 + width/2, h_stats, width,
                 color=COLOR_KRUSKAL, edgecolor='white', linewidth=1.0, zorder=3)

for bar, v in zip(bars_f, f_stats):
    if v > 0:
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(f_stats)*0.01,
                 f'{v:.0f}', ha='center', va='bottom',
                 fontsize=7.5, fontweight='bold', color=COLOR_ANOVA)
for bar, v in zip(bars_h, h_stats):
    if v > 0:
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(f_stats)*0.01,
                 f'{v:.0f}', ha='center', va='bottom',
                 fontsize=7.5, fontweight='bold', color=COLOR_KRUSKAL)

ax2.set_xticks(x2)
ax2.set_xticklabels(X_LABELS, fontsize=9, color=TICK_21)
ax2.set_ylim(0, max(max(f_stats), max(h_stats)) * 1.20)
ax2.set_ylabel('Test Statistic Value', fontsize=10, color=LABEL_21, labelpad=6)
ax2.tick_params(axis='both', length=0, labelsize=9)
ax2.yaxis.grid(False)
ax2.xaxis.grid(False)
for side in ['top', 'right', 'left']:
    ax2.spines[side].set_visible(False)
ax2.spines['bottom'].set_color(SPINE_21)
ax2.spines['bottom'].set_linewidth(0.8)

handles21 = [
    mpatches.Patch(facecolor=COLOR_ANOVA,   edgecolor='white', label='ANOVA  F-statistic'),
    mpatches.Patch(facecolor=COLOR_KRUSKAL, edgecolor='white', label='Kruskal-Wallis  H-statistic'),
]
fig2.legend(handles=handles21, loc='lower center', bbox_to_anchor=(0.5, 0.01),
            ncol=2, fontsize=9, frameon=True, facecolor='white',
            edgecolor=SPINE_21, framealpha=1.0,
            handlelength=1.6, handleheight=1.0,
            columnspacing=2.5, borderpad=0.8)
fig2.text(0.015, 0.94,
          'Step 21  |  ANOVA vs Kruskal-Wallis — Test Statistics Comparison',
          fontsize=13, fontweight='bold', color=TITLE_21)
fig2.text(0.015, 0.906,
          'Grouped bars — blue = ANOVA F-stat  |  red = Kruskal H-stat  |  0 = skipped',
          fontsize=8.5, color='#7A7A8A', style='italic')
fig2.savefig(OUT / 'step21_anova_vs_kruskal_stats.png',
             dpi=300, bbox_inches='tight', facecolor=BG_21)
plt.close()
print("Saved: step21_anova_vs_kruskal_stats.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 24 — Admission Source
# ═══════════════════════════════════════════════════════════════════════════
CLASS_COLORS_24 = {
    '<30': '#D85A30',
    '>30': '#4A6FA5',
    'NO':  '#1D9E75',
}
CLASS_LABELS_24 = {
    '<30': 'Readmitted < 30 days',
    '>30': 'Readmitted > 30 days',
    'NO':  'Not Readmitted',
}

source_readmit = pd.crosstab(df['admission_source_id'], df['readmitted'],
                              normalize='index') * 100
source_readmit = source_readmit.sort_values('<30', ascending=False)

categories = source_readmit.index.tolist()
n_cats     = len(categories)
n_cls      = len(CLASS_ORDER)
group_w    = 0.7
bar_w      = group_w / n_cls
offsets    = np.linspace(-group_w/2 + bar_w/2, group_w/2 - bar_w/2, n_cls)
x          = np.arange(n_cats)

fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.28)

ax.set_facecolor(BG)
ax.yaxis.grid(False)
ax.xaxis.grid(False)

for i, cls in enumerate(CLASS_ORDER):
    if cls not in source_readmit.columns:
        continue
    vals = source_readmit[cls].values
    xpos = x + offsets[i]
    bars = ax.bar(xpos, vals,
                  width=bar_w - 0.02,
                  color=CLASS_COLORS_24[cls],
                  edgecolor='white', linewidth=0.8, zorder=3,
                  label=CLASS_LABELS_24[cls])
    for bar, v in zip(bars, vals):
        if v >= 3:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{v:.1f}%',
                    ha='center', va='bottom',
                    fontsize=7, fontweight='bold',
                    color=CLASS_COLORS_24[cls], zorder=5)

for j, cat in enumerate(categories):
    n = (df['admission_source_id'] == cat).sum()
    ax.text(j, -3.5, f'n={n:,}', ha='center', va='top',
            fontsize=7, color='#888888')

ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=40, ha='right', fontsize=8.5)
ax.set_ylim(0, source_readmit.values.max() * 1.18)
ax.set_ylabel('Percentage of Patients (%)', fontsize=10, labelpad=6)
ax.set_xlabel('Admission Source ID', fontsize=10, labelpad=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v)}%'))
ax.tick_params(axis='both', length=0)
for side in ['top', 'right', 'left']:
    ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color(SPINE_CLR)
ax.spines['bottom'].set_linewidth(0.8)

fig.text(0.015, 0.94,
         'Step 24  |  Readmission Rate by Admission Source',
         fontsize=13, fontweight='bold', color='#1A2535')
fig.text(0.015, 0.906,
         'Grouped bars — sorted by highest <30 day readmission rate',
         fontsize=8.5, color='#5A6A7A', style='italic')

handles24 = [
    mpatches.Patch(facecolor=CLASS_COLORS_24[c], edgecolor='white',
                   linewidth=0.5, label=CLASS_LABELS_24[c])
    for c in CLASS_ORDER
]
fig.legend(handles=handles24,
           loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=3, fontsize=9,
           frameon=True, facecolor='white', edgecolor=SPINE_CLR,
           framealpha=1.0, handlelength=1.6, handleheight=1.0,
           columnspacing=2.0, borderpad=0.8)

plt.tight_layout()
plt.savefig(OUT / 'step24_admission_source_readmission.png',
            dpi=300, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: step24_admission_source_readmission.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 26 — Lab / Diagnoses Ratio
# ═══════════════════════════════════════════════════════════════════════════
CLASS_COLORS_26 = {
    '<30': '#D85A30',
    '>30': '#4A6FA5',
    'NO':  '#1D9E75',
}
CLASS_LABELS_26 = {
    '<30': 'Readmitted < 30 days',
    '>30': 'Readmitted > 30 days',
    'NO':  'Not Readmitted',
}

df['lab_diag_ratio'] = df['num_lab_procedures'] / (df['number_diagnoses'] + 1e-6)
ratio_capped         = df['lab_diag_ratio'].clip(upper=df['lab_diag_ratio'].quantile(0.95))
df['ratio_capped']   = ratio_capped

fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(
    data=df, x='readmitted', y='ratio_capped', order=CLASS_ORDER,
    palette={c: CLASS_COLORS_26[c] for c in CLASS_ORDER},
    width=0.5, linewidth=1.2,
    flierprops=dict(marker='o', markerfacecolor='#888780', markersize=3,
                    alpha=0.4, linestyle='none'),
    medianprops=dict(color='white', linewidth=2.5),
    boxprops=dict(edgecolor='white', linewidth=0.8),
    whiskerprops=dict(linewidth=1.0, color='#5F5E5A'),
    capprops=dict(linewidth=1.2, color='#5F5E5A'),
    ax=ax
)

for i, cls in enumerate(CLASS_ORDER):
    med = df.loc[df['readmitted'] == cls, 'ratio_capped'].median()
    ax.text(i, med + 0.15, f'{med:.2f}',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#2C2C2A')

ax.set_title('Step 26 | Lab Procedures / Diagnoses Ratio by Readmission',
             fontsize=13, fontweight='bold', pad=14, color='#2C2C2A')
ax.set_xlabel('Readmission Status', fontsize=11, labelpad=8, color='#444441')
ax.set_ylabel('Ratio (capped at 95th percentile)',
              fontsize=11, labelpad=8, color='#444441')
ax.set_xticklabels([CLASS_LABELS_26[c] for c in CLASS_ORDER], fontsize=10)
ax.tick_params(axis='y', labelsize=9, colors='#5F5E5A')
ax.grid(False)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(0.6)
ax.spines[['left', 'bottom']].set_color('#B4B2A9')

handles26 = [mpatches.Patch(color=CLASS_COLORS_26[c], label=CLASS_LABELS_26[c])
             for c in CLASS_ORDER]
ax.legend(handles=handles26, loc='upper right', frameon=True,
          framealpha=0.9, edgecolor='#D3D1C7',
          fontsize=9, title='Readmission Status', title_fontsize=9)

plt.tight_layout()
plt.savefig(OUT / 'step26_lab_diag_ratio.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: step26_lab_diag_ratio.png")

df['ratio_cat'] = pd.cut(df['lab_diag_ratio'],
                         bins=[0, 5, 10, 20, 100],
                         labels=['Low (<5)', 'Medium (5-10)',
                                 'High (10-20)', 'Very High (20+)'])
ratio_readmit = pd.crosstab(df['ratio_cat'], df['readmitted'],
                             normalize='index') * 100

fig, ax = plt.subplots(figsize=(9, 5))
ratio_readmit[['<30', '>30', 'NO']].plot(
    kind='bar', stacked=True, ax=ax,
    color=[CLASS_COLORS_26[c] for c in CLASS_ORDER],
    edgecolor='white', linewidth=0.6
)

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', label_type='center',
                 fontsize=8, fontweight='bold', color='white')

ax.set_title('Step 26 | Readmission by Lab/Diagnoses Ratio Category',
             fontsize=13, fontweight='bold', pad=14, color='#2C2C2A')
ax.set_ylabel('Percentage (%)', fontsize=11, labelpad=8, color='#444441')
ax.set_xlabel('Lab/Diagnoses Ratio Category', fontsize=11, labelpad=8, color='#444441')
ax.tick_params(labelsize=10, colors='#5F5E5A')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(False)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(0.6)
ax.spines[['left', 'bottom']].set_color('#B4B2A9')

ax.legend().remove()
ax.legend(handles=handles26, loc='upper right', frameon=True,
          framealpha=0.9, edgecolor='#D3D1C7', fontsize=9)

plt.tight_layout()
plt.savefig(OUT / 'step26_lab_diag_ratio_grouped.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: step26_lab_diag_ratio_grouped.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 27 — Medication Change Depth
# ═══════════════════════════════════════════════════════════════════════════
CLASS_COLORS_27 = {
    '<30': '#D85A30',
    '>30': '#4A6FA5',
    'NO':  '#1D9E75',
}
CLASS_LABELS_27 = {
    '<30': 'Readmitted < 30 days',
    '>30': 'Readmitted > 30 days',
    'NO':  'Not Readmitted',
}

med_cols_27 = ['metformin', 'insulin', 'glipizide', 'glyburide', 'pioglitazone',
               'rosiglitazone', 'glimepiride', 'repaglinide', 'nateglinide']
# FIX: dynamic filter — kuch meds (jaise nateglinide) Code 1 mein already
# num_rare_meds_used mein consolidate ho kar drop ho chuke hain. Future-proof
# rehne ke liye sirf wahi columns rakho jo actually df mein maujood hain.
med_cols_27 = [c for c in med_cols_27 if c in df.columns]

def count_changes(row):
    return sum(1 for med in med_cols_27 if med in row.index and row[med] in ['Up', 'Down'])

df['med_change_depth'] = df[med_cols_27].apply(count_changes, axis=1)

df['depth_cat'] = pd.cut(df['med_change_depth'],
                         bins=[-1, 0, 2, 10],
                         labels=['0 changes', '1-2 changes', '3+ changes'])

depth_cat_order = ['0 changes', '1-2 changes', '3+ changes']
depth_readmit   = pd.crosstab(df['depth_cat'], df['readmitted'],
                               normalize='index') * 100
depth_readmit   = depth_readmit.reindex(depth_cat_order)

fig, ax = plt.subplots(figsize=(10, 6))

x          = np.arange(len(depth_cat_order))
width      = 0.25
multiplier = 0

for cls in CLASS_ORDER:
    vals   = depth_readmit[cls].values
    offset = width * multiplier
    bars   = ax.bar(x + offset, vals, width,
                    label=CLASS_LABELS_27[cls],
                    color=CLASS_COLORS_27[cls],
                    edgecolor='white', linewidth=0.8, zorder=3)
    for bar, v in zip(bars, vals):
        if v >= 5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f'{v:.1f}%', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold',
                    color=CLASS_COLORS_27[cls], zorder=5)
    multiplier += 1

for j, cat in enumerate(depth_cat_order):
    n = (df['depth_cat'] == cat).sum()
    ax.text(j, -4, f'n={n:,}', ha='center', va='top',
            fontsize=8, color='#888888')

ax.set_xticks(x + width)
ax.set_xticklabels(depth_cat_order, fontsize=10)
ax.set_ylim(0, depth_readmit.values.max() * 1.18)
ax.set_ylabel('Percentage of Patients (%)', fontsize=11, labelpad=8, color='#444441')
ax.set_xlabel('Medication Change Depth', fontsize=11, labelpad=8, color='#444441')
ax.set_title('Step 27 | Readmission by Medication Change Depth',
             fontsize=13, fontweight='bold', pad=14, color='#2C2C2A')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v)}%'))
ax.tick_params(axis='both', length=0, labelsize=9)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='#B4B2A9')
ax.set_axisbelow(True)
ax.grid(axis='x', visible=False)
for side in ['top', 'right', 'left']:
    ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color('#B4B2A9')
ax.spines['bottom'].set_linewidth(0.6)

handles27 = [mpatches.Patch(color=CLASS_COLORS_27[c], label=CLASS_LABELS_27[c])
             for c in CLASS_ORDER]
ax.legend(handles=handles27, loc='upper right', frameon=True,
          framealpha=0.9, edgecolor='#D3D1C7',
          fontsize=9, title='Readmission Status', title_fontsize=9)

plt.tight_layout()
plt.grid(False)
plt.savefig(OUT / 'step27_med_change_depth_grouped.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: step27_med_change_depth_grouped.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 28 — Polypharmacy Groups (Donut Charts)
# 🟡 FIX: Rebalanced bins — 0-9 / 10-19 / 20+ instead of 0-4 / 5-9 / 10+
#         Old bins gave 80% in one group (imbalanced, misleading donut)
# ═══════════════════════════════════════════════════════════════════════════
CLASS_COLORS_28 = {
    '<30': '#D85A30',
    '>30': '#4A6FA5',
    'NO':  '#1D9E75',
}
CLASS_LABELS_28 = {
    '<30': 'Readmitted < 30 days',
    '>30': 'Readmitted > 30 days',
    'NO':  'Not Readmitted',
}

df['med_group'] = pd.cut(df['num_medications'],
                         bins=[0, 9, 19, 100],
                         labels=['Low (0–9)', 'Medium (10–19)', 'High (20+)'])
polypharm = pd.crosstab(df['med_group'], df['readmitted'],
                        normalize='index') * 100

groups       = ['Low (0–9)', 'Medium (10–19)', 'High (20+)']
group_labels = ['Low\n(0–9 meds)', 'Medium\n(10–19 meds)', 'High\n(20+ meds)']
colors_donut = [CLASS_COLORS_28['<30'], CLASS_COLORS_28['>30'], CLASS_COLORS_28['NO']]

fig, axes = plt.subplots(1, 3, figsize=(12, 5))

for i, (grp, lbl) in enumerate(zip(groups, group_labels)):
    ax     = axes[i]
    vals   = polypharm.loc[grp, ['<30', '>30', 'NO']].values
    n_grp  = (df['med_group'] == grp).sum()

    wedges, texts, autotexts = ax.pie(
        vals,
        colors=colors_donut,
        startangle=90,
        autopct='%1.1f%%',
        pctdistance=0.78,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
    )

    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    ax.set_title(f'{lbl}\n(n={n_grp:,})', fontsize=11, fontweight='bold', pad=10)

    pct = polypharm.loc[grp, '<30']
    ax.text(0, 0, f'{pct:.1f}%', ha='center', va='center',
            fontsize=14, fontweight='bold',
            color=CLASS_COLORS_28['<30'] if grp == 'High (20+)' else '#333333')

handles28 = [mpatches.Patch(color=CLASS_COLORS_28[c], label=CLASS_LABELS_28[c])
             for c in CLASS_ORDER]
fig.legend(handles=handles28, loc='lower center', ncol=3, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, -0.05))

fig.suptitle('Step 28 | Polypharmacy — Number of Medications vs Readmission',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(OUT / 'step28_polypharmacy.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: step28_polypharmacy.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 29 — Efficiency Ratio (Procedures per Day)
# 🟡 FIX: Filter out num_procedures == 0 before computing efficiency
#         Old code: 46% zero rows skewed distribution heavily toward 0
#         New code: only patients who actually had procedures — honest metric
# ═══════════════════════════════════════════════════════════════════════════
CLASS_COLORS_29 = {
    '<30': '#D85A30',
    '>30': '#4A6FA5',
    'NO':  '#1D9E75',
}
CLASS_LABELS_29 = {
    '<30': 'Readmitted < 30 days',
    '>30': 'Readmitted > 30 days',
    'NO':  'Not Readmitted',
}

df_eff              = df[df['num_procedures'] > 0].copy()
df_eff['efficiency'] = df_eff['num_procedures'] / df_eff['time_in_hospital']
efficiency_capped   = df_eff['efficiency'].clip(upper=df_eff['efficiency'].quantile(0.99))
df_eff['efficiency_capped'] = efficiency_capped

eff_stats = df_eff.groupby('readmitted')['efficiency_capped'].agg(['mean', 'std']).reindex(CLASS_ORDER)

fig, ax = plt.subplots(figsize=(9, 6))

x    = np.arange(len(CLASS_ORDER))
bars = ax.bar(
    x,
    eff_stats['mean'],
    yerr=eff_stats['std'],
    width=0.5,
    color=[CLASS_COLORS_29[c] for c in CLASS_ORDER],
    edgecolor='white', linewidth=0.8,
    error_kw=dict(ecolor='#888780', elinewidth=1.2, capsize=5, capthick=1.2),
    zorder=3
)

for bar, cls in zip(bars, CLASS_ORDER):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height + eff_stats.loc[cls, 'std'] + 0.02,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=10,
            fontweight='bold', color='#2C2C2A')

ax.set_title('Step 29 | Procedures per Day (Efficiency) by Readmission Status',
             fontsize=13, fontweight='bold', pad=14, color='#2C2C2A')
ax.set_xlabel('Readmission Status', fontsize=11, labelpad=8, color='#444441')
ax.set_ylabel('Mean Procedures per Day\n(capped at 99th percentile ± SD)',
              fontsize=11, labelpad=8, color='#444441')
ax.set_xticks(x)
ax.set_xticklabels([CLASS_LABELS_29[c] for c in CLASS_ORDER],
                   fontsize=10, color='#444441')
ax.tick_params(axis='y', labelsize=9, colors='#5F5E5A')
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='#B4B2A9')
ax.set_axisbelow(True)
ax.grid(axis='x', visible=False)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(0.6)
ax.spines[['left', 'bottom']].set_color('#B4B2A9')

handles29 = [mpatches.Patch(color=CLASS_COLORS_29[c], label=CLASS_LABELS_29[c])
             for c in CLASS_ORDER]
ax.legend(handles=handles29, loc='upper right', frameon=True,
          framealpha=0.9, edgecolor='#D3D1C7',
          fontsize=9, title='Readmission Status', title_fontsize=9)

# Note: analysis on patients who had at least one procedure
fig.text(0.5, -0.01,
         f'Analysis restricted to patients with ≥1 procedure  |  n = {len(df_eff):,} '
         f'({len(df_eff)/len(df)*100:.0f}% of total)',
         ha='center', fontsize=7.5, color='#888', style='italic')

plt.tight_layout()
plt.grid(False)
plt.savefig(OUT / 'step29_efficiency_ratio.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: step29_efficiency_ratio.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 30 — Admission Type × Source Combination Heatmap
# ═══════════════════════════════════════════════════════════════════════════
df['admit_comb'] = (df['admission_type_id'].astype(str) + ' | ' +
                    df['admission_source_id'].astype(str))
pivot_comb = pd.crosstab(df['admit_comb'], df['target_binary'],
                         normalize='index') * 100
pivot_comb = pivot_comb.sort_values(1, ascending=False).head(15)
pivot_comb.columns = ['Not <30 (%)', 'Readmitted <30 (%)']

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    pivot_comb, ax=ax,
    annot=True, fmt='.1f',
    cmap=sns.color_palette([
        '#E1F5EE', '#9FE1CB', '#5DCAA5', '#1D9E75', '#0F6E56', '#085041'
    ], as_cmap=True),
    linewidths=0.5, linecolor='#f0f0f0',
    cbar_kws={'label': 'Percentage (%)', 'shrink': 0.7, 'pad': 0.02},
    annot_kws={'size': 9, 'weight': 'bold', 'color': '#2C2C2A'}
)

for i in range(5):
    ax.add_patch(plt.Rectangle(
        (0, i), 2, 1,
        fill=False, edgecolor='#D85A30', lw=2, clip_on=False
    ))

ax.set_title(
    'Step 30 | Top 15 High-Risk Admission Type × Source Combinations\n30-Day Readmission Rate (%)',
    fontsize=13, fontweight='bold', pad=16, color='#2C2C2A')
ax.set_xlabel('Readmission Outcome', fontsize=11, labelpad=8, color='#444441')
ax.set_ylabel('Admission Type | Source ID', fontsize=11, labelpad=8, color='#444441')
ax.tick_params(axis='x', labelsize=10, colors='#444441', length=0)
ax.tick_params(axis='y', labelsize=9,  colors='#444441', length=0)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=8, colors='#5F5E5A')
cbar.set_label('Percentage (%)', fontsize=9, color='#444441')

ax.text(2.08, 2.5, '← Top 5\n   high-risk',
        fontsize=8, color='#D85A30', fontweight='bold',
        va='center', transform=ax.transData)

plt.tight_layout()
plt.savefig(OUT / 'step30_admit_type_source_combination.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: step30_admit_type_source_combination.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 33 — DiabetesMed × Change × Readmission
# ═══════════════════════════════════════════════════════════════════════════
CLASS_COLORS_33 = {
    '<30': '#7F77DD',
    '>30': '#EF9F27',
    'NO':  '#0F6E56',
}
CLASS_LABELS_33 = {
    '<30': 'Readmitted <30 days',
    '>30': 'Readmitted >30 days',
    'NO':  'Not readmitted',
}

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'font.size': 10,
})

cross_tab       = pd.crosstab([df['diabetesMed'], df['change']], df['readmitted'],
                               normalize='index') * 100
cross_tab_reset = cross_tab.reset_index()
cross_tab_reset['group'] = cross_tab_reset['diabetesMed'] + ' | ' + cross_tab_reset['change']

fig, ax = plt.subplots(figsize=(10, 6))

x     = np.arange(len(cross_tab_reset))
width = 0.25

for i, cls in enumerate(CLASS_ORDER):
    bars = ax.bar(x + i * width, cross_tab_reset[cls], width,
                  color=CLASS_COLORS_33[cls],
                  edgecolor='white', linewidth=0.7, zorder=3,
                  label=CLASS_LABELS_33[cls])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + 0.5, f'{h:.1f}%',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#2C2C2A')

ax.set_xticks(x + width)
ax.set_xticklabels(cross_tab_reset['group'], rotation=15,
                   ha='right', fontsize=10, color='#444441')
ax.set_ylabel('Percentage (%)', fontsize=11, labelpad=8, color='#444441')
ax.set_xlabel('DiabetesMed | Change Group', fontsize=11, labelpad=8, color='#444441')
ax.set_title('Step 33 | Readmission by DiabetesMed × Change',
             fontsize=13, fontweight='bold', pad=14, color='#2C2C2A')
ax.tick_params(axis='y', labelsize=9, colors='#5F5E5A')
ax.set_ylim(0, cross_tab_reset[CLASS_ORDER].values.max() * 1.2)
ax.grid(False)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(0.6)
ax.spines[['left', 'bottom']].set_color('#B4B2A9')

handles33 = [mpatches.Patch(color=CLASS_COLORS_33[c], label=CLASS_LABELS_33[c])
             for c in CLASS_ORDER]
ax.legend(handles=handles33, loc='upper right', frameon=True,
          framealpha=0.9, edgecolor='#D3D1C7',
          fontsize=9, title='Readmission Status', title_fontsize=9)

plt.tight_layout()
plt.savefig(OUT / 'step33_diabetesMed_change_crosstab.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: step33_diabetesMed_change_crosstab.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 36 — Age × Gender × Readmission Heatmap
# ═══════════════════════════════════════════════════════════════════════════
pivot_age_gender = pd.crosstab(df['age'], df['gender'],
                               values=df['target_binary'],
                               aggfunc='mean') * 100
pivot_age_gender = pivot_age_gender.fillna(0)

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    pivot_age_gender, ax=ax,
    annot=True, fmt='.1f',
    cmap=sns.color_palette([
        '#FAECE7', '#F5C4B3', '#F0997B', '#D85A30', '#993C1D', '#712B13'
    ], as_cmap=True),
    linewidths=0.5, linecolor='#f5f5f5',
    cbar_kws={'label': '30-Day Readmission Rate (%)', 'shrink': 0.7},
    annot_kws={'size': 11, 'weight': 'bold', 'color': '#2C2C2A'}
)

ax.set_title('Step 36 | 30-Day Readmission Rate by Age Group and Gender',
             fontsize=13, fontweight='bold', pad=16, color='#2C2C2A')
ax.set_xlabel('Gender', fontsize=11, labelpad=8, color='#444441')
ax.set_ylabel('Age Group', fontsize=11, labelpad=8, color='#444441')
ax.tick_params(axis='x', labelsize=10, colors='#444441', length=0)
ax.tick_params(axis='y', labelsize=10, colors='#444441', length=0)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=8, colors='#5F5E5A')
cbar.set_label('30-Day Readmission Rate (%)', fontsize=9, color='#444441')

plt.tight_layout()
plt.savefig(OUT / 'step36_age_gender_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: step36_age_gender_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# SAVE FINAL DATAFRAME
# NOTE: EDA-only columns (time_cat, med_group, depth_cat, etc.) are dropped
#       before saving — they will be re-created in Feature Engineering (Code 3)
#       after train/test split to avoid data leakage.
# ═══════════════════════════════════════════════════════════════════════════
eda_only_cols = [
    'time_cat', 'med_group', 'depth_cat', 'ratio_cat',
    'lab_diag_ratio', 'ratio_capped',
    'med_change_depth', 'efficiency', 'efficiency_capped',
    'admit_comb', 'encounter_count', 'multiple_encounters',
    'medical_specialty_grouped', 'payer_code_grouped',
    'target_binary',  # FIX: consistency — Code 3 apna target khud banayega
                       # train/test split ke baad, isliye is EDA-only column
                       # ko bhi baaki EDA-derived columns ki tarah drop karo
]
cols_to_drop_final = [c for c in eda_only_cols if c in df.columns]
df_save = df.drop(columns=cols_to_drop_final)

OUT_CSV = BASE_DIR / 'data' / 'processed' / 'diabetic_data_eda_final.csv'
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df_save.to_csv(OUT_CSV, index=False)

print(f"\n✅ Final DataFrame saved to : {OUT_CSV}")
print(f"✅ Shape                     : {df_save.shape}")
print(f"✅ Columns                   : {df_save.shape[1]}")
print(f"\n✅ All figures saved to      : {OUT}")
print("\nNEXT STEPS:")
print("  Code 2 → Train/Test split using GroupShuffleSplit(groups=patient_nbr)")
print("  Code 3 → Feature Engineering (outlier capping, encoding, ICD-9 grouping)")
print("  Code 4 → Model Training & Evaluation")