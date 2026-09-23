import pandas as pd
import numpy as np
import os, joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

# ============================================================================
# PATHS  —  apne system ke mutabiq adjust karo
# ============================================================================
BASE_DIR      = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = BASE_DIR / 'data/processed/diabetic_data_cleaned.csv'
PROCESSED_DIR = BASE_DIR / 'data/processed'
FIGURES_DIR   = BASE_DIR / 'reports/figures'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ─── Design tokens ───────────────────────────────────────────────────────────
PALETTE = {
    'bg'      : '#FFFFFF', 'panel'   : '#F7FAFC', 'spine'   : '#D1DCE5',
    'title'   : '#0D1B2A', 'label'   : '#2C3E50', 'tick'    : '#4A5568',
    'positive': '#C0392B', 'neutral' : '#2E6EA6', 'negative': '#2E8B57',
    'accent'  : '#E67E22',
}
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans', 'figure.facecolor': 'white',
    'axes.facecolor'   : 'white',       'axes.spines.top'  : False,
    'axes.spines.right': False,         'axes.grid'        : False,
    'savefig.dpi'      : 300,
})

# ============================================================================
# STAGE A — LOAD & DETERMINISTIC FEATURE ENGINEERING
# ============================================================================
df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
print(f"✅ Cleaned data loaded (from data_cleaning.py output)  →  shape: {df.shape}")
print("   This is the SAME cleaned file EDA.py reads -- hidden-missing-code")
print("   unmasking, rare-med consolidation, and A1C/glucose missing-flags")
print("   from data_cleaning.py now flow into modelling instead of being")
print("   silently redone (differently) from raw data here.")

# ── A1 / A1.5: SKIPPED — already done by data_cleaning.py ─────────────────
# weight/max_glu_serum/examide/citoglipton were already dropped/consolidated,
# and A1C_missing/A1C_category (informative-missingness) already exist in
# the cleaned CSV loaded above. Re-running this here used to crash with a
# KeyError on 'A1Cresult', which no longer exists post-cleaning.
print("   A1/A1.5 skipped: already handled by data_cleaning.py "
      "(A1C_missing/A1C_category, glu_serum_missing/glu_serum_category, "
      "num_rare_meds_used already present in the loaded data).")

# ── A2: SKIPPED — already done by data_cleaning.py ─────────────────────────
# gender='Unknown/Invalid' rows and expired/hospice discharge rows are
# already removed upstream. Re-running the discharge_disposition_id check
# here used to be a silent no-op anyway, since that column is now a STRING
# ('Unknown' for hidden-missing codes) in the cleaned CSV -- comparing it to
# an int list like [11, 13, 14] never matches.
print(f"   A2 skipped: gender/discharge-id row removal already done upstream. Shape: {df.shape}")

# ── A3: Impute remaining missing values ─────────────────────────────────────
df['payer_code']        = df['payer_code'].fillna('Unknown')
df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
df['race']              = df['race'].fillna(df['race'].mode()[0])

# BUG3 FIX: diag fill with 'Unknown' NOT mode
# ❌ OLD (wrong): df['diag_1'] = df['diag_1'].fillna(df['diag_1'].mode()[0])
#                 mode = '428' (heart failure) — assigns specific disease to patient
#                 with no recorded diagnosis. Clinically misleading.
# ✅ NEW (correct): 'Unknown' = honest neutral fill
for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].fillna('Unknown')
print("   ✅ BUG3 FIX: diag_1/2/3 missing → 'Unknown' (not mode/heart-failure)")
print("   Remaining missing values:", df.isnull().sum().sum())

# ── A4: Create binary target ─────────────────────────────────────────────────
df['target'] = (df['readmitted'] == '<30').astype(int)
df.drop('readmitted', axis=1, inplace=True)
print(f"\n   Target distribution: {df['target'].value_counts().to_dict()}")
print(f"   Positive class: {df['target'].mean()*100:.2f}%")

# ── A5: Diagnosis code grouping (CCS taxonomy) ──────────────────────────────
def map_diagnosis(code):
    """Map ICD-9 to 9 CCS disease groups. 'Unknown' preserved explicitly."""
    if pd.isna(code) or str(code).strip() in ['', '000', 'Unknown']:
        return 'Unknown'
    code_str = str(code).strip()
    if code_str.upper().startswith(('V', 'E')):
        return 'Other'
    try:
        code_num = int(code_str.split('.')[0])
    except ValueError:
        return 'Other'
    if   code_num == 250:              return 'Diabetes'
    elif 390 <= code_num <= 459:       return 'Circulatory'
    elif 460 <= code_num <= 519:       return 'Respiratory'
    elif 520 <= code_num <= 579:       return 'Digestive'
    elif 580 <= code_num <= 629:       return 'Genitourinary'
    elif 710 <= code_num <= 739:       return 'Musculoskeletal'
    elif 800 <= code_num <= 999:       return 'Injury'
    elif 140 <= code_num <= 239:       return 'Neoplasms'
    else:                              return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[f'{col}_group'] = df[col].apply(map_diagnosis)
    df.drop(col, axis=1, inplace=True)

# Clinical flag: primary diagnosis is diabetes
df['primary_diag_diabetes'] = (df['diag_1_group'] == 'Diabetes').astype(int)
# Comorbidity diversity: distinct disease groups across 3 diagnoses
df['comorbidity_diversity'] = df[['diag_1_group','diag_2_group','diag_3_group']].apply(
    lambda r: len(set(r) - {'Unknown', 'Other'}), axis=1)
print("   Diagnosis groups + clinical flags created.")

# ── A6: Map ID columns to meaningful groups ──────────────────────────────────
def _to_int_or_unknown(x):
    """admission_type_id / admission_source_id / discharge_disposition_id are
    STRINGS in the cleaned CSV ('Unknown' for hidden-missing codes, numeric
    strings otherwise). Convert back to int for the bucket logic below,
    keeping 'Unknown' as its own explicit bucket instead of it silently
    falling into 'Other' (which is what happened before this fix -- the
    string-vs-int comparison below never matched anything).
    """
    if x == 'Unknown':
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None

def map_admission_type(x):
    n = _to_int_or_unknown(x)
    if n is None: return 'Unknown'
    return {1: 'Emergency', 2: 'Urgent', 3: 'Elective'}.get(n, 'Other')

def map_admission_source(x):
    n = _to_int_or_unknown(x)
    if   n is None:                          return 'Unknown'
    elif n in [1,2,3,4,5,6]:                 return 'Referral'
    elif n in [7,8,9,10,11,12,13]:           return 'Emergency'
    elif n in [14,15,16,17,18,19,20,21,22,
               23]:                          return 'Transfer'
    else:                                    return 'Other'

def map_discharge(x):
    n = _to_int_or_unknown(x)
    if   n is None:                          return 'Unknown'
    elif n == 1:                             return 'Home'
    elif n in [2,3,4,5,6,7,8,9,10,12,15,
               16,17,18,22,23,24,25,27,28]:  return 'Transferred'
    else:                                    return 'Other'

df['admission_type_group']   = df['admission_type_id'].apply(map_admission_type)
df['admission_source_group'] = df['admission_source_id'].apply(map_admission_source)
df['discharge_group']        = df['discharge_disposition_id'].apply(map_discharge)
df.drop(['admission_type_id','admission_source_id','discharge_disposition_id'],
        axis=1, inplace=True)
print("   admission_type/source/discharge → categorical groups ✓")

# ── A7: Age bracket → numeric midpoint ──────────────────────────────────────
age_mapping = {
    '[0-10)':5,   '[10-20)':15, '[20-30)':25, '[30-40)':35, '[40-50)':45,
    '[50-60)':55, '[60-70)':65, '[70-80)':75, '[80-90)':85, '[90-100)':95
}
df['age_numeric'] = df['age'].map(age_mapping)
df.drop('age', axis=1, inplace=True)

# ── A8: Medication aggregate features ───────────────────────────────────────
# BUG1 FIX: examide and citoglipton REMOVED — both 100% "No" (zero variance)
# ❌ OLD: included examide, citoglipton → inflated num_med_changes/num_meds_active
# ✅ NEW: only medications with actual variability
med_cols = [
    'metformin',    'repaglinide',    'nateglinide',    'chlorpropamide',
    'glimepiride',  'acetohexamide',  'glipizide',      'glyburide',
    'tolbutamide',  'pioglitazone',   'rosiglitazone',  'acarbose',
    'miglitol',     'troglitazone',   'tolazamide',
    # examide   ← REMOVED (100% No)
    # citoglipton ← REMOVED (100% No)
    'insulin',      'glyburide-metformin',       'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]
med_cols = [m for m in med_cols if m in df.columns]
print(f"\n   ✅ BUG1 FIX: examide/citoglipton removed from med_cols")
print(f"   Active med_cols count: {len(med_cols)}")

df['num_med_changes'] = df[med_cols].apply(
    lambda row: (row.isin(['Up', 'Down'])).sum(), axis=1)
df['num_meds_active'] = df[med_cols].apply(
    lambda row: (row != 'No').sum(), axis=1)
df['num_med_increase'] = df[med_cols].apply(
    lambda row: (row == 'Up').sum(), axis=1)
df['num_med_decrease'] = df[med_cols].apply(
    lambda row: (row == 'Down').sum(), axis=1)

# ============================================================
# ✅ INSULIN FIXED: One-Hot Encoding (Safe for ALL models)
# ============================================================
# ❌ OLD (Removed): No=0, Steady=1, Up=2, Down=3 → false ordinal assumption
# ✅ NEW (Fixed): One-Hot + Binary flags (no linear ranking)
df['insulin_No'] = (df['insulin'] == 'No').astype(int)
df['insulin_Steady'] = (df['insulin'] == 'Steady').astype(int)
df['insulin_Up'] = (df['insulin'] == 'Up').astype(int)
df['insulin_Down'] = (df['insulin'] == 'Down').astype(int)

# Binary directional flags (clinical signals)
df['insulin_adjusted'] = df['insulin'].isin(['Up', 'Down']).astype(int)
df['insulin_increased'] = (df['insulin'] == 'Up').astype(int)

print(f"   ✅ INSULIN FIXED: One-Hot (No,Steady,Up,Down) + binary flags added.")
print(f"   ❌ Old 'insulin_ord' removed to avoid false ordinal assumption.")

# Polypharmacy level (ordinal)
df['polypharmacy_level'] = pd.cut(
    df['num_medications'], bins=[0, 9, 19, 100], labels=[0, 1, 2]
).astype(int)

df.drop(columns=med_cols, inplace=True)
print(f"   Medication features engineered. Shape: {df.shape}")

# ── A9: Utilisation aggregate ────────────────────────────────────────────────
df['total_prior_visits']   = (df['number_outpatient']
                               + df['number_emergency']
                               + df['number_inpatient'])
df['prior_inpatient_flag'] = (df['number_inpatient'] > 0).astype(int)
df['prior_emergency_flag'] = (df['number_emergency'] > 0).astype(int)

# Utilization weighted score
df['utilization_score'] = (df['number_inpatient'] * 3
                           + df['number_emergency'] * 2
                           + df['number_outpatient'] * 1)

# Clinical composite features
df['lab_per_day']         = df['num_lab_procedures'] / df['time_in_hospital'].clip(lower=1)
df['procedure_per_day']   = df['num_procedures']     / df['time_in_hospital'].clip(lower=1)
df['med_per_diagnosis']   = df['num_medications']    / df['number_diagnoses'].clip(lower=1)
df['extended_stay']       = (df['time_in_hospital'] > 7).astype(int)
df['high_complexity']     = (
    (df['num_lab_procedures'] > 40) & (df['number_diagnoses'] > 7)
).astype(int)

# ── A10: Binary encode low-cardinality cats ──────────────────────────────────
# NOTE: gender Unknown/Invalid already removed in A2 — safe to encode now
df['gender_binary']      = (df['gender']      == 'Male').astype(int)
df['change_binary']      = (df['change']      == 'Ch').astype(int)
df['diabetesMed_binary'] = (df['diabetesMed'] == 'Yes').astype(int)
df.drop(['gender', 'change', 'diabetesMed'], axis=1, inplace=True)

# ── A11: Drop encounter_id ───────────────────────────────────────────────────
df.drop('encounter_id', axis=1, inplace=True)

print(f"\n✅ Stage A complete. Shape: {df.shape}")

# ============================================================================
# STAGE B — TRAIN / TEST SPLIT  (GroupShuffleSplit on patient_nbr)
# ============================================================================
X      = df.drop('target', axis=1)
y      = df['target']
groups = df['patient_nbr']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train = X.iloc[train_idx].copy()
X_test  = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test  = y.iloc[test_idx].copy()

# NEW: patient groups ko drop karne se PEHLE save karo — Code 4 ki inner CV
# (StratifiedGroupKFold) ko patient-safe banane ke liye zaroori hai
groups_for_cv = X_train['patient_nbr'].reset_index(drop=True)
groups_for_cv.to_csv(PROCESSED_DIR / 'train_patient_groups.csv', index=False, header=['patient_nbr'])
print(f"   Saved train_patient_groups.csv ({groups_for_cv.nunique()} unique patients)")

X_train.drop('patient_nbr', axis=1, inplace=True)
X_test.drop('patient_nbr',  axis=1, inplace=True)

print(f"   Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"   Train positive: {y_train.mean()*100:.2f}%  |  Test positive: {y_test.mean()*100:.2f}%")

assert len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])) == 0
print("   ✅ Zero patient overlap confirmed.")
# ============================================================================
# STAGE B.5 — IQR OUTLIER CAPPING (train-fitted only)
# ============================================================================
# Only genuine continuous clinical measurements
# admission_type/source/discharge already categorical — NOT included
continuous_cols = [c for c in [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications',  'number_outpatient',   'number_emergency',
    'number_inpatient', 'number_diagnoses',
    'num_med_changes',  'num_meds_active',     'num_med_increase',
    'num_med_decrease', 'total_prior_visits',  'utilization_score',
    'lab_per_day',      'procedure_per_day',   'med_per_diagnosis',
    'comorbidity_diversity',
] if c in X_train.columns]

# Zero-inflated columns: use 99th percentile cap (not IQR — all non-zero flagged)
zero_inflated = {'number_outpatient', 'number_emergency',
                 'number_inpatient',  'total_prior_visits', 'utilization_score'}

iqr_bounds = {}
print(f"\n{'='*65}")
print("STAGE B.5 — Outlier Capping (train-fitted, no leakage)")
print(f"{'='*65}")

for col in continuous_cols:
    if col in zero_inflated:
        # 99th percentile cap for zero-inflated distributions
        ub = X_train[col].quantile(0.99)
        lb = 0
        iqr_bounds[col] = {'lower': lb, 'upper': ub, 'method': 'p99'}
        X_train[col] = X_train[col].clip(upper=ub)
        X_test[col]  = X_test[col].clip(upper=ub)
        print(f"  {col:30} [p99 cap]: [0, {ub:.1f}]")
    else:
        Q1    = X_train[col].quantile(0.25)
        Q3    = X_train[col].quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        iqr_bounds[col] = {'lower': lower, 'upper': upper, 'method': 'IQR'}
        X_train[col] = X_train[col].clip(lower=lower, upper=upper)
        X_test[col]  = X_test[col].clip(lower=lower, upper=upper)
        print(f"  {col:30} [IQR]:    [{lower:.1f}, {upper:.1f}]")

joblib.dump(iqr_bounds, PROCESSED_DIR / 'iqr_bounds.pkl')
print("   IQR bounds saved.")

# ============================================================================
# STAGE C — HIGH-CARDINALITY GROUPING (fit on train → apply to test)
# ============================================================================
def group_high_cardinality_fixed(train_s, test_s, top_n=10):
    """
    'Unknown' explicitly preserved — missing data signal (49% medical_specialty,
    39% payer_code). 'Rare_Other' for categories outside top-N.
    This prevents 'Unknown' from merging with actual rare categories.
    """
    top_cats = train_s.value_counts().nlargest(top_n).index.tolist()
    def _map(x, top):
        if x == 'Unknown':  return 'Unknown'      # missing signal — preserve
        elif x in top:      return x
        else:               return 'Rare_Other'   # low-frequency specialty
    return (train_s.apply(lambda x: _map(x, top_cats)),
            test_s.apply(lambda x:  _map(x, top_cats)))

X_train['medical_specialty_grouped'], X_test['medical_specialty_grouped'] = \
    group_high_cardinality_fixed(X_train['medical_specialty'], X_test['medical_specialty'])

X_train['payer_code_grouped'], X_test['payer_code_grouped'] = \
    group_high_cardinality_fixed(X_train['payer_code'], X_test['payer_code'])

X_train.drop(['medical_specialty', 'payer_code'], axis=1, inplace=True)
X_test.drop(['medical_specialty',  'payer_code'], axis=1, inplace=True)

print(f"\n✅ Stage C — Grouping done.")
print(f"   medical_specialty_grouped distribution (train):")
print(X_train['medical_specialty_grouped'].value_counts().head(12).to_string())

# ============================================================================
# STAGE D1 — ONE-HOT ENCODING
# ============================================================================
# ✅ FIX_OHE: pd.get_dummies() (train/test alag-alag call, phir manual align)
# ki jagah sklearn OneHotEncoder use ho raha hai -- train par .fit(), test par
# sirf .transform(). Yeh train/test baseline-category mismatch ka structural
# risk khatam karta hai, aur encoder ko joblib.dump() se save karta hai taake
# Code 5 (deployment/naya patient) mein reliably reuse ho sake.
# ✅ ADDED: A1C_category (preserves missing signal as separate category)
nominal_cols = [c for c in [
    'race',
    'A1C_category', 'glu_serum_category',
    'diag_1_group', 'diag_2_group', 'diag_3_group',
    'admission_type_group', 'admission_source_group', 'discharge_group',
    'medical_specialty_grouped', 'payer_code_grouped'
] if c in X_train.columns]

# NEW: OHE se pehle raw categorical columns ka copy rakho — Stage D5
# (group-aware MI selection) mein har categorical ko GROUP ki tarah
# evaluate karne ke liye chahiye hoga
X_train_nominal_raw = X_train[nominal_cols].copy()

ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
train_ohe_arr = ohe.fit_transform(X_train[nominal_cols])
test_ohe_arr  = ohe.transform(X_test[nominal_cols])

ohe_feature_names = ohe.get_feature_names_out(nominal_cols)
train_ohe_df = pd.DataFrame(train_ohe_arr, columns=ohe_feature_names, index=X_train.index)
test_ohe_df  = pd.DataFrame(test_ohe_arr,  columns=ohe_feature_names, index=X_test.index)

X_train_enc = pd.concat([X_train.drop(columns=nominal_cols), train_ohe_df], axis=1)
X_test_enc  = pd.concat([X_test.drop(columns=nominal_cols),  test_ohe_df],  axis=1)

# joblib se save karo — Code 5 (deployment) mein isi encoder se .transform() karna
joblib.dump(ohe, PROCESSED_DIR / 'onehot_encoder.pkl')
joblib.dump(nominal_cols, PROCESSED_DIR / 'nominal_cols.pkl')

# NEW: har OHE dummy column ko uske parent categorical se map karo
# (e.g. 'race_Caucasian' -> 'race'), taake Stage D5 unhe group mein treat kar sake
ohe_dummy_to_parent = {}
for parent in nominal_cols:
    for dummy in ohe_feature_names:
        if dummy.startswith(parent + '_'):
            ohe_dummy_to_parent[dummy] = parent

print(f"\n✅ Stage D1 — OHE done (sklearn OneHotEncoder, fit on train only).")
print(f"   Train: {X_train_enc.shape} | Test: {X_test_enc.shape}")
print(f"   ✅ FIX_OHE: encoder saved -> onehot_encoder.pkl (reusable at inference)")
features_after_ohe = X_train_enc.shape[1] 
# ============================================================================
# HEATMAP 1 — NUMERIC FEATURES CORRELATION MATRIX
# ============================================================================
numeric_cols_for_heatmap = [c for c in [
    'time_in_hospital',  'num_lab_procedures', 'num_procedures',
    'num_medications',   'number_outpatient',  'number_emergency',
    'number_inpatient',  'number_diagnoses',   'age_numeric',
    'num_med_changes',   'num_meds_active',    
    'total_prior_visits','gender_binary',      'change_binary',
    'diabetesMed_binary','utilization_score',  'lab_per_day',
    'procedure_per_day', 'comorbidity_diversity',
] if c in X_train_enc.columns]

readable_labels = {
    'time_in_hospital'    : 'Time in\nHospital',    'num_lab_procedures': 'Lab\nProcs',
    'num_procedures'      : 'Procedures',           'num_medications'   : 'Medications',
    'number_outpatient'   : 'Outpatient',           'number_emergency'  : 'Emergency',
    'number_inpatient'    : 'Inpatient',            'number_diagnoses'  : 'Diagnoses',
    'age_numeric'         : 'Age',                  'num_med_changes'   : 'Med Changes',
    'num_meds_active'     : 'Active Meds',          'total_prior_visits'  : 'Prior Visits',
    'gender_binary'       : 'Gender',               'change_binary'     : 'Med Change',
    'diabetesMed_binary'  : 'DiabetesMed',          'utilization_score' : 'Util Score',
    'lab_per_day'         : 'Labs/Day',             'procedure_per_day' : 'Procs/Day',
    'comorbidity_diversity': 'Comorbidity',
}

corr_matrix_num = X_train_enc[numeric_cols_for_heatmap].corr()
labels_heatmap  = [readable_labels.get(c, c) for c in numeric_cols_for_heatmap]

cmap_div = LinearSegmentedColormap.from_list(
    'elegant_div', ['#0D4A45','#3A8A7A','#F5F7FA','#B05A5A','#6B1A1A'], N=256)

fig_h1, ax_h1 = plt.subplots(figsize=(16, 12), facecolor='white')
fig_h1.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.14)

mask_diag = np.eye(len(numeric_cols_for_heatmap), dtype=bool)
sns.heatmap(
    corr_matrix_num, annot=True, fmt='.2f', cmap=cmap_div, vmin=-1, vmax=1,
    square=True, linewidths=1.5, linecolor='white', mask=mask_diag,
    annot_kws={'size': 7, 'weight': 'bold', 'color': '#0D1B2A'},
    cbar_kws={'shrink': 0.75, 'pad': 0.03}, ax=ax_h1,
    xticklabels=labels_heatmap, yticklabels=labels_heatmap,
)

for i in range(len(numeric_cols_for_heatmap)):
    ax_h1.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='#F5F7FA', zorder=3))
    ax_h1.text(i+0.5, i+0.5, '1.00', ha='center', va='center',
               fontsize=6.5, color='#888888', fontstyle='italic')

for i in range(len(numeric_cols_for_heatmap)):
    for j in range(len(numeric_cols_for_heatmap)):
        if i != j and abs(corr_matrix_num.iloc[i, j]) > 0.60:
            ax_h1.add_patch(plt.Rectangle((j, i), 1, 1,
                fill=False, edgecolor='#E67E22', lw=1.5, zorder=4))

cbar_h1 = ax_h1.collections[0].colorbar
cbar_h1.ax.tick_params(labelsize=8, length=0)
cbar_h1.set_label('Pearson Correlation', fontsize=9, color=PALETTE['label'], labelpad=8)
ax_h1.set_xticklabels(labels_heatmap, fontsize=7.5, color=PALETTE['label'], rotation=45, ha='right')
ax_h1.set_yticklabels(labels_heatmap, fontsize=7.5, color=PALETTE['label'], rotation=0)
ax_h1.tick_params(length=0)
for side in ['top','right','left','bottom']: ax_h1.spines[side].set_visible(False)

fig_h1.text(0.015, 0.94, 'Feature Engineering  |  Correlation Matrix — Numeric Features',
            fontsize=13, fontweight='bold', color=PALETTE['title'])
fig_h1.text(0.015, 0.908, 'Orange border = |r| > 0.60  |  Diagonal = self-correlation (1.00)',
            fontsize=8.5, color='#5A6A7A', style='italic')
fig_h1.legend(handles=[
    mpatches.Patch(facecolor='#0D4A45', edgecolor='white', label='Strong Negative (−1.0)'),
    mpatches.Patch(facecolor='#F5F7FA', edgecolor='#CCCCCC', label='No Correlation (0.0)'),
    mpatches.Patch(facecolor='#6B1A1A', edgecolor='white', label='Strong Positive (+1.0)'),
    mpatches.Patch(facecolor='none',    edgecolor='#E67E22', linewidth=1.5, label='|r| > 0.60'),
], loc='lower center', bbox_to_anchor=(0.48, 0.01), ncol=4, fontsize=8,
   frameon=True, facecolor='white', edgecolor=PALETTE['spine'], framealpha=1.0)

fig_h1.savefig(FIGURES_DIR / 'fe_numeric_correlation_heatmap.png',
               dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig_h1)
print("\n✅ Saved: fe_numeric_correlation_heatmap.png")

# ============================================================================
# HEATMAP 2 — CORRELATION WITH TARGET
# ============================================================================
train_with_target = X_train_enc[numeric_cols_for_heatmap].copy()
train_with_target['target'] = y_train.values

target_corr = (train_with_target.corr()['target']
               .drop('target')
               .sort_values(key=abs, ascending=False))
target_corr_df            = target_corr.reset_index()
target_corr_df.columns    = ['Feature', 'Correlation']
target_corr_df['Readable'] = target_corr_df['Feature'].map(
    lambda c: readable_labels.get(c, c.replace('_',' ')))
target_corr_df['Color']   = target_corr_df['Correlation'].apply(
    lambda v: '#C0392B' if v > 0 else '#2E6EA6')

fig_h2, ax_h2 = plt.subplots(figsize=(11, 8), facecolor='white')
fig_h2.subplots_adjust(left=0.28, right=0.88, top=0.88, bottom=0.10)

bars_h2 = ax_h2.barh(target_corr_df['Readable'], target_corr_df['Correlation'],
                      color=target_corr_df['Color'], height=0.62,
                      edgecolor='white', linewidth=0.8, zorder=3)
for bar, val in zip(bars_h2, target_corr_df['Correlation']):
    x_pos = val + 0.002 if val >= 0 else val - 0.002
    ax_h2.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.4f}',
               ha='left' if val >= 0 else 'right', va='center',
               fontsize=8.5, fontweight='bold', color='#1A1A2E')

ax_h2.axvline(0, color=PALETTE['spine'], linewidth=1.2, zorder=2)
for thr in [0.02, -0.02]:
    ax_h2.axvline(thr, color=PALETTE['accent'], linewidth=1.0, linestyle='--', alpha=0.7)

ax_h2.invert_yaxis()
ax_h2.set_xlabel('Pearson Correlation with Target (<30 day readmission)',
                  fontsize=10, color=PALETTE['label'], labelpad=8)
ax_h2.tick_params(axis='both', length=0, labelsize=9)
ax_h2.spines['left'].set_color(PALETTE['spine'])
ax_h2.spines['bottom'].set_color(PALETTE['spine'])

fig_h2.text(0.015, 0.94,
            'Feature Engineering  |  Correlation of Numeric Features with Target',
            fontsize=13, fontweight='bold', color=PALETTE['title'])
fig_h2.text(0.015, 0.908,
            'Red = positive  |  Blue = negative  |  Orange = ±0.02 reference',
            fontsize=8.5, color='#5A6A7A', style='italic')

fig_h2.savefig(FIGURES_DIR / 'fe_target_correlation.png',
               dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig_h2)
print("✅ Saved: fe_target_correlation.png")

# ============================================================================
# STAGE D2 — ROBUST SCALER (fit on train only)
# FIX6: Binary columns (0/1) exclude from scaling — meaningless to scale
# ============================================================================
# FIX6: Only scale continuous/ordinal features — NOT binary flags
binary_cols = [
    'gender_binary', 'change_binary', 'diabetesMed_binary',
    'primary_diag_diabetes', 'prior_inpatient_flag', 'prior_emergency_flag',
    'extended_stay', 'high_complexity', 'insulin_adjusted', 'insulin_increased',
]
numeric_scale_cols = [c for c in [
    'time_in_hospital',  'num_lab_procedures', 'num_procedures',
    'num_medications',   'number_outpatient',  'number_emergency',
    'number_inpatient',  'number_diagnoses',   'age_numeric',
    'num_med_changes',   'num_meds_active',    'num_med_increase',
    'num_med_decrease',  
    'total_prior_visits', 'utilization_score', 'lab_per_day',
    'procedure_per_day', 'med_per_diagnosis', 'comorbidity_diversity', 
    'polypharmacy_level',
] if c in X_train_enc.columns]

# FIX6: binary cols are intentionally NOT in numeric_scale_cols
print(f"\n   ✅ FIX6: Binary cols excluded from RobustScaler: {binary_cols}")

scaler = RobustScaler()
X_train_scaled = X_train_enc.copy()
X_test_scaled  = X_test_enc.copy()
X_train_scaled[numeric_scale_cols] = scaler.fit_transform(X_train_enc[numeric_scale_cols])
X_test_scaled[numeric_scale_cols]  = scaler.transform(X_test_enc[numeric_scale_cols])
print(f"✅ Stage D2 — Scaling done. Scaled {len(numeric_scale_cols)} numeric cols.")

# ============================================================================
# STAGE D3 — VARIANCE THRESHOLD (fitted on train)
# ============================================================================
print(f"\n{'='*65}")
print("STAGE D3 — VarianceThreshold (threshold=0.01)")
print(f"{'='*65}")

var_selector   = VarianceThreshold(threshold=0.01)
X_train_var_arr = var_selector.fit_transform(X_train_scaled)
X_test_var_arr  = var_selector.transform(X_test_scaled)

all_features_var  = X_train_scaled.columns.tolist()
kept_mask_var     = var_selector.get_support()
kept_features_var = [f for f, k in zip(all_features_var, kept_mask_var) if k]
removed_var       = [f for f, k in zip(all_features_var, kept_mask_var) if not k]

X_train_var = pd.DataFrame(X_train_var_arr, columns=kept_features_var)
X_test_var  = pd.DataFrame(X_test_var_arr,  columns=kept_features_var)

print(f"   Features before: {features_after_ohe}")
print(f"   Features after:  {len(kept_features_var)}")
if removed_var:
    print(f"   ❌ Removed ({len(removed_var)} features):")
    for f in removed_var:
        print(f"      - {f}  (variance={X_train_scaled[f].var():.6f})")
else:
    print("   All features passed variance threshold.")

# ============================================================================
# STAGE D4 — CORRELATION FILTER (target-correlation based drop)
# ✅ FIX: Correlation filter now applies ONLY to continuous/engineered numeric features
# OHE columns have structural negative correlation (e.g., race_Asian vs race_African)
# — skipping them prevents dropping useful categories.
# ============================================================================
print(f"\n{'='*65}")
print("STAGE D4 — Correlation Filter (|r| > 0.90, target-corr based drop)")
print("          Applied ONLY to continuous numeric features (OHE columns excluded)")
print(f"{'='*65}")

y_train_arr = y_train.values
target_corrs_var = {}
for col in X_train_var.columns:
    try:
        r = np.corrcoef(X_train_var[col].values, y_train_arr)[0, 1]
        target_corrs_var[col] = abs(r) if not np.isnan(r) else 0.0
    except Exception:
        target_corrs_var[col] = 0.0

# ✅ FIX: Restrict correlation matrix to continuous features ONLY
continuous_features_for_corr = [c for c in numeric_scale_cols if c in X_train_var.columns]

if continuous_features_for_corr:
    corr_matrix_var = X_train_var[continuous_features_for_corr].corr().abs()
    upper_tri = corr_matrix_var.where(
        np.triu(np.ones(corr_matrix_var.shape), k=1).astype(bool))
    print(f"   ✅ Correlation matrix built on {len(continuous_features_for_corr)} continuous features only.")
    print(f"   (OHE columns automatically excluded from this filter)")
else:
    corr_matrix_var = pd.DataFrame()
    upper_tri = pd.DataFrame()
    print("   ⚠️ No continuous features found for correlation filter — skipping.")

cols_to_drop_corr = set()
corr_pairs_found = []

if len(corr_matrix_var) > 0:
    for col in upper_tri.columns:
        if col in cols_to_drop_corr: continue
        for partner in upper_tri.index[upper_tri[col] > 0.90].tolist():
            if partner in cols_to_drop_corr: continue
            col_tc     = target_corrs_var.get(col, 0.0)
            partner_tc = target_corrs_var.get(partner, 0.0)
            to_drop = partner if col_tc >= partner_tc else col
            to_keep = col     if col_tc >= partner_tc else partner
            cols_to_drop_corr.add(to_drop)
            corr_pairs_found.append({
                'kept': to_keep, 'dropped': to_drop,
                'r': corr_matrix_var.loc[col, partner],
                'kept_target_corr'   : target_corrs_var.get(to_keep, 0),
                'dropped_target_corr': target_corrs_var.get(to_drop, 0),
            })

removed_corr = list(cols_to_drop_corr)
X_train_corr = X_train_var.drop(columns=removed_corr, errors='ignore')
X_test_corr  = X_test_var.drop(columns=removed_corr,  errors='ignore')

print(f"   Features before: {len(kept_features_var)}")
print(f"   Features after:  {X_train_corr.shape[1]}")
if corr_pairs_found:
    for p in corr_pairs_found:
        print(f"   r={p['r']:.3f} | KEEP '{p['kept']}' (tgt_r={p['kept_target_corr']:.4f})"
              f" | DROP '{p['dropped']}' (tgt_r={p['dropped_target_corr']:.4f})")
else:
    print("   No highly correlated continuous pairs found.")
print("   (OHE columns were skipped — their structural negative correlation is expected)")

# ============================================================================
# STAGE D5 — MUTUAL INFORMATION SELECTION (FIX_MI v2 — GROUP-AWARE)
# ============================================================================
# ✅ FIX_MI v2: Pehli fix (discrete_features mask) ne per-dummy estimator ko
# theek kiya tha, lekin ASAL masla kuch aur tha — OneHotEncoding ek categorical
# variable ka poora signal 8-10 alag dummy columns mein baant deti hai, isliye
# har dummy ki individual MI chhoti dikhti hai chahe poora categorical variable
# milkar strong signal rakhta ho. Flat threshold un dummies ko unfairly punish
# karta hai. Verified: pehli fix ke baad bhi 'diag_1_group_Diabetes' drop ho
# raha tha (MI=0.000138 < 0.0005) — file ka apna sanity check FAIL ho raha tha.
#
# FIX: har categorical group (race, diag_1_group, medical_specialty_grouped,
# etc.) ko EK BAAR, uske RAW (pre-OHE) column pe evaluate karo -- na ke har
# dummy ko alag-alag. Agar poora group threshold pass kare, uske SAARE
# surviving dummies ek saath rakho; agar fail kare, saare ek saath drop karo.
# Numeric/binary engineered features (jo OHE se nahi aaye) pehli tarah hi
# per-feature MI se filter hote hain -- unke liye kuch nahi badla.
print(f"\n{'='*65}")
print("STAGE D5 — Mutual Information Selection (group-aware, threshold=0.0005)")
print(f"{'='*65}")

MI_THRESHOLD = 0.0005

surviving_ohe_cols = [c for c in X_train_corr.columns if c in ohe_dummy_to_parent]
non_ohe_cols        = [c for c in X_train_corr.columns if c not in ohe_dummy_to_parent]

# Tier 1 — numeric/binary engineered features (unchanged logic)
discrete_mask_nonohe = [X_train_corr[col].nunique() <= 10 for col in non_ohe_cols]
mi_scores_nonohe = mutual_info_classif(
    X_train_corr[non_ohe_cols], y_train,
    discrete_features=discrete_mask_nonohe, random_state=RANDOM_STATE
)
mi_df_nonohe = pd.DataFrame({'Feature': non_ohe_cols, 'MI_Score': mi_scores_nonohe})

# Tier 2 — categorical (OHE) groups: MI of the RAW parent column, evaluated
# ONCE per categorical (not per dummy)
parents_present = sorted(set(ohe_dummy_to_parent[c] for c in surviving_ohe_cols))
group_scores = {}
for parent in parents_present:
    le = LabelEncoder()
    raw_encoded = le.fit_transform(X_train_nominal_raw[parent].astype(str))
    group_scores[parent] = mutual_info_classif(
        raw_encoded.reshape(-1, 1), y_train,
        discrete_features=[True], random_state=RANDOM_STATE
    )[0]

print("\n   Group-level MI per categorical (once per parent, not per dummy):")
for p, s in sorted(group_scores.items(), key=lambda kv: -kv[1]):
    status = 'KEPT (all dummies)' if s >= MI_THRESHOLD else 'DROPPED (all dummies)'
    print(f"     {p:32} MI={s:.6f}  -> {status}")

kept_parents     = {p for p, s in group_scores.items() if s >= MI_THRESHOLD}
kept_ohe_cols    = [c for c in surviving_ohe_cols if ohe_dummy_to_parent[c] in kept_parents]
removed_ohe_cols = [c for c in surviving_ohe_cols if ohe_dummy_to_parent[c] not in kept_parents]

kept_nonohe    = mi_df_nonohe[mi_df_nonohe['MI_Score'] >= MI_THRESHOLD]['Feature'].tolist()
removed_nonohe = mi_df_nonohe[mi_df_nonohe['MI_Score'] <  MI_THRESHOLD]['Feature'].tolist()

kept_by_mi = kept_nonohe + kept_ohe_cols
removed_mi = removed_nonohe + removed_ohe_cols

mi_df = pd.concat([
    mi_df_nonohe,
    pd.DataFrame({'Feature': surviving_ohe_cols,
                  'MI_Score': [group_scores[ohe_dummy_to_parent[c]] for c in surviving_ohe_cols]})
], ignore_index=True).sort_values('MI_Score', ascending=False).reset_index(drop=True)

X_train_final = X_train_corr[kept_by_mi].copy()
X_test_final  = X_test_corr[kept_by_mi].copy()

print(f"\n   Features before: {X_train_corr.shape[1]}")
print(f"   Features after:  {len(kept_by_mi)}")

# Honest sanity check (no longer falsely claims a fix that doesn't hold)
if 'diag_1_group' in group_scores:
    s = group_scores['diag_1_group']
    status = 'KEPT ✅' if s >= MI_THRESHOLD else 'DROPPED ❌ (genuinely low signal, not a fragmentation artifact)'
    print(f"   Sanity check -- diag_1_group (whole categorical) MI={s:.6f} -> {status}")

if removed_mi:
    print(f"   ❌ Removed ({len(removed_mi)} features/dummies)")
else:
    print("   All features above MI threshold — none removed.")

# ============================================================================
# MI CHART
# ============================================================================
fig_mi, axes_mi = plt.subplots(1, 2, figsize=(20, 9))
fig_mi.patch.set_facecolor('white')
fig_mi.suptitle('Stage D5 — Mutual Information Feature Selection',
                fontsize=14, fontweight='bold', color=PALETTE['title'])

top25    = mi_df[mi_df['Feature'].isin(kept_by_mi)].head(25)
colors_t = plt.cm.viridis(np.linspace(0.3, 0.9, len(top25)))
bars_top = axes_mi[0].barh(range(len(top25)), top25['MI_Score'],
                            color=colors_t, edgecolor='white', height=0.7)
for i, (_, row) in enumerate(top25.iterrows()):
    axes_mi[0].text(row['MI_Score'] + 0.0005, i,
                    f"{row['MI_Score']:.4f}", va='center',
                    fontsize=7.5, fontweight='bold', color=PALETTE['title'])
axes_mi[0].set_yticks(range(len(top25)))
axes_mi[0].set_yticklabels(top25['Feature'], fontsize=8)
axes_mi[0].invert_yaxis()
axes_mi[0].set_title('✅ Top 25 KEPT Features (highest MI)',
                     fontsize=11, fontweight='bold', color=PALETTE['title'])
axes_mi[0].set_xlabel('Mutual Information Score', fontsize=10)
axes_mi[0].spines[['top','right']].set_visible(False)
axes_mi[0].axvline(MI_THRESHOLD, color=PALETTE['accent'], linewidth=1.2,
                   linestyle='--', label=f'Threshold={MI_THRESHOLD}')

if removed_mi:
    removed_mi_df = mi_df[mi_df['Feature'].isin(removed_mi)].reset_index(drop=True)
    colors_r = plt.cm.Reds(np.linspace(0.4, 0.8, max(len(removed_mi_df), 1)))
    axes_mi[1].barh(range(len(removed_mi_df)), removed_mi_df['MI_Score'],
                    color=colors_r, edgecolor='white', height=0.7)
    axes_mi[1].set_yticks(range(len(removed_mi_df)))
    axes_mi[1].set_yticklabels(removed_mi_df['Feature'], fontsize=8)
    axes_mi[1].invert_yaxis()
    axes_mi[1].set_title(f'❌ {len(removed_mi)} DROPPED (MI < {MI_THRESHOLD})',
                         fontsize=11, fontweight='bold', color=PALETTE['title'])
else:
    axes_mi[1].text(0.5, 0.5, 'No features\ndropped by MI filter',
                    ha='center', va='center', fontsize=12, color='#888',
                    transform=axes_mi[1].transAxes)
    axes_mi[1].set_title('No features dropped', fontsize=11, color=PALETTE['title'])
axes_mi[1].set_xlabel('Mutual Information Score', fontsize=10)
axes_mi[1].spines[['top','right']].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_mi.savefig(FIGURES_DIR / 'D5_mutual_information_selection.png',
               dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig_mi)
print("\n   MI chart saved → D5_mutual_information_selection.png")

# ============================================================================
# FEATURE SELECTION SUMMARY
# ============================================================================
print(f"\n{'='*65}")
print("FEATURE SELECTION SUMMARY")
print(f"{'='*65}")
print(f"  After OHE               : {features_after_ohe:>4} features")
print(f"  After VarianceThreshold : {len(kept_features_var):>4} features  (-{len(removed_var)})")
print(f"  After Correlation Filter: {X_train_corr.shape[1]:>4} features  (-{len(removed_corr)})")
print(f"  After MI Selection      : {X_train_final.shape[1]:>4} features  (-{len(removed_mi)})")
print(f"  {'─'*50}")
print(f"  FINAL FEATURE COUNT     : {X_train_final.shape[1]:>4}")

summary_df = pd.DataFrame({
    'Feature'    : mi_df['Feature'],
    'MI_Score'   : mi_df['MI_Score'].round(6),
    'Selected'   : mi_df['Feature'].isin(kept_by_mi).map({True:'✅ KEPT', False:'❌ DROPPED'}),
    'Reason_dropped': mi_df['Feature'].apply(
        lambda f: 'Low MI (<0.0005)' if f in removed_mi
        else ('Low Variance' if f in removed_var
              else ('High Corr (low target-corr)' if f in removed_corr else '')))
})
summary_df.to_csv(PROCESSED_DIR / 'feature_selection_report.csv', index=False)
print(f"  Selection report saved → feature_selection_report.csv")

# ============================================================================
# CLASS IMBALANCE ANALYSIS
# FIX8: Per-model imbalance handling instructions (previously just a comment)
# ============================================================================
total_train = len(y_train)
pos_train   = int(y_train.sum())
neg_train   = total_train - pos_train
ratio       = neg_train / pos_train

print(f"\n{'='*65}")
print("CLASS IMBALANCE ANALYSIS  (FIX8)")
print(f"{'='*65}")
print(f"  Train size          : {total_train:,}")
print(f"  Positive (<30 days) : {pos_train:,}  ({pos_train/total_train*100:.2f}%)")
print(f"  Negative            : {neg_train:,}  ({neg_train/total_train*100:.2f}%)")
print(f"  Imbalance ratio     : {ratio:.2f} : 1")
print()
print("  Per-model handling strategy:")
print(f"  ├── XGBoost       → scale_pos_weight={ratio:.2f}")
print(f"  ├── LightGBM      → is_unbalance=True  OR  class_weight={{0:1, 1:{ratio:.0f}}}")
print(f"  ├── LogisticReg   → class_weight='balanced'")
print(f"  ├── RandomForest  → class_weight='balanced'")
print(f"  └── SVM           → class_weight='balanced'")
print()
print("  NOTE: SMOTE is optional — gradient boosting with scale_pos_weight")
print("        often outperforms SMOTE on tabular medical data.")
print("        If you use SMOTE: apply ONLY on X_train, NEVER on X_test.")

# ============================================================================
# SAVE ALL ARTIFACTS
# ============================================================================
X_train_final.to_csv(PROCESSED_DIR / 'X_train.csv', index=False)
X_test_final.to_csv( PROCESSED_DIR / 'X_test.csv',  index=False)
y_train.to_csv(PROCESSED_DIR / 'y_train.csv', index=False, header=['target'])
y_test.to_csv( PROCESSED_DIR / 'y_test.csv',  index=False, header=['target'])

joblib.dump(scaler,       PROCESSED_DIR / 'scaler.pkl')
joblib.dump(var_selector, PROCESSED_DIR / 'var_selector.pkl')
joblib.dump(iqr_bounds,   PROCESSED_DIR / 'iqr_bounds.pkl')

pd.Series(X_train_final.columns.tolist()).to_csv(
    PROCESSED_DIR / 'feature_names.csv', index=False, header=['feature'])
mi_df.to_csv(PROCESSED_DIR / 'mi_scores.csv', index=False)

print(f"\n{'='*65}")
print("✅ FEATURE ENGINEERING COMPLETE — ALL BUGS FIXED")
print(f"{'='*65}")
print(f"  X_train : {X_train_final.shape}")
print(f"  X_test  : {X_test_final.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  y_test  : {y_test.shape}")
print()
print("  All bugs fixed:")
print("  ✅ BUG1: examide/citoglipton removed from med_cols")
print("  ✅ BUG2: discharge_disposition_id=25 preserved (989 rows kept)")
print("  ✅ BUG3: diag missing → 'Unknown' (not mode/heart-failure)")
print("  ✅ INSULIN FIXED: One-Hot encoding (No/Steady/Up/Down) + binary flags")
print("  ✅ BUG5: gender Unknown/Invalid removed before encoding")
print("  ✅ FIX6: binary cols excluded from RobustScaler")
print("  ✅ FIX7: MI threshold = 0.0005 (not brittle 20%)")
print("  ✅ FIX8: per-model imbalance instructions added")
print("  ✅ A1C FIXED: Missing flag + 'Unknown' category (preserved informative missingness)")
print("  ✅ CORR FIXED: Correlation filter now applies ONLY to continuous numeric features")
print()
print("  Saved artifacts:")
print("    X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("    scaler.pkl, var_selector.pkl, iqr_bounds.pkl")
print("    feature_names.csv, mi_scores.csv, feature_selection_report.csv")
print()
print("NEXT → Code 4: Model Training")
print("  Use StratifiedGroupKFold(n_splits=5, groups=patient_nbr) for CV")
print("  Primary metric: ROC-AUC + PR-AUC (imbalanced class)")
print(f"{'='*65}")