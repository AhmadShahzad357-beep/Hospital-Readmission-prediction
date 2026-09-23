"""
====================================================================
CODE 1 — DATA CLEANING (v3 — hidden missing codes + rare-med fix)
Hospital Readmission Prediction — Diabetic Data
====================================================================
"""

import pandas as pd
import numpy as np
import os

# ==================== 1. LOAD DATA ====================
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / 'data' / 'raw' / 'diabetic_data.csv'
df = pd.read_csv(file_path, na_values='?', low_memory=False)

print("===== Dataset Shape (Raw) =====")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

# ==================== 2. BASIC INFO ====================
print("===== First 5 Rows =====")
print(df.head(), "\n")

print("===== Data Types & Non-Null Counts =====")
print(df.info(), "\n")

print("===== Statistical Summary (Numerical Columns) =====")
print(df.describe(), "\n")

# ==================== 3. MISSING VALUES (ALL COLUMNS) ====================
missing_counts  = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100
missing_all = pd.DataFrame({
    'Missing Count'      : missing_counts,
    'Missing Percentage' : missing_percent
}).sort_values(by='Missing Percentage', ascending=False)

print("===== Missing Values per Column (isnull-visible only) =====")
print(missing_all[missing_all['Missing Count'] > 0], "\n")

# ==================== 4. DUPLICATE ROWS ====================
duplicate_count = df.duplicated().sum()
print(f"===== Exact Duplicate Rows Found: {duplicate_count} ({(duplicate_count/len(df))*100:.2f}%) =====")
if duplicate_count > 0:
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} exact duplicate rows.")
else:
    print("No exact duplicate rows found — nothing to drop.")
print(f"Shape after duplicate check: {df.shape}\n")

# ==================== 5. TARGET VARIABLE ====================
print("===== Target Column 'readmitted' Value Counts =====")
print(df['readmitted'].value_counts(dropna=False), "\n")
print("NOTE (Class Imbalance): '<30' class is the minority (~11%).")
print("      Agar binary classification bana rahe ho (<30 vs rest), to")
print("      class_weight='balanced' ya stratified split use karna.")
print("      SMOTE/oversampling agar use karo to SIRF training fold par,")
print("      train/test split ke BAAD — pehle karne se leakage hoga.\n")

# ==================== 6. PATIENT VISIT ANALYSIS ====================
# DECISION: Pura data rakha ja raha hai — koi encounter drop NAHI kiya
# ja raha. Leakage prevention Code 2 mein GroupShuffleSplit/GroupKFold
# (groups=patient_nbr) se hogi, row-drop se nahi.
print("===== Repeated Patient Encounters =====")
patient_visit_counts = df['patient_nbr'].value_counts()
multi_visit_patients = (patient_visit_counts > 1).sum()
multi_visit_encounters = patient_visit_counts[patient_visit_counts > 1].sum()
print(f"Unique patients          : {df['patient_nbr'].nunique()}")
print(f"Patients with >1 visit   : {multi_visit_patients}")
print(f"Encounters from repeaters: {multi_visit_encounters}")
print("DECISION: Full data retained. Leakage will be prevented via")
print("          GroupShuffleSplit/GroupKFold(groups=patient_nbr) in Code 2.\n")

# ==================== 7. REMOVE CLINICALLY INVALID ROWS ====================

# --- 7a. Remove deceased / hospice patients ---
dead_discharge_ids = [11, 13, 14, 19, 20, 21]
before = len(df)
df = df[~df['discharge_disposition_id'].isin(dead_discharge_ids)]
print(f"Removed deceased/hospice patients: {before - len(df)} rows removed")
print(f"Shape after removal: {df.shape}\n")

# --- 7b. Remove gender = 'Unknown/Invalid' (only 3 rows — not imputable) ---
before = len(df)
df = df[df['gender'] != 'Unknown/Invalid']
print(f"Removed 'Unknown/Invalid' gender rows: {before - len(df)} rows removed")
print(f"Shape after removal: {df.shape}\n")

# --- 7c. NEW: UNMASK HIDDEN MISSING VALUES (numeric-coded) ---
# admission_type_id, discharge_disposition_id, admission_source_id are
# CATEGORICAL CODES, not real numeric quantities. Per the dataset's
# IDs_mapping reference, some of these codes actually mean
# "Not Available / NULL / Not Mapped" -- i.e. they ARE missing values,
# just disguised as valid-looking integers. isnull() cannot catch these
# because the column dtype is numeric. If left as-is, Code 3 would
# one-hot encode them as if they were meaningful categories, silently
# injecting noise into the model.
print("=" * 55)
print("UNMASKING HIDDEN MISSING VALUES (numeric-coded categories)")
print("=" * 55)
hidden_missing_map = {
    'admission_type_id':        [5, 6, 8],
    'discharge_disposition_id': [18, 25, 26],
    'admission_source_id':      [9, 15, 17, 20, 21],
}
for col, codes in hidden_missing_map.items():
    n_hidden = df[col].isin(codes).sum()
    df[col] = df[col].replace(codes, np.nan)
    print(f"  {col:<28} | unmasked {n_hidden:>6} hidden-missing rows "
          f"({n_hidden/len(df)*100:.2f}%)")

# These 3 columns are categorical IDs (ID 5 is not "more" than ID 1), so
# cast to string category dtype and fill missing with 'Unknown' -- same
# treatment already used for payer_code/medical_specialty for consistency.
# NOTE: astype('Int64').astype('str') is unreliable across pandas versions
# for representing <NA> as a string -- using an explicit apply instead.
for col in hidden_missing_map:
    df[col] = df[col].apply(lambda x: 'Unknown' if pd.isna(x) else str(int(x)))
print("  -> Cast to categorical string + filled hidden-missing with 'Unknown'\n")

# ==================== 8. DROP HIGH-MISSINGNESS COLUMNS (>80%) ====================
cols_high_missing = ['weight']
df.drop(columns=cols_high_missing, inplace=True)
print(f"Dropped high-missing columns (no signal): {cols_high_missing}")

# A1Cresult / max_glu_serum: informative-missingness preserved (test not
# ordered is itself a clinical signal, not random missingness).
df['A1C_missing']  = df['A1Cresult'].isna().astype(int)
df['A1C_category'] = df['A1Cresult'].fillna('Not Tested')
df.drop('A1Cresult', axis=1, inplace=True)
print("A1Cresult PRESERVED: missing flag + 'Not Tested' category created")

df['glu_serum_missing']  = df['max_glu_serum'].isna().astype(int)
df['glu_serum_category'] = df['max_glu_serum'].fillna('Not Tested')
df.drop('max_glu_serum', axis=1, inplace=True)
print("max_glu_serum PRESERVED: missing flag + 'Not Tested' category created")
print(f"Shape after high-missingness column processing: {df.shape}\n")

# ==================== 9. NEAR-ZERO-VARIANCE MEDICATION COLUMNS ====================
# FIX (was: drop entirely -> data/information loss):
# Rare medications (near-100% "No") individually carry almost no
# statistical variance, BUT a patient being on ANY rare/unusual
# medication can itself be a signal of an atypical or more complex
# case. Instead of deleting that information, consolidate all
# near-zero-variance medication columns into ONE numeric feature
# before dropping them -- no information is lost, just compressed.
print("=" * 55)
print("NEAR-ZERO-VARIANCE MEDICATION COLUMNS -> CONSOLIDATE (not drop blind)")
print("=" * 55)
NEAR_ZERO_THRESHOLD = 0.99
med_like_cols = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
                  'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
                  'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
                  'examide','citoglipton','insulin','glyburide-metformin',
                  'glipizide-metformin','glimepiride-pioglitazone',
                  'metformin-rosiglitazone','metformin-pioglitazone']
med_like_cols = [c for c in med_like_cols if c in df.columns]

rare_med_cols = []
for col in med_like_cols:
    top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
    if top_freq >= NEAR_ZERO_THRESHOLD:
        rare_med_cols.append(col)
        print(f"  {col:<30} | top value share: {top_freq*100:.2f}%  -> fold into summary feature")

if rare_med_cols:
    # 1 if patient was ever put on ANY of these rare meds, summed across columns
    df['num_rare_meds_used'] = (df[rare_med_cols] != 'No').sum(axis=1)
    df.drop(columns=rare_med_cols, inplace=True)
    print(f"\nConsolidated {len(rare_med_cols)} rare-medication columns into "
          f"'num_rare_meds_used' (range 0-{len(rare_med_cols)}).")
    print(f"Patients with >=1 rare medication: "
          f"{(df['num_rare_meds_used']>0).sum()} ({(df['num_rare_meds_used']>0).mean()*100:.2f}%)")
else:
    print("No near-zero-variance medication columns found.")

# Any remaining non-medication object columns still at near-zero variance
# (unlikely here, but keep the systematic safety-net check for other cols)
other_object_cols = [c for c in df.select_dtypes(include='object').columns
                      if c not in med_like_cols]
other_zero_var = []
for col in other_object_cols:
    top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
    if top_freq >= NEAR_ZERO_THRESHOLD:
        other_zero_var.append(col)
if other_zero_var:
    print(f"Also near-zero-variance (non-medication, dropped): {other_zero_var}")
    df.drop(columns=other_zero_var, inplace=True)
print(f"Shape after medication consolidation: {df.shape}\n")

# ==================== 10. HANDLE REMAINING MISSING VALUES ====================
df['payer_code']         = df['payer_code'].fillna('Unknown')
df['medical_specialty']  = df['medical_specialty'].fillna('Unknown')
print("Filled payer_code & medical_specialty with 'Unknown'")

race_mode = df['race'].mode()[0]
df['race'] = df['race'].fillna(race_mode)
print(f"Filled race with mode: '{race_mode}'")

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].fillna('Unknown')
print("Filled diag_1 / diag_2 / diag_3 with 'Unknown' (rows NOT dropped)")
print("NOTE (High Cardinality): diag_1/2/3 are raw ICD-9 codes with hundreds")
print("      of unique values. Do NOT one-hot encode directly. Group into")
print("      broader disease categories in Code 3.\n")

# ==================== 11. VERIFY NO MISSING VALUES REMAIN ====================
print("===== Remaining Missing Values (post hidden-code unmasking + fills) =====")
remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]
if len(remaining_missing) > 0:
    print(remaining_missing)
else:
    print("No missing values remaining (including previously-hidden numeric-coded ones). ✓")
print()

# ==================== 12. ID COLUMNS — MARK, DO NOT USE AS FEATURES ====================
id_columns = ['encounter_id', 'patient_nbr']
print(f"ID columns retained for grouping only (EXCLUDE from model features later): {id_columns}\n")

# ==================== 13. OUTLIER DETECTION (Reporting Only) ====================
# Capping deferred to Code 3, AFTER train/test split, to avoid fitting
# bounds on data that includes the test set (leakage).
continuous_cols = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications',  'number_outpatient',  'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

print("=" * 55)
print("OUTLIER DETECTION REPORT (Detection only — no capping here)")
print("=" * 55)
for col in continuous_cols:
    if col in df.columns:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lb  = Q1 - 1.5 * IQR
        ub  = Q3 + 1.5 * IQR
        n_out = df[(df[col] < lb) | (df[col] > ub)].shape[0]
        print(f"  {col:<25} | Outliers: {n_out:>5} ({n_out/len(df)*100:.2f}%)"
              f" | Range: [{df[col].min()}, {df[col].max()}]"
              f" | Cap bounds: [{lb:.1f}, {ub:.1f}]")

print()
print("NOTE: number_outpatient & number_emergency are zero-inflated distributions.")
print("      IQR flags all non-zero values as outliers — do NOT blindly cap these.")
print("      Consider log1p transform in Code 3 instead.\n")

# ==================== 14. FINAL DATASET SUMMARY ====================
print("=" * 55)
print("FINAL CLEANED DATASET SUMMARY")
print("=" * 55)
print(f"  Shape        : {df.shape}")
print(f"  Numeric cols : {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"  Object cols  : {df.select_dtypes(include='object').shape[1]}")
print(f"\n  Target distribution (readmitted):")
vc = df['readmitted'].value_counts()
for label, count in vc.items():
    print(f"    {label:<5} : {count:>6}  ({count/len(df)*100:.1f}%)")
print()
print("NEXT STEPS:")
print("  Code 2 → Train/Test split using GroupShuffleSplit(groups=patient_nbr)")
print("           + GroupKFold for any CV/tuning (patient never split across sets)")
print("  Code 3 → Feature Engineering:")
print("             - Outlier capping (fit bounds on TRAIN only)")
print("             - ICD-9 grouping for diag_1/2/3 (avoid high-cardinality one-hot)")
print("             - Encoding (fit on TRAIN only, transform on test)")
print("             - Scaling (fit on TRAIN only, transform on test)")
print("  Code 4 → Model Training & Evaluation (class_weight/stratification for")
print("           imbalance; SMOTE, if used, only on training fold post-split)\n")

# ==================== 15. SAVE PROCESSED DATA ====================
save_dir  = BASE_DIR / 'data' / 'processed'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'diabetic_data_cleaned.csv')

df.to_csv(save_path, index=False)
print(f"Cleaned data saved to : {save_path}")
print(f"Final shape           : {df.shape}")