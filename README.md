# Hospital Readmission Prediction

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20Dashboard-2ea44f?style=for-the-badge)](https://hospital-readmission-prediction-chi.vercel.app)

---

## Overview

This project answers one question: *"Is this patient likely to be readmitted within 30 days?"* It is
built on 99,340 hospital encounters, using patient-level splitting so no patient's data ever appears in
both train and test. A tuned, calibrated LightGBM model produces a readmission-risk probability per
encounter, served live through a results dashboard and a real-time Case Checker.

---

## Aim

- Predict whether a diabetic patient will be readmitted within 30 days of discharge
- Rank patients by risk so care teams can prioritize follow-up for those most likely to bounce back
- Keep every reported probability honestly calibrated (a "60%" case really means about 60%)
- Score any new patient encounter live through a real-time Case Checker
- Present model performance, fairness, and cost-benefit results through one dashboard

---

## Problem Statement

When a diabetic patient is readmitted to hospital within 30 days of discharge, it is costly for the
hospital and often a sign that the patient needed more support after going home. Care teams can prevent
some of these readmissions with follow-up calls, visits, or medication checks -- but their time is
limited, and they cannot give every discharged patient the same level of attention.

Deciding who needs follow-up is hard for several reasons:

- **Risk is hidden in many small signals** -- diagnoses, lab tests, medications, and the patient's own
  history of past visits all matter, and no one can weigh all of them by hand for every patient.
- **A raw risk score is not a real probability** -- unless it is calibrated, a score cannot be trusted at
  face value for planning.
- **Mistakes do not cost the same** -- missing an at-risk patient is more costly than making one
  unnecessary follow-up call.
- **Data leakage can make a model look better than it is** -- if the same patient appears in both training
  and testing, results are overly optimistic.

The goal is to give care teams a trustworthy, calibrated readmission risk for each patient encounter, so
that limited follow-up resources go to the patients who need them most.

---

## Key Features

- Patient-level train/test and cross-validation splitting so no patient's data ever leaks between sets
- Hidden-missing-code detection and informative-missingness preservation for lab tests
- Patient-history features (prior encounters and prior readmissions for that patient)
- Refined, clinically grouped diagnosis categories
- Platt-calibrated probabilities, verified with an Expected Calibration Error check
- Both an F1-optimal and a cost-sensitive decision threshold
- Live **Case Checker** -- score any real patient encounter with the actual trained model
- Fairness audit, risk stratification (Low/Medium/High), and a cost-benefit simulation
- SHAP explainability, calibration curves, and error analysis
- One-command results dashboard; every number shown comes from the real model, not a static demo

---

## Architecture

<img src="docs/assets/architecture.png" width="850">

Data flows in one direction, start to finish: `data_cleaning.py` is the single source of truth for
cleaning decisions, `feature_engineering.py` consumes its output, `model_training.py` trains, calibrates,
and selects thresholds using only out-of-fold predictions, and the live Case Checker reuses the exact same
saved pipeline artifacts for every new prediction.

---

## Dashboard

**Case Checker** -- describe a patient's encounter and get the real model's calibrated 30-day
readmission risk, the decision threshold, and the cost-sensitive threshold side by side.

<img src="docs/assets/screenshots/01_case_checker.png" width="800">

---

## Benefits

- **Better prioritization** -- limited follow-up resources go to the patients who need them most
- **Trustworthy probabilities** -- calibration is checked and corrected, so risk scores can be used at
  face value, not just for ranking
- **Live scoring** -- any new patient encounter can be scored instantly through the Case Checker, using
  the exact same pipeline as training
- **Cost-aware decisions** -- a cost-sensitive threshold reflects that missing an at-risk patient is more
  costly than an unnecessary follow-up call
- **Fairness screening** -- selection rates and recall are checked across gender and race groups
- **Clear business case** -- a cost-benefit simulation shows the potential savings from outreach to
  high-risk patients

---

## Conclusion

This project turns 99,340 hospital encounters into a practical tool for deciding which diabetic patients
need follow-up after discharge. Patient-level splitting keeps the evaluation honest, calibration makes the
risk scores usable at face value, and the cost-sensitive threshold reflects the real trade-off between
missing an at-risk patient and making an extra follow-up call.

With fairness checks, SHAP explanations, and a live Case Checker that uses the exact same pipeline as
training, the model supports care teams in prioritizing their limited follow-up resources -- while final
decisions about each patient stay with the clinicians who know them.
