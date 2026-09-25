"""ADDITION 8 -- Cost-benefit simulation: converts the High-risk bucket
(from ADDITION 7) into a labeled dollar estimate under stated, adjustable
assumptions -- same discipline as a payment-proxy: every number here is
explicitly an assumption, not a measured fact.
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "reports" / "results"

strat_path = RESULTS_DIR / "addition7_risk_stratification.csv"
if not strat_path.exists():
    raise FileNotFoundError(f"{strat_path} not found -- run risk_stratification.py first.")

strat = pd.read_csv(strat_path)
high_row = strat[strat["bucket"] == "High"].iloc[0]
n_high = int(high_row["n"])
actual_rate_high = high_row["actual_readmit_rate"]

# -- Labeled assumptions (not measured facts; change these and re-run) --
COST_PER_OUTREACH_CALL = 50          # dollars, per patient contacted
COST_PER_READMISSION = 15000         # dollars, a commonly cited public average 30-day
                                      # readmission cost estimate -- not this hospital's own figure
EFFECTIVENESS_SCENARIOS = {"low": 0.10, "base": 0.20, "high": 0.30}  # % of readmissions
                                      # in the High-risk bucket assumed preventable by outreach

print(f"High-risk bucket: {n_high:,} patients, actual readmission rate {actual_rate_high:.1%}")
expected_readmissions_high = n_high * actual_rate_high
print(f"Expected readmissions in this bucket without intervention: {expected_readmissions_high:,.0f}")

rows = []
outreach_cost = n_high * COST_PER_OUTREACH_CALL
for scenario, effectiveness in EFFECTIVENESS_SCENARIOS.items():
    prevented = expected_readmissions_high * effectiveness
    savings = prevented * COST_PER_READMISSION
    net = savings - outreach_cost
    rows.append({
        "scenario": scenario,
        "effectiveness_assumption": effectiveness,
        "outreach_cost": outreach_cost,
        "readmissions_prevented": prevented,
        "gross_savings": savings,
        "net_savings": net,
    })

result = pd.DataFrame(rows)
print(result.to_string(index=False))
print(f"\nAssumptions used (change and re-run to test sensitivity):")
print(f"  Cost per outreach call: ${COST_PER_OUTREACH_CALL}")
print(f"  Cost per prevented readmission (public average estimate): ${COST_PER_READMISSION:,}")
print(f"  Effectiveness of outreach at preventing a readmission: {EFFECTIVENESS_SCENARIOS}")

result.to_csv(RESULTS_DIR / "addition8_cost_benefit_simulation.csv", index=False)
print(f"\nSaved: {RESULTS_DIR / 'addition8_cost_benefit_simulation.csv'}")
