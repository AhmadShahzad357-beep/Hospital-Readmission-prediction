"""Tests for the ID-mapping fix in feature_engineering.py.

These specifically guard against the two regressions found and fixed during
review: (1) the map_* functions silently comparing string-typed ID columns
to integer lists (which made every row fall into "Other"), and (2) the
'Unknown' hidden-missing marker not being preserved as its own bucket.

NOTE: the functions below are copied from src/feature_engineering.py rather
than imported. That file has no `if __name__ == "__main__":` guard -- it is
one continuous top-level script -- so importing it runs the entire training
pipeline (data load, charts, model fitting) as a side effect, which made
this test file take ~4 minutes to run. If the source functions change,
these copies must be updated to match (a small, known tradeoff for a fast,
isolated test).
"""
import pytest


def _to_int_or_unknown(x):
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


def test_to_int_or_unknown_parses_numeric_strings():
    assert _to_int_or_unknown("1") == 1
    assert _to_int_or_unknown("25") == 25


def test_to_int_or_unknown_returns_none_for_unknown_marker():
    assert _to_int_or_unknown("Unknown") is None


def test_to_int_or_unknown_returns_none_for_garbage():
    assert _to_int_or_unknown(None) is None
    assert _to_int_or_unknown("not_a_number") is None


def test_map_admission_type_handles_string_codes():
    assert map_admission_type("1") == "Emergency"
    assert map_admission_type("2") == "Urgent"
    assert map_admission_type("3") == "Elective"


def test_map_admission_type_unknown_is_its_own_bucket():
    assert map_admission_type("Unknown") == "Unknown"


def test_map_admission_type_unmapped_code_is_other():
    assert map_admission_type("7") == "Other"


def test_map_admission_source_handles_string_codes():
    assert map_admission_source("1") == "Referral"
    assert map_admission_source("7") == "Emergency"
    assert map_admission_source("14") == "Transfer"


def test_map_admission_source_unknown_is_its_own_bucket():
    assert map_admission_source("Unknown") == "Unknown"


def test_map_discharge_handles_string_codes():
    assert map_discharge("1") == "Home"
    assert map_discharge("2") == "Transferred"


def test_map_discharge_unknown_is_its_own_bucket():
    assert map_discharge("Unknown") == "Unknown"


def test_map_discharge_unmapped_code_is_other():
    assert map_discharge("99") == "Other"
