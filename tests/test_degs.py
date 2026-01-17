import pandas as pd

from mondrian_map.data_processing import filter_genes_by_expression_threshold
from mondrian_map.degs import (
    call_degs_by_fc_thresholds,
    compute_fold_change,
    compute_profile_degs,
)


def test_compute_fold_change_flooring():
    numerator = pd.Series([0.0, 2.0], index=["G1", "G2"])
    denominator = pd.Series([0.0, 1.0], index=["G1", "G2"])
    fc = compute_fold_change(numerator, denominator, min_value=0.001)
    assert fc.loc["G1"] == 1.0
    assert fc.loc["G2"] == 2.0


def test_call_degs_thresholds_matches_fc_ratio_spec():
    fc = pd.Series([2.0, 1.0, 0.4], index=["A", "B", "C"])
    up, down = call_degs_by_fc_thresholds(fc, up_threshold=1.5, down_threshold=0.5)
    assert list(up) == ["A"]
    assert list(down) == ["C"]


def test_compute_profile_degs_returns_two_contrasts_with_expected_counts_on_synthetic_data():
    profile_expr = {
        "TP": pd.Series([1.0, 1.0, 1.0], index=["G1", "G2", "G3"]),
        "R1": pd.Series([2.0, 0.4, 1.0], index=["G1", "G2", "G3"]),
        "R2": pd.Series([1.0, 2.0, 0.4], index=["G1", "G2", "G3"]),
    }
    degs = compute_profile_degs(profile_expr, min_value=0.001, up_threshold=1.5, down_threshold=0.5)
    assert degs["R1_vs_TP"]["up"].tolist() == ["G1"]
    assert degs["R1_vs_TP"]["down"].tolist() == ["G2"]
    assert degs["R2_vs_TP"]["up"].tolist() == ["G2"]
    assert degs["R2_vs_TP"]["down"].tolist() == ["G3"]


def test_filter_genes_by_expression_threshold_drops_low_genes_and_records_stats():
    tp = pd.DataFrame({"A": [0.0, 0.01]}, index=["G1", "G2"])
    r1 = pd.DataFrame({"A": [0.0, 0.02]}, index=["G1", "G2"])
    r2 = pd.DataFrame({"A": [0.0, 0.03]}, index=["G1", "G2"])
    ftp, fr1, fr2, stats = filter_genes_by_expression_threshold(tp, r1, r2, threshold=0.001)
    assert list(ftp.index) == ["G2"]
    assert stats["genes_before"] == 2
    assert stats["genes_after"] == 1
