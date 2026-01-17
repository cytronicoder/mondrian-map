import pandas as pd

from mondrian_map.pathway_stats import (compute_pathway_wfc_table,
                                        compute_weighted_fold_change)


def test_compute_weighted_fold_change_known_example():
    fc = pd.Series({"A": 2.0, "B": 1.0})
    rp_scores = {"A": 1.0, "B": 1.0}
    wfc = compute_weighted_fold_change(fc, rp_scores)
    assert wfc == 1.5


def test_compute_pathway_wfc_table_records_coverage_and_is_deterministic():
    pag_df = pd.DataFrame({"GS_ID": ["P1", "P2"]})
    fc = pd.Series({"G1": 2.0, "G2": 0.5})
    rp_scores_map = {
        "P1": {"G1": 1.0, "G2": 1.0},
        "P2": {"G2": 1.0, "G3": 1.0},
    }
    result = compute_pathway_wfc_table(pag_df, fc, rp_scores_map)
    assert "wFC_gene_coverage" in result.columns
    assert result.loc[result["GS_ID"] == "P1", "wFC_gene_coverage"].iloc[0] == 1.0
    assert result.loc[result["GS_ID"] == "P2", "wFC_genes_used"].iloc[0] == 1
