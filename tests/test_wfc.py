"""
Tests for wFC (weighted Fold Change) computation.

These tests verify that the wFC formula matches the paper specification:
    wFC = sum(W_i * FC_i) / sum(W_i)
where W_i = RP_SCORE and FC_i = FoldChange
"""

import pandas as pd
import pytest

from mondrian_map.pathway_stats import compute_pathway_wfc, compute_wfc


class TestWFCComputation:
    """Test suite for weighted fold change computation."""

    def test_simple_wfc(self):
        """Test weighted fold-change with standard test values."""
        # Standard case: two genes
        # Gene 1: log fold-change=2.0, weight=1.0
        # Gene 2: log fold-change=4.0, weight=1.0
        # Expected wFC = (1*2 + 1*4) / (1+1) = 3.0
        gene_fc = pd.Series([2.0, 4.0], index=["GENE1", "GENE2"])
        weights = pd.Series([1.0, 1.0], index=["GENE1", "GENE2"])

        result = compute_wfc(gene_fc, weights)
        assert result == pytest.approx(3.0)

    def test_weighted_wfc(self):
        """Test wFC with different weights."""
        # Gene 1: FC=2.0, weight=3.0
        # Gene 2: FC=4.0, weight=1.0
        # Expected wFC = (3*2 + 1*4) / (3+1) = 10/4 = 2.5
        gene_fc = pd.Series([2.0, 4.0], index=["GENE1", "GENE2"])
        weights = pd.Series([3.0, 1.0], index=["GENE1", "GENE2"])

        result = compute_wfc(gene_fc, weights)
        assert result == pytest.approx(2.5)

    def test_single_gene(self):
        """Test wFC with a single gene returns that gene's FC."""
        gene_fc = pd.Series([1.5], index=["GENE1"])
        weights = pd.Series([2.0], index=["GENE1"])

        result = compute_wfc(gene_fc, weights)
        assert result == pytest.approx(1.5)

    def test_zero_weights(self):
        """Test wFC with all zero weights returns 0.0."""
        gene_fc = pd.Series([2.0, 4.0], index=["GENE1", "GENE2"])
        weights = pd.Series([0.0, 0.0], index=["GENE1", "GENE2"])

        result = compute_wfc(gene_fc, weights)
        # Implementation returns 0.0 for zero weights
        assert result == 0.0

    def test_empty_series(self):
        """Test wFC with empty series returns 0.0."""
        gene_fc = pd.Series([], dtype=float)
        weights = pd.Series([], dtype=float)

        result = compute_wfc(gene_fc, weights)
        assert result == 0.0

    def test_negative_fold_changes(self):
        """Test wFC with negative fold changes (down-regulated)."""
        # Using log2 fold changes
        gene_fc = pd.Series([-2.0, -1.0, 1.0], index=["G1", "G2", "G3"])
        weights = pd.Series([1.0, 1.0, 1.0], index=["G1", "G2", "G3"])

        # Expected: (-2 + -1 + 1) / 3 = -2/3
        result = compute_wfc(gene_fc, weights)
        assert result == pytest.approx(-2 / 3, rel=1e-3)

    def test_partial_overlap(self):
        """Test wFC when gene_fc and weights have partial overlap."""
        gene_fc = pd.Series([2.0, 4.0, 6.0], index=["G1", "G2", "G3"])
        weights = pd.Series([1.0, 1.0], index=["G1", "G2"])  # Missing G3

        # Should only use common genes G1 and G2
        result = compute_wfc(gene_fc, weights)
        assert result == pytest.approx(3.0)


class TestPathwayWFC:
    """Test suite for pathway-level wFC computation."""

    def test_pathway_wfc_from_dataframe(self):
        """Test computing wFC from a ranked genes DataFrame."""
        # Create mock ranked genes data
        ranked_genes = pd.DataFrame(
            {
                "GENE_SYM": ["GENE1", "GENE2", "GENE3"],
                "RP_SCORE": [0.8, 0.5, 0.3],
            }
        )

        # Fold change DataFrame with genes as index
        fold_change_df = pd.DataFrame(
            {"fold_change": [2.0, 1.5, 3.0]}, index=["GENE1", "GENE2", "GENE3"]
        )

        # Expected: (0.8*2 + 0.5*1.5 + 0.3*3) / (0.8+0.5+0.3)
        # = (1.6 + 0.75 + 0.9) / 1.6 = 3.25 / 1.6 = 2.03125
        result = compute_pathway_wfc(
            pag_id="WP001",
            ranked_genes_df=ranked_genes,
            fold_change_df=fold_change_df,
            fc_column="fold_change",
        )
        assert result == pytest.approx(3.25 / 1.6, rel=1e-3)

    def test_pathway_wfc_missing_genes(self):
        """Test wFC when some genes are not in fold change lookup."""
        ranked_genes = pd.DataFrame(
            {
                "GENE_SYM": ["GENE1", "GENE2", "MISSING"],
                "RP_SCORE": [1.0, 1.0, 1.0],
            }
        )

        fold_change_df = pd.DataFrame(
            {"fold_change": [2.0, 4.0]}, index=["GENE1", "GENE2"]
        )  # MISSING not present

        # Only GENE1 and GENE2 should be used
        # Expected: (1*2 + 1*4) / 2 = 3.0
        result = compute_pathway_wfc(
            pag_id="WP001",
            ranked_genes_df=ranked_genes,
            fold_change_df=fold_change_df,
            fc_column="fold_change",
        )
        assert result == pytest.approx(3.0)

    def test_pathway_wfc_all_missing(self):
        """Test wFC when no genes match returns 0.0."""
        ranked_genes = pd.DataFrame(
            {
                "GENE_SYM": ["MISSING1", "MISSING2"],
                "RP_SCORE": [1.0, 1.0],
            }
        )

        fold_change_df = pd.DataFrame({"fold_change": [2.0]}, index=["OTHER"])

        result = compute_pathway_wfc(
            pag_id="WP001",
            ranked_genes_df=ranked_genes,
            fold_change_df=fold_change_df,
            fc_column="fold_change",
        )
        assert result == 0.0


class TestWFCEdgeCases:
    """Test edge cases and numerical stability."""

    def test_large_weights(self):
        """Test weighted fold-change with large weight magnitudes."""
        gene_fc = pd.Series([1.0, 2.0], index=["G1", "G2"])
        weights = pd.Series([1e10, 1e10], index=["G1", "G2"])

        result = compute_wfc(gene_fc, weights)
        assert result == pytest.approx(1.5)

    def test_small_weights(self):
        """Test weighted fold-change with small weight magnitudes."""
        gene_fc = pd.Series([1.0, 2.0], index=["G1", "G2"])
        weights = pd.Series([1e-10, 1e-10], index=["G1", "G2"])

        result = compute_wfc(gene_fc, weights)
        assert result == pytest.approx(1.5)

    def test_mixed_scale_weights(self):
        """Test wFC with weights of different scales."""
        gene_fc = pd.Series([1.0, 2.0], index=["G1", "G2"])
        weights = pd.Series([1e-5, 1e5], index=["G1", "G2"])

        # Second gene dominates
        result = compute_wfc(gene_fc, weights)
        assert result == pytest.approx(2.0, rel=1e-4)
