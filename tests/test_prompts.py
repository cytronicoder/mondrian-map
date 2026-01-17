import pytest
import pandas as pd

from mondrian_map.prompts import build_pathway_prompts, summarize_pathway_descriptions


def test_build_pathway_prompts_gene_symbols_truncates_to_max_genes_and_is_deterministic():
    pag_df = pd.DataFrame(
        {
            "GS_ID": ["P1"],
            "MEMBERSHIP": ["G1,G2,G3,G4"],
        }
    )
    rp_gene_order = {"P1": ["G4", "G3", "G2", "G1"]}
    prompts = build_pathway_prompts(
        pag_df,
        prompt_type="gene_symbols",
        rp_gene_order=rp_gene_order,
        max_genes=2,
    )
    assert prompts == ["G4 G3"]


def test_summarize_pathway_descriptions_requires_precomputed_or_explicit_fallback(tmp_path):
    pag_df = pd.DataFrame(
        {
            "GS_ID": ["P1"],
            "DESCRIPTION": ["A pathway description."],
        }
    )
    with pytest.raises(ValueError):
        summarize_pathway_descriptions(
            pag_df,
            model="none",
            cache_path=str(tmp_path / "missing.json"),
        )
