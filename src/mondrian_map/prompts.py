"""
Prompt generation utilities for pathway embedding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def build_pathway_prompts(
    pag_df: pd.DataFrame,
    prompt_type: str,
    rp_gene_order: Optional[Dict[str, List[str]]] = None,
    gene_descriptions: Optional[Dict[str, str]] = None,
    pathway_summaries: Optional[Dict[str, str]] = None,
    max_genes: int = 100,
) -> List[str]:
    """Build prompts for pathway embedding."""
    prompts: List[str] = []
    gene_descriptions = gene_descriptions or {}
    pathway_summaries = pathway_summaries or {}

    for _, row in pag_df.iterrows():
        pag_id = str(row.get("GS_ID"))
        if prompt_type == "gene_symbols":
            genes = rp_gene_order.get(pag_id) if rp_gene_order else None
            if genes is None:
                membership = str(row.get("MEMBERSHIP", "")).split(",")
                genes = [g.strip() for g in membership if g.strip()]
            prompt = " ".join(genes[:max_genes])
        elif prompt_type == "gene_descriptions":
            genes = rp_gene_order.get(pag_id) if rp_gene_order else None
            if genes is None:
                membership = str(row.get("MEMBERSHIP", "")).split(",")
                genes = [g.strip() for g in membership if g.strip()]
            entries = [
                f"{gene} ({gene_descriptions.get(gene, '')})"
                for gene in genes[:max_genes]
            ]
            prompt = ", ".join(entries)
        elif prompt_type == "pathway_name":
            prompt = str(row.get("NAME", pag_id))
        elif prompt_type == "pathway_description_summary":
            prompt = pathway_summaries.get(pag_id) or str(row.get("DESCRIPTION", ""))
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        prompts.append(prompt)

    return prompts


def summarize_pathway_descriptions(
    pag_df: pd.DataFrame,
    model: str,
    max_words: int = 300,
    cache_path: str = "cache/pathway_summaries.json",
    force: bool = False,
) -> Dict[str, str]:
    """
    Summarize pathway descriptions to <= max_words with caching.

    If model is "fallback", use a deterministic truncation and emit a warning.
    If model is "none" or empty and no cache exists, raise an error.
    """
    cache_file = Path(cache_path)
    if cache_file.exists() and not force:
        with cache_file.open("r") as f:
            return json.load(f)

    if not model or model == "none":
        raise ValueError(
            "No summarizer configured and no cached summaries available. "
            "Provide a summarizer or enable fallback summarization."
        )

    summaries: Dict[str, str] = {}
    if model == "fallback":
        logger.warning("Using fallback summarization (truncation) for pathway prompts.")
        for _, row in pag_df.iterrows():
            pag_id = str(row.get("GS_ID"))
            description = str(row.get("DESCRIPTION", ""))
            summaries[pag_id] = _truncate_words(description, max_words)
    else:
        raise ValueError(
            "External summarization is not configured in this environment. "
            "Use model='fallback' or provide cached summaries."
        )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w") as f:
        json.dump(summaries, f, indent=2)

    return summaries
