import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from mondrian_map.pager_client import PagerClient


def make_synthetic_glass_matrices(
    n_genes: int = 200,
    patient_ids: List[str] | None = None,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if patient_ids is None:
        patient_ids = ["0279", "0027", "5965", "F922", "A7RK", "R064"]
    genes = [f"GENE{i:04d}" for i in range(n_genes)]

    base = rng.uniform(0.01, 2.0, size=(n_genes, len(patient_ids)))
    tp = pd.DataFrame(base, index=genes, columns=patient_ids)
    r1 = tp.copy()
    r2 = tp.copy()

    r1.iloc[:10] = tp.iloc[:10] * 2.0
    r2.iloc[10:20] = tp.iloc[10:20] * 0.4

    tp.index.name = "gene"
    r1.index.name = "gene"
    r2.index.name = "gene"

    return tp, r1, r2


class MockPagerClient(PagerClient):
    def __init__(self, config):
        super().__init__(config)

    def run_gnpa(self, gene_symbols, source: str):  # type: ignore[override]
        pag_ids = [f"WAG{1000 + i:06d}" for i in range(12)]
        data = []
        for i, pag_id in enumerate(pag_ids):
            genes = gene_symbols[:5]
            rp_genes = ";".join([f"{g}:{1.0 / (j + 1):.3f}" for j, g in enumerate(genes)])
            data.append(
                {
                    "GS_ID": pag_id,
                    "NAME": f"Pathway {i}",
                    "PVALUE": 0.001 * (i + 1),
                    "pFDR": 0.002 * (i + 1),
                    "RP_GENES": rp_genes,
                    "MEMBERSHIP": ",".join(genes),
                }
            )
        df = pd.DataFrame(data)
        self._write_manifest("mock_gnpa")
        return df

    def filter_significant_pags(self, pag_df, pvalue_col, cutoff=0.05):  # type: ignore[override]
        return pag_df[pag_df[pvalue_col] < cutoff].reset_index(drop=True)

    def get_pag_pag_network(self, pag_ids, network_type="m", source=None):  # type: ignore[override]
        edges = []
        for i in range(min(len(pag_ids) - 1, 5)):
            edges.append(
                {
                    "GS_ID_A": pag_ids[i],
                    "GS_ID_B": pag_ids[i + 1],
                    "TYPE": "m",
                    "SOURCE": source or "WikiPathways",
                }
            )
        df = pd.DataFrame(edges)
        self._write_manifest("mock_network")
        return df

    def _write_manifest(self, endpoint: str) -> None:
        if not self.cache_dir:
            return
        manifest = Path(self.cache_dir) / "requests_manifest.jsonl"
        payload = {"endpoint": endpoint}
        with manifest.open("a") as f:
            f.write(json.dumps(payload) + "\n")
