import numpy as np
import pandas as pd

from mondrian_map.config import PipelineConfig
from mondrian_map.embeddings import EmbeddingGenerator
from mondrian_map.pipeline import run_case_study
from tests.utils_synth import MockPagerClient, make_synthetic_glass_matrices


def test_run_case_study_synthetic_end_to_end(tmp_path, monkeypatch):
    tp, r1, r2 = make_synthetic_glass_matrices()
    tp_path = tmp_path / "tp.csv"
    r1_path = tmp_path / "r1.csv"
    r2_path = tmp_path / "r2.csv"
    tp.to_csv(tp_path, index=True)
    r1.to_csv(r1_path, index=True)
    r2.to_csv(r2_path, index=True)

    def fake_embed_texts(self, texts, **kwargs):
        return np.tile(np.array([1.0, 2.0]), (len(texts), 1))

    monkeypatch.setattr(EmbeddingGenerator, "embed_texts", fake_embed_texts)
    monkeypatch.setattr("mondrian_map.pager_client.PagerClient", MockPagerClient)
    monkeypatch.setattr(
        "mondrian_map.projection.project_tsne", lambda X, **kwargs: X[:, :2]
    )

    cfg = PipelineConfig()
    cfg.embedding.prompt_type = "pathway_name"
    cfg.embedding.model_type = "sentence_transformer"
    cfg.embedding.normalize = False
    cfg.tsne.perplexity = 5.0
    cfg.tsne.learning_rate = 100.0
    cfg.tsne.n_iter = 250

    out_dir = tmp_path / "outputs"
    run_case_study(cfg, str(tp_path), str(r1_path), str(r2_path), str(out_dir))

    profiles = ["aggressive", "baseline", "non_aggressive"]
    contrasts = ["R1_vs_TP", "R2_vs_TP"]
    for profile in profiles:
        for contrast in contrasts:
            attrs = out_dir / "profiles" / profile / contrast / "attributes.csv"
            rels = out_dir / "profiles" / profile / contrast / "relations.csv"
            assert attrs.exists()
            assert rels.exists()
            df = pd.read_csv(attrs)
            assert len(df) <= 10

    assert (out_dir / "figures" / "mondrian_panel_3x2.html").exists()
    assert (out_dir / "run_metadata.json").exists()
    assert (out_dir / "cache" / "pager" / "requests_manifest.jsonl").exists()
