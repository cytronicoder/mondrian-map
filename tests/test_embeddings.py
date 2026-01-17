import numpy as np

from mondrian_map.config import EmbeddingConfig
from mondrian_map.embeddings import EmbeddingGenerator, embed_pathways


def test_embed_pathways_llm2vec_does_not_overwrite_embeddings(monkeypatch):
    fake_vectors = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=float)

    class FakeModel:
        def encode(self, texts):
            return fake_vectors

    def fake_ensure(self):
        self._model = FakeModel()

    monkeypatch.setattr(EmbeddingGenerator, "_ensure_model_loaded", fake_ensure)

    config = EmbeddingConfig(model_type="llm2vec", normalize=True, cache_dir="")
    embeddings = embed_pathways(["P1", "P2"], ["a", "b"], config)

    assert embeddings.shape == (2, 2)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0)
