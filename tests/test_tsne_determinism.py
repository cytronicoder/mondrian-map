"""
Tests for t-SNE determinism and reproducibility.

These tests verify that t-SNE projection produces identical results
when given the same random seed.
"""

import numpy as np
import pytest

from mondrian_map.projection import TSNEConfig, tsne_project, verify_determinism


class TestTSNEDeterminism:
    """Test suite for t-SNE reproducibility."""

    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        np.random.seed(123)  # Different from t-SNE seed
        return np.random.randn(50, 384)  # 50 samples, 384-dim embeddings

    def test_same_seed_same_result(self, sample_embeddings):
        """Verify that same seed produces identical results."""
        config = TSNEConfig(random_state=42, n_iter=250)

        result1 = tsne_project(sample_embeddings, config)
        result2 = tsne_project(sample_embeddings, config)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_different_seed_different_result(self, sample_embeddings):
        """Verify that different seeds produce different results."""
        config1 = TSNEConfig(random_state=42, n_iter=250)
        config2 = TSNEConfig(random_state=99, n_iter=250)

        result1 = tsne_project(sample_embeddings, config1)
        result2 = tsne_project(sample_embeddings, config2)

        # Results should be different
        assert not np.allclose(result1, result2)

    def test_verify_determinism_helper(self, sample_embeddings):
        """Test the verify_determinism helper function."""
        config = TSNEConfig(random_state=42, n_iter=250)

        is_deterministic, diff = verify_determinism(sample_embeddings, config)

        assert is_deterministic is True
        assert diff == 0.0

    def test_output_shape(self, sample_embeddings):
        """Verify output has correct shape."""
        config = TSNEConfig(n_components=2, random_state=42, n_iter=250)

        result = tsne_project(sample_embeddings, config)

        assert result.shape == (50, 2)

    def test_3d_projection(self, sample_embeddings):
        """Test 3D t-SNE projection."""
        config = TSNEConfig(n_components=3, random_state=42, n_iter=250)

        result = tsne_project(sample_embeddings, config)

        assert result.shape == (50, 3)

    def test_perplexity_effect(self, sample_embeddings):
        """Verify different perplexity values produce different layouts."""
        config1 = TSNEConfig(perplexity=5, random_state=42, n_iter=250)
        config2 = TSNEConfig(perplexity=30, random_state=42, n_iter=250)

        result1 = tsne_project(sample_embeddings, config1)
        result2 = tsne_project(sample_embeddings, config2)

        # Different perplexity should give different results
        assert not np.allclose(result1, result2)


class TestTSNEConfig:
    """Test TSNEConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TSNEConfig()

        assert config.n_components == 2
        assert config.perplexity == 30
        assert config.random_state == 42
        assert config.n_iter == 1000
        assert config.metric == "cosine"

    def test_custom_values(self):
        """Test custom configuration."""
        config = TSNEConfig(
            n_components=3,
            perplexity=50,
            random_state=123,
            n_iter=500,
        )

        assert config.n_components == 3
        assert config.perplexity == 50
        assert config.random_state == 123
        assert config.n_iter == 500


class TestTSNEWithRealEmbeddings:
    """Tests using realistic embedding dimensions."""

    def test_mpnet_dimensions(self):
        """Test with all-mpnet-base-v2 embedding dimensions (768)."""
        np.random.seed(456)
        embeddings = np.random.randn(30, 768)

        config = TSNEConfig(random_state=42, n_iter=250)
        result = tsne_project(embeddings, config)

        assert result.shape == (30, 2)
        # Values should be finite
        assert np.all(np.isfinite(result))

    def test_minilm_dimensions(self):
        """Test with all-MiniLM-L6-v2 embedding dimensions (384)."""
        np.random.seed(789)
        embeddings = np.random.randn(30, 384)

        config = TSNEConfig(random_state=42, n_iter=250)
        result = tsne_project(embeddings, config)

        assert result.shape == (30, 2)
        assert np.all(np.isfinite(result))


class TestTSNEEdgeCases:
    """Test edge cases for t-SNE projection."""

    def test_small_sample_size(self):
        """Test t-SNE with minimal sample dataset."""
        np.random.seed(111)
        embeddings = np.random.randn(5, 100)

        config = TSNEConfig(perplexity=2, random_state=42, n_iter=250)
        result = tsne_project(embeddings, config)

        assert result.shape == (5, 2)

    def test_high_dimensional_input(self):
        """Test with high-dimensional embeddings."""
        np.random.seed(222)
        embeddings = np.random.randn(20, 4096)

        config = TSNEConfig(random_state=42, n_iter=250)
        result = tsne_project(embeddings, config)

        assert result.shape == (20, 2)

    def test_identical_embeddings(self):
        """Test behavior with identical embeddings."""
        # All same embedding
        embeddings = np.ones((10, 100))

        config = TSNEConfig(random_state=42, n_iter=250)
        result = tsne_project(embeddings, config)

        # Should still produce output (though may be degenerate)
        assert result.shape == (10, 2)
