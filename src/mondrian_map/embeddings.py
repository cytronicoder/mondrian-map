"""
Embeddings Module for Mondrian Map

This module handles pathway embedding generation using various language models
including SentenceTransformers and optionally LLM2Vec.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from .pager_client import PagerClient

# Default model configurations
DEFAULT_SENTENCE_TRANSFORMER = "all-mpnet-base-v2"
DEFAULT_MINILM = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_GENES = 100

# Prompt instructions for LLM2Vec
GENE_SYMBOL_INSTRUCTION = (
    "Given a list of gene symbols, encode the collective biological "
    "significance and pathway associations of these genes:"
)

GENE_DESCRIPTION_INSTRUCTION = (
    "Given a list of gene symbols and its description, encode the collective "
    "biological significance and pathway associations of these genes:"
)

PATHWAY_NAME_INSTRUCTION = (
    "Given a pathway name, encode the biological significance and "
    "functional associations of this pathway:"
)

PATHWAY_DESCRIPTION_INSTRUCTION = (
    "Given a pathway name and its description, encode the biological "
    "significance and functional associations of this pathway:"
)


@dataclass
class EmbeddingConfig:
    """Configuration parameters for neural embedding generation.

    Attributes
    ----------
    model_name : str
        Identifier of the pre-trained embedding model.
    model_type : {'sentence_transformer', 'llm2vec'}
        Embedding model architecture.
    batch_size : int
        Number of samples processed in parallel during encoding.
    normalize : bool
        Whether to apply L2 normalization to embeddings.
    max_length : int
        Maximum token sequence length for model input.
    prompt_type : str
        Template for constructing pathway descriptions.
    max_genes : int
        Maximum genes included in pathway prompts.
    device : str, optional
        Compute device ('cpu', 'cuda', or None for auto-detection).
    """


class EmbeddingGenerator:
    """Neural pathway embedding generator supporting multiple model architectures.

    Encodes textual pathway descriptions to fixed-dimensional vector embeddings
    suitable for dimensionality reduction and visualization.

    Supports:
    - SentenceTransformers: Lightweight, pre-trained encoders for semantic similarity
    - LLM2Vec: Large language model-based encoders with instruction-tuning

    Parameters
    ----------
    config : EmbeddingConfig, optional
        Embedding configuration. Uses defaults if not provided.

    Example
    -------
    >>> from src.mondrian_map.embeddings import EmbeddingGenerator
    >>> gen = EmbeddingGenerator()
    >>> embeddings = gen.embed_texts(['Apoptosis', 'Cell cycle'])
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding generator.

        Args:
            config: Embedding configuration. Uses defaults if not provided.
        """
        self.config = config or EmbeddingConfig()
        self._model = None
        self._tokenizer = None

    def _load_sentence_transformer(self):
        """Load SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Loading SentenceTransformer: {self.config.model_name}")
        self._model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )

    def _load_llm2vec(self):
        """Load LLM2Vec model."""
        try:
            import torch
            from llm2vec import LLM2Vec
            from peft import PeftModel
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"LLM2Vec dependencies not installed: {e}. "
                "Install with: pip install llm2vec peft transformers torch"
            )

        model_name = self.config.llm2vec_model
        postfix = self.config.llm2vec_postfix

        logger.info(f"Loading LLM2Vec model: {model_name}")

        # Determine device
        if self.config.device:
            device = self.config.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        # Load MNTP model
        model = PeftModel.from_pretrained(model, model_name)
        model = model.merge_and_unload()

        # Load supervised LoRA weights
        model = PeftModel.from_pretrained(model, f"{model_name}-{postfix}")

        # Create LLM2Vec wrapper
        self._model = LLM2Vec(
            model,
            tokenizer,
            pooling_mode=self.config.pooling_mode,
            max_length=self.config.max_length,
        )
        self._tokenizer = tokenizer

        logger.info(f"LLM2Vec model loaded on {device}")

    def _ensure_model_loaded(self):
        """Ensure model is loaded before use."""
        if self._model is None:
            if self.config.model_type == "llm2vec":
                self._load_llm2vec()
            else:
                self._load_sentence_transformer()

    def embed_texts(
        self,
        texts: List[str],
        normalize: Optional[bool] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            normalize: Whether to L2-normalize embeddings (default from config)
            batch_size: Batch size for encoding (default from config)
            show_progress: Whether to show progress bar

        Returns:
            NumPy array of embeddings with shape (n_texts, embedding_dim)
        """
        self._ensure_model_loaded()

        normalize = normalize if normalize is not None else self.config.normalize
        batch_size = batch_size or self.config.batch_size

        logger.info(f"Embedding {len(texts)} texts with batch_size={batch_size}")

        if self.config.model_type == "sentence_transformer":
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
            )
        else:
            # LLM2Vec
            embeddings = self._model.encode(texts)
            if isinstance(embeddings, np.ndarray) is False:
                embeddings = embeddings.numpy()
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-8)

        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def embed_with_instruction(
        self,
        texts: List[str],
        instruction: str,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Generate embeddings with an instruction prefix (for LLM2Vec).

        Args:
            texts: List of text strings
            instruction: Instruction prefix for the model
            normalize: Whether to normalize embeddings

        Returns:
            NumPy array of embeddings
        """
        if self.config.model_type != "llm2vec":
            logger.warning(
                "embed_with_instruction is designed for LLM2Vec. "
                "Using standard embedding for SentenceTransformer."
            )
            return self.embed_texts(texts, normalize=normalize)

        self._ensure_model_loaded()

        # Prepare instruction-text pairs
        instruction_texts = [[instruction, text] for text in texts]

        embeddings = self._model.encode(instruction_texts)
        if isinstance(embeddings, np.ndarray) is False:
            embeddings = embeddings.numpy()

        if normalize or (normalize is None and self.config.normalize):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings


def build_gene_symbol_prompts(
    pathway_ids: List[str],
    pager_client: "PagerClient",
    max_genes: int = DEFAULT_MAX_GENES,
    sort_by_rp: bool = True,
) -> Dict[str, str]:
    """
    Build gene symbol prompts for pathway embedding.

    Prompt format: "GENE1 GENE2 GENE3 ..."

    Args:
        pathway_ids: List of pathway IDs
        pager_client: PagerClient for fetching gene data
        max_genes: Maximum number of genes to include
        sort_by_rp: Whether to sort genes by RP score (descending)

    Returns:
        Dictionary mapping pathway ID to prompt string
    """
    prompts = {}

    for pag_id in pathway_ids:
        ranked_genes = pager_client.get_pag_ranked_genes(pag_id)

        if len(ranked_genes) == 0:
            prompts[pag_id] = ""
            continue

        if sort_by_rp and "RP_SCORE" in ranked_genes.columns:
            ranked_genes["RP_SCORE"] = pd.to_numeric(
                ranked_genes["RP_SCORE"], errors="coerce"
            ).fillna(0)
            ranked_genes = ranked_genes.sort_values("RP_SCORE", ascending=False)

        genes = ranked_genes["GENE_SYM"].head(max_genes).tolist()
        prompts[pag_id] = " ".join(genes)

    logger.info(f"Built gene symbol prompts for {len(prompts)} pathways")
    return prompts


def build_gene_description_prompts(
    pathway_ids: List[str],
    pager_client: "PagerClient",
    max_genes: int = DEFAULT_MAX_GENES,
    sort_by_rp: bool = True,
) -> Dict[str, str]:
    """
    Build gene symbol + description prompts for pathway embedding.

    Prompt format: "GENE1 (description), GENE2 (description), ..."

    Args:
        pathway_ids: List of pathway IDs
        pager_client: PagerClient for fetching gene data
        max_genes: Maximum number of genes to include
        sort_by_rp: Whether to sort genes by RP score

    Returns:
        Dictionary mapping pathway ID to prompt string
    """
    prompts = {}

    for pag_id in pathway_ids:
        ranked_genes = pager_client.get_pag_ranked_genes(pag_id)

        if len(ranked_genes) == 0:
            prompts[pag_id] = ""
            continue

        if sort_by_rp and "RP_SCORE" in ranked_genes.columns:
            ranked_genes["RP_SCORE"] = pd.to_numeric(
                ranked_genes["RP_SCORE"], errors="coerce"
            ).fillna(0)
            ranked_genes = ranked_genes.sort_values("RP_SCORE", ascending=False)

        ranked_genes = ranked_genes.head(max_genes)

        if "DESCRIPTION" in ranked_genes.columns:
            entries = [
                f"{row['GENE_SYM']} ({row.get('DESCRIPTION', '')})"
                for _, row in ranked_genes.iterrows()
            ]
        else:
            entries = ranked_genes["GENE_SYM"].tolist()

        prompts[pag_id] = ", ".join(entries)

    logger.info(f"Built gene description prompts for {len(prompts)} pathways")
    return prompts


def build_pathway_name_prompts(
    pathway_ids: List[str],
    pathway_info: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """
    Build pathway name prompts for embedding.

    Args:
        pathway_ids: List of pathway IDs
        pathway_info: Dictionary mapping pathway ID to metadata

    Returns:
        Dictionary mapping pathway ID to pathway name
    """
    prompts = {}

    for pag_id in pathway_ids:
        info = pathway_info.get(pag_id, {})
        name = info.get("NAME", pag_id)
        prompts[pag_id] = name

    logger.info(f"Built pathway name prompts for {len(prompts)} pathways")
    return prompts


def build_pathway_description_prompts(
    pathway_ids: List[str],
    pathway_info: Dict[str, Dict[str, Any]],
    use_summary: bool = True,
) -> Dict[str, str]:
    """
    Build pathway description/summary prompts for embedding.

    Args:
        pathway_ids: List of pathway IDs
        pathway_info: Dictionary mapping pathway ID to metadata
        use_summary: If True, use LLM-generated summary; else use raw description

    Returns:
        Dictionary mapping pathway ID to description text
    """
    prompts = {}

    for pag_id in pathway_ids:
        info = pathway_info.get(pag_id, {})

        if use_summary and "Summary" in info:
            text = info["Summary"]
        elif "Description" in info:
            text = info["Description"]
        else:
            text = info.get("NAME", pag_id)

        prompts[pag_id] = text

    logger.info(f"Built pathway description prompts for {len(prompts)} pathways")
    return prompts


def build_llm2vec_prompts(
    prompts: Dict[str, str],
    instruction: str,
) -> Dict[str, List[str]]:
    """
    Convert prompts to LLM2Vec format with instruction.

    Args:
        prompts: Dictionary mapping ID to prompt text
        instruction: Instruction string for the model

    Returns:
        Dictionary mapping ID to [instruction, text] pairs
    """
    return {pag_id: [instruction, text] for pag_id, text in prompts.items()}


def build_prompts(
    pathway_ids: List[str],
    prompt_type: str,
    pathway_info: Optional[Dict[str, Dict[str, Any]]] = None,
    pager_client: Optional["PagerClient"] = None,
    max_genes: int = DEFAULT_MAX_GENES,
    for_llm2vec: bool = False,
) -> Dict[str, Any]:
    """
    Build prompts for pathway embedding based on prompt type.

    Args:
        pathway_ids: List of pathway IDs
        prompt_type: Type of prompt to build:
            - "gene_symbol": Gene symbols only
            - "gene_description": Gene symbols with descriptions
            - "pathway_name": Pathway names
            - "pathway_description_summary": Pathway description summaries
        pathway_info: Pathway metadata dictionary
        pager_client: PagerClient for gene data (required for gene prompts)
        max_genes: Maximum genes for gene-based prompts
        for_llm2vec: Whether to format for LLM2Vec (with instruction)

    Returns:
        Dictionary of prompts (str or [instruction, text] for LLM2Vec)
    """
    if prompt_type in ["gene_symbol", "gene_description"]:
        if pager_client is None:
            raise ValueError(f"pager_client required for prompt_type='{prompt_type}'")

    if prompt_type in ["pathway_name", "pathway_description_summary"]:
        if pathway_info is None:
            raise ValueError(f"pathway_info required for prompt_type='{prompt_type}'")

    # Build base prompts
    if prompt_type == "gene_symbol":
        prompts = build_gene_symbol_prompts(pathway_ids, pager_client, max_genes)
        instruction = GENE_SYMBOL_INSTRUCTION
    elif prompt_type == "gene_description":
        prompts = build_gene_description_prompts(pathway_ids, pager_client, max_genes)
        instruction = GENE_DESCRIPTION_INSTRUCTION
    elif prompt_type == "pathway_name":
        prompts = build_pathway_name_prompts(pathway_ids, pathway_info)
        instruction = PATHWAY_NAME_INSTRUCTION
    elif prompt_type == "pathway_description_summary":
        prompts = build_pathway_description_prompts(
            pathway_ids, pathway_info, use_summary=True
        )
        instruction = PATHWAY_DESCRIPTION_INSTRUCTION
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    # Convert to LLM2Vec format if needed
    if for_llm2vec:
        prompts = build_llm2vec_prompts(prompts, instruction)

    return prompts


def embed_pathways(
    pathway_ids: List[str],
    prompt_type: str,
    pathway_info: Optional[Dict[str, Dict[str, Any]]] = None,
    pager_client: Optional["PagerClient"] = None,
    config: Optional[EmbeddingConfig] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate embeddings for pathways using specified prompt type.

    Args:
        pathway_ids: List of pathway IDs
        prompt_type: Type of prompt (see build_prompts)
        pathway_info: Pathway metadata
        pager_client: PagerClient for gene data
        config: Embedding configuration

    Returns:
        Tuple of (embeddings array, ordered pathway IDs)
    """
    config = config or EmbeddingConfig(prompt_type=prompt_type)

    # Build prompts
    is_llm2vec = config.model_type == "llm2vec"
    prompts = build_prompts(
        pathway_ids,
        prompt_type,
        pathway_info=pathway_info,
        pager_client=pager_client,
        max_genes=config.max_genes,
        for_llm2vec=is_llm2vec,
    )

    # Filter out empty prompts
    valid_ids = [pid for pid, p in prompts.items() if p]
    texts = [prompts[pid] for pid in valid_ids]

    if len(texts) == 0:
        raise ValueError("No valid prompts generated")

    # Generate embeddings
    generator = EmbeddingGenerator(config)

    if is_llm2vec:
        embeddings = generator._ensure_model_loaded()
        # For LLM2Vec, texts are [instruction, text] pairs
        embeddings = generator._model.encode(texts)
        if not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.numpy()
        if config.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
    else:
        embeddings = generator.embed_texts(texts)

    return embeddings, valid_ids


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings.

    Args:
        embeddings: Array of embeddings (n_samples, n_features)

    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)
