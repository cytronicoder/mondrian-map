"""
PAGER Client Module

This module provides a typed, cached interface to the PAGER API
(Pathway, Annotation, Gene Enrichment and Retrieval).

API Documentation: https://discovery.informatics.uab.edu/PAGER/
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# PAGER API base URL
PAGER_BASE_URL = "https://discovery.informatics.uab.edu/PAGER"

# Default API endpoints
ENDPOINTS = {
    "gnpa": f"{PAGER_BASE_URL}/index.php/geneset/pagerapi",
    "members": f"{PAGER_BASE_URL}/index.php/geneset/get_members_by_ids/",
    "pag_interaction": f"{PAGER_BASE_URL}/index.php/pag_pag/inter_network_int_api/",
    "pag_regulation": f"{PAGER_BASE_URL}/index.php/pag_pag/inter_network_reg_api/",
    "ranked_genes": f"{PAGER_BASE_URL}/index.php/genesinPAG/viewgenes/",
    "gene_interaction": f"{PAGER_BASE_URL}/index.php/pag_mol_mol_map/interactions/",
    "gene_regulation": f"{PAGER_BASE_URL}/index.php/pag_mol_mol_map/regulations/",
    "ngsea": f"{PAGER_BASE_URL}/index.php/geneset/ngseaapi/",
}

# Default sources for WikiPathways
DEFAULT_SOURCES = ["WikiPathway_2021"]

# Rate limiting settings
DEFAULT_RATE_LIMIT = 1.0  # seconds between requests
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5.0  # seconds


@dataclass
class PagerConfig:
    """Configuration for PAGER API client."""

    # API settings
    base_url: str = PAGER_BASE_URL
    timeout: int = 60

    # Rate limiting
    rate_limit: float = DEFAULT_RATE_LIMIT
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY

    # Caching
    cache_dir: Optional[Path] = None
    use_cache: bool = True

    # Default query parameters
    default_sources: List[str] = field(default_factory=lambda: DEFAULT_SOURCES.copy())
    default_pag_type: str = "P"  # P=Pathway, A=Annotation, G=Gene set
    default_min_size: int = 1
    default_max_size: int = 2000
    default_similarity: float = 0.05
    default_overlap: int = 1
    default_ncoco: float = 0
    default_pvalue: float = 0.05
    default_fdr: float = 0.5
    default_organism: str = "All"


class PagerClient:
    """
    Client for the PAGER API with caching and rate limiting.

    Example:
        >>> client = PagerClient(cache_dir="cache/pager")
        >>> # Get enriched pathways for a gene list
        >>> pags = client.get_significant_pags(["BRCA1", "TP53", "EGFR"])
        >>> # Get ranked genes for a pathway
        >>> genes = client.get_pag_ranked_genes("WAG003294")
    """

    def __init__(self, config: Optional[PagerConfig] = None):
        """
        Initialize PAGER client.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or PagerConfig()
        self._last_request_time = 0.0

        # Setup cache directory
        if self.config.cache_dir:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit:
            time.sleep(self.config.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key from endpoint and parameters."""
        param_str = json.dumps(params, sort_keys=True)
        hash_input = f"{endpoint}:{param_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Optional[Path]:
        """Get cache file path for a given key."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load data from cache if available."""
        if not self.config.use_cache:
            return None

        cache_path = self._get_cache_path(cache_key)
        if cache_path is None or not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                logger.debug(f"Cache hit: {cache_key}")
                return data
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save data to cache."""
        if not self.config.use_cache:
            return

        cache_path = self._get_cache_path(cache_key)
        if cache_path is None:
            return

        try:
            with open(cache_path, "w") as f:
                json.dump(data, f)
            logger.debug(f"Cached: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "POST",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Make a request to the PAGER API with caching and retry logic.

        Args:
            endpoint: API endpoint URL
            params: Request parameters
            method: HTTP method (GET or POST)
            use_cache: Whether to use caching for this request

        Returns:
            JSON response data

        Raises:
            requests.RequestException: If request fails after retries
        """
        params = params or {}

        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(endpoint, params)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        # Make request with retry
        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()

                if method.upper() == "GET":
                    response = requests.get(
                        endpoint, params=params, timeout=self.config.timeout
                    )
                else:
                    response = requests.post(
                        endpoint, data=params, timeout=self.config.timeout
                    )

                response.raise_for_status()
                data = response.json()

                # Cache successful response
                if use_cache:
                    self._save_to_cache(cache_key, data)

                return data

            except requests.RequestException as e:
                last_exception = e
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        raise last_exception

    def run_gnpa(
        self,
        genes: List[str],
        source: Optional[List[str]] = None,
        pag_type: str = "P",
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        similarity: Optional[float] = None,
        overlap: Optional[int] = None,
        organism: Optional[str] = None,
        ncoco: Optional[float] = None,
        pvalue: Optional[float] = None,
        fdr: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Run GNPA (Gene-set Network Pathway Analysis) to find enriched pathways.

        Args:
            genes: List of gene symbols
            source: Data sources (default: WikiPathway_2021)
            pag_type: PAG type - "P" (Pathway), "A" (Annotation), "G" (Gene set), or "All"
            min_size: Minimum PAG size
            max_size: Maximum PAG size
            similarity: Minimum similarity score (0-1)
            overlap: Minimum overlap count
            organism: Organism filter
            ncoco: Minimum nCoCo score
            pvalue: p-value threshold
            fdr: FDR threshold

        Returns:
            DataFrame with enriched PAGs

        Example:
            >>> pags = client.run_gnpa(["BRCA1", "TP53"], source=["WikiPathway_2021"])
        """
        # Apply defaults
        source = source or self.config.default_sources
        min_size = min_size if min_size is not None else self.config.default_min_size
        max_size = max_size if max_size is not None else self.config.default_max_size
        similarity = (
            similarity if similarity is not None else self.config.default_similarity
        )
        overlap = overlap if overlap is not None else self.config.default_overlap
        organism = organism or self.config.default_organism
        ncoco = ncoco if ncoco is not None else self.config.default_ncoco
        pvalue = pvalue if pvalue is not None else self.config.default_pvalue
        fdr = fdr if fdr is not None else self.config.default_fdr

        # Build params with PAGER API encoding
        params = {
            "genes": "%20".join(genes),
            "source": "%20".join(source),
            "type": pag_type,
            "ge": min_size,
            "le": max_size,
            "sim": str(similarity),
            "olap": str(overlap),
            "organism": organism,
            "cohesion": str(ncoco),
            "pvalue": pvalue,
            "FDR": fdr,
        }

        logger.info(f"Running GNPA with {len(genes)} genes, source={source}")

        data = self._request(ENDPOINTS["gnpa"], params)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame(data)

        logger.info(f"GNPA returned {len(df)} enriched PAGs")
        return df

    def get_significant_pags(
        self,
        genes: List[str],
        source: str = "WikiPathway_2021",
        pval_thresh: float = 0.05,
        fdr_thresh: float = 0.5,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Get significantly enriched PAGs (convenience wrapper for run_gnpa).

        Args:
            genes: List of gene symbols
            source: Data source (e.g., "WikiPathway_2021")
            pval_thresh: p-value threshold
            fdr_thresh: FDR threshold
            **kwargs: Additional arguments passed to run_gnpa

        Returns:
            DataFrame with significant PAGs
        """
        return self.run_gnpa(
            genes=genes,
            source=[source] if isinstance(source, str) else source,
            pvalue=pval_thresh,
            fdr=fdr_thresh,
            **kwargs,
        )

    def get_pag_members(self, pag_ids: List[str]) -> pd.DataFrame:
        """
        Get gene members for a list of PAGs.

        Args:
            pag_ids: List of PAG IDs (e.g., ["WAG003294", "WAG002659"])

        Returns:
            DataFrame with PAG membership information
        """
        if not pag_ids:
            return pd.DataFrame()

        params = {"pag": ",".join(pag_ids)}

        logger.info(f"Getting members for {len(pag_ids)} PAGs")
        data = self._request(ENDPOINTS["members"], params)

        if isinstance(data, dict) and "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame(data)

        logger.info(f"Retrieved membership for {len(df)} gene-PAG pairs")
        return df

    def get_pag_ranked_genes(self, pag_id: str) -> pd.DataFrame:
        """
        Get RP-ranked genes for a specific PAG with RP scores.

        Args:
            pag_id: PAG identifier (e.g., "WAG003294")

        Returns:
            DataFrame with columns: GENE_SYM, RP_SCORE, DESCRIPTION, etc.
        """
        endpoint = f"{ENDPOINTS['ranked_genes']}{pag_id}"

        logger.debug(f"Getting ranked genes for {pag_id}")
        data = self._request(endpoint, method="GET")

        if isinstance(data, dict) and "gene" in data:
            df = pd.DataFrame(data["gene"])
        else:
            df = pd.DataFrame(data)

        if "RP_SCORE" in df.columns:
            df["RP_SCORE"] = pd.to_numeric(df["RP_SCORE"], errors="coerce")

        return df

    def get_pag_pag_network(
        self,
        pag_ids: List[str],
        network_type: str = "m-type",
    ) -> pd.DataFrame:
        """
        Get PAG-PAG relationships (network edges).

        Args:
            pag_ids: List of PAG IDs
            network_type: "m-type" for molecular interactions, "r-type" for regulatory

        Returns:
            DataFrame with PAG relationship edges
        """
        if not pag_ids:
            return pd.DataFrame()

        params = {"pag": ",".join(pag_ids)}

        if network_type == "m-type":
            endpoint = ENDPOINTS["pag_interaction"]
        elif network_type == "r-type":
            endpoint = ENDPOINTS["pag_regulation"]
        else:
            raise ValueError(f"Unknown network_type: {network_type}")

        logger.info(f"Getting {network_type} network for {len(pag_ids)} PAGs")
        data = self._request(endpoint, params)

        if isinstance(data, dict) and "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame(data)

        logger.info(f"Retrieved {len(df)} PAG-PAG relationships")
        return df

    def get_gene_interactions(self, pag_id: str) -> pd.DataFrame:
        """
        Get gene-gene interaction network for a PAG.

        Args:
            pag_id: PAG identifier

        Returns:
            DataFrame with gene interaction edges
        """
        endpoint = f"{ENDPOINTS['gene_interaction']}{pag_id}"
        data = self._request(endpoint, method="GET")

        if isinstance(data, dict) and "data" in data:
            return pd.DataFrame(data["data"])
        return pd.DataFrame(data)

    def get_gene_regulations(self, pag_id: str) -> pd.DataFrame:
        """
        Get gene-gene regulatory network for a PAG.

        Args:
            pag_id: PAG identifier

        Returns:
            DataFrame with gene regulatory edges
        """
        endpoint = f"{ENDPOINTS['gene_regulation']}{pag_id}"
        data = self._request(endpoint, method="GET")

        if isinstance(data, dict) and "data" in data:
            return pd.DataFrame(data["data"])
        return pd.DataFrame(data)

    def clear_cache(self) -> int:
        """
        Clear all cached responses.

        Returns:
            Number of cache files deleted
        """
        if self.cache_dir is None:
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        logger.info(f"Cleared {count} cached responses")
        return count


# Convenience function for backward compatibility with notebooks
def create_pager_client(
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
) -> PagerClient:
    """
    Create a configured PAGER client.

    Args:
        cache_dir: Directory for caching API responses
        use_cache: Whether to enable caching

    Returns:
        Configured PagerClient instance
    """
    config = PagerConfig(
        cache_dir=Path(cache_dir) if cache_dir else None,
        use_cache=use_cache,
    )
    return PagerClient(config)
