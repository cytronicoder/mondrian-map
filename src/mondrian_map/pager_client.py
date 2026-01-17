"""
PAGER Client Module

This module provides a typed, cached interface to the PAGER API
(Pathway, Annotation, Gene Enrichment and Retrieval).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .config import PagerConfig

logger = logging.getLogger(__name__)

PAGER_BASE_URL = "https://discovery.informatics.uab.edu/PAGER"

ENDPOINTS = {
    "gnpa": f"{PAGER_BASE_URL}/index.php/geneset/pagerapi",
    "pag_interaction": f"{PAGER_BASE_URL}/index.php/pag_pag/inter_network_int_api/",
}


class PagerClient:
    """Client for the PAGER API with caching and rate limiting."""

    def __init__(self, config: PagerConfig, session: Optional[requests.Session] = None):
        self.config = config

        if session is None:
            # Use a fresh requests.Session when no custom session is provided.
            self.session = requests.Session()
        else:
            # Validate that the provided session behaves like a requests.Session.
            missing_attrs = [
                name
                for name in ("get", "post")
                if not hasattr(session, name) or not callable(getattr(session, name, None))
            ]
            if missing_attrs:
                raise TypeError(
                    "session must be a requests.Session-like object with callable "
                    f"methods: {', '.join(missing_attrs)}"
                )
            self.session = session
        self._last_request_time = 0.0

        self.cache_dir = Path(self.config.cache_dir) if self.config.cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = (
            self.cache_dir / "requests_manifest.jsonl" if self.cache_dir else None
        )

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit:
            time.sleep(self.config.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _cache_key(self, endpoint: str, params: Dict[str, Any], method: str) -> str:
        payload = json.dumps({"endpoint": endpoint, "params": params, "method": method}, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _cache_paths(self, cache_key: str) -> Dict[str, Path]:
        if not self.cache_dir:
            raise ValueError("cache_dir is not configured")
        return {
            "raw": self.cache_dir / f"{cache_key}.json",
            "parsed": self.cache_dir / f"{cache_key}.parquet",
        }

    def _log_request(self, endpoint: str, params: Dict[str, Any], cache_key: str) -> None:
        if not self._manifest_path:
            return
        record = {
            "timestamp": time.time(),
            "endpoint": endpoint,
            "params": params,
            "cache_key": cache_key,
        }
        with self._manifest_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _request(self, endpoint: str, params: Dict[str, Any], method: str = "POST") -> Any:
        cache_key = self._cache_key(endpoint, params, method)
        if self.config.use_cache and self.cache_dir:
            cache_paths = self._cache_paths(cache_key)
            if cache_paths["raw"].exists():
                self._log_request(endpoint, params, cache_key)
                with cache_paths["raw"].open("r") as f:
                    return json.load(f)

        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()
                if method.upper() == "GET":
                    response = self.session.get(endpoint, params=params, timeout=30)
                else:
                    response = self.session.post(endpoint, data=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if self.config.use_cache and self.cache_dir:
                    cache_paths = self._cache_paths(cache_key)
                    with cache_paths["raw"].open("w") as f:
                        json.dump(data, f)
                self._log_request(endpoint, params, cache_key)
                return data
            except requests.RequestException as exc:
                last_exception = exc
                wait = self.config.retry_delay * (attempt + 1)
                logger.warning(
                    "PAGER request failed (attempt %s/%s): %s",
                    attempt + 1,
                    self.config.max_retries,
                    exc,
                )
                time.sleep(wait)

        raise last_exception

    def run_gnpa(self, gene_symbols: List[str], source: str) -> pd.DataFrame:
        if source.lower() != "wikipathways":
            raise ValueError(
                "PAGER GNPA must use WikiPathways only. "
                "Set pager.source to 'WikiPathways'."
            )
        params = {
            "genes": "%20".join(gene_symbols),
            "source": source,
            "type": self.config.pag_type,
            "ge": self.config.min_size,
            "le": self.config.max_size,
            "sim": str(self.config.similarity),
            "olap": str(self.config.overlap),
            "organism": self.config.organism,
            "cohesion": str(self.config.ncoco),
        }

        cache_key = self._cache_key(ENDPOINTS["gnpa"], params, "POST")
        if self.config.use_cache and self.cache_dir:
            cache_paths = self._cache_paths(cache_key)
            if cache_paths["parsed"].exists():
                return pd.read_parquet(cache_paths["parsed"])

        data = self._request(ENDPOINTS["gnpa"], params)
        if isinstance(data, dict) and "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame(data)

        df = df.rename(
            columns={
                "P": "PVALUE",
                "PVALUE": "PVALUE",
                "FDR": "pFDR",
                "pFDR": "pFDR",
                "GS_ID": "GS_ID",
                "NAME": "NAME",
                "RP_GENES": "RP_GENES",
                "MEMBERSHIP": "MEMBERSHIP",
            }
        )
        if "pFDR" not in df.columns and "FDR" in df.columns:
            df["pFDR"] = df["FDR"]

        if self.config.use_cache and self.cache_dir:
            cache_paths = self._cache_paths(cache_key)
            df.to_parquet(cache_paths["parsed"], index=False)

        return df

    def filter_significant_pags(
        self,
        pag_df: pd.DataFrame,
        pvalue_col: str,
        cutoff: float = 0.05,
    ) -> pd.DataFrame:
        return pag_df[pag_df[pvalue_col] < cutoff].reset_index(drop=True)

    def get_pag_pag_network(
        self,
        pag_ids: List[str],
        network_type: str = "m",
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        if network_type != "m":
            raise ValueError("Only m-type PAG-PAG networks are supported.")
        if source and source.lower() != "wikipathways":
            raise ValueError("PAG-PAG network source must be WikiPathways only.")

        params = {
            "pag": ",".join(pag_ids),
        }

        cache_key = self._cache_key(ENDPOINTS["pag_interaction"], params, "POST")
        if self.config.use_cache and self.cache_dir:
            cache_paths = self._cache_paths(cache_key)
            if cache_paths["parsed"].exists():
                return pd.read_parquet(cache_paths["parsed"])

        data = self._request(ENDPOINTS["pag_interaction"], params)
        if isinstance(data, dict) and "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame(data)

        df = df.rename(
            columns={
                "GS_A_ID": "GS_ID_A",
                "GS_B_ID": "GS_ID_B",
                "WEIGHT": "WEIGHT",
                "TYPE": "TYPE",
                "SOURCE": "SOURCE",
            }
        )
        if "TYPE" not in df.columns:
            df["TYPE"] = "m"
        if "SOURCE" not in df.columns:
            df["SOURCE"] = source or "WikiPathways"

        if self.config.use_cache and self.cache_dir:
            cache_paths = self._cache_paths(cache_key)
            df.to_parquet(cache_paths["parsed"], index=False)

        return df

    def extract_rp_scores(self, pag_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        rp_scores: Dict[str, Dict[str, float]] = {}
        for _, row in pag_df.iterrows():
            pag_id = str(row.get("GS_ID"))
            rp_raw = row.get("RP_GENES")
            if isinstance(rp_raw, dict):
                rp_scores[pag_id] = {k: float(v) for k, v in rp_raw.items()}
                continue
            if not rp_raw:
                rp_scores[pag_id] = {}
                continue
            entries = []
            if isinstance(rp_raw, str):
                if ";" in rp_raw:
                    entries = rp_raw.split(";")
                elif "|" in rp_raw:
                    entries = rp_raw.split("|")
                else:
                    entries = rp_raw.split(",")
            scores: Dict[str, float] = {}
            for entry in entries:
                if ":" not in entry:
                    continue
                gene, score = entry.split(":", 1)
                gene = gene.strip()
                try:
                    scores[gene] = float(score)
                except ValueError:
                    continue
            rp_scores[pag_id] = scores
        return rp_scores
