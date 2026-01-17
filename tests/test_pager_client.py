import pandas as pd
import pytest

from mondrian_map.config import PagerConfig
from mondrian_map.pager_client import PagerClient


def test_pager_client_parses_minimal_mock_response(monkeypatch, tmp_path):
    config = PagerConfig(cache_dir=str(tmp_path), use_cache=False)
    client = PagerClient(config)

    def fake_request(endpoint, params, method="POST"):
        return {
            "data": [
                {
                    "GS_ID": "WAG000001",
                    "NAME": "Test",
                    "P": 0.01,
                    "FDR": 0.02,
                    "RP_GENES": "G1:1.0",
                    "MEMBERSHIP": "G1,G2",
                }
            ]
        }

    monkeypatch.setattr(client, "_request", fake_request)
    df = client.run_gnpa(["G1"], source="WikiPathways")
    assert isinstance(df, pd.DataFrame)
    assert "GS_ID" in df.columns
    assert "pFDR" in df.columns


def test_pager_client_enforces_wikipathways_only_and_m_network_type():
    config = PagerConfig(cache_dir="", use_cache=False)
    client = PagerClient(config)

    with pytest.raises(ValueError):
        client.run_gnpa(["G1"], source="Reactome")

    with pytest.raises(ValueError):
        client.get_pag_pag_network(["WAG000001"], network_type="r")
