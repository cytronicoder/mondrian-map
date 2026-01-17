import pandas as pd

from mondrian_map.visualization import create_authentic_mondrian_map


def _build_df():
    return pd.DataFrame(
        {
            "GS_ID": ["WAG000001", "WAG000002", "WAG000003", "WAG000004"],
            "NAME": ["A", "B", "C", "D"],
            "wFC": [1.3, 0.6, 1.0, 2.0],
            "pFDR": [0.01, 0.01, 0.01, 0.1],
            "x": [100, 300, 500, 700],
            "y": [100, 300, 500, 700],
        }
    )


def test_create_authentic_mondrian_map_has_square_layout():
    fig = create_authentic_mondrian_map(_build_df(), dataset_name="Test")
    assert fig.layout.width == fig.layout.height
    assert fig.layout.yaxis.scaleanchor == "x"


def test_tile_traces_have_black_borders():
    fig = create_authentic_mondrian_map(_build_df(), dataset_name="Test")
    tile_traces = [
        trace for trace in fig.data if getattr(trace, "fill", "") == "toself"
    ]
    assert tile_traces
    for trace in tile_traces:
        assert trace.line.color in ["black", "#050103"]


def test_color_mapping_matches_thresholds_and_pFDR_black_rule():
    fig = create_authentic_mondrian_map(_build_df(), dataset_name="Test")
    tile_traces = [
        trace for trace in fig.data if getattr(trace, "fill", "") == "toself"
    ]
    colors = {}
    for trace in tile_traces:
        pid = trace.customdata[0]["pathway_id"]
        colors[pid] = trace.fillcolor
    assert colors["WAG000001"] in ["red", "#E70503"]
    assert colors["WAG000002"] in ["blue", "#0300AD"]
    assert colors["WAG000003"] in ["yellow", "#FDDE06"]
    assert colors["WAG000004"] in ["black", "#050103"]


def test_labels_are_last4_digits():
    fig = create_authentic_mondrian_map(_build_df(), dataset_name="Test")
    texts = {ann["text"] for ann in fig.layout.annotations}
    assert {"0001", "0002", "0003", "0004"}.issubset(texts)


def test_click_targets_have_customdata():
    fig = create_authentic_mondrian_map(_build_df(), dataset_name="Test")
    click_traces = [
        trace
        for trace in fig.data
        if getattr(trace, "mode", "") == "markers" and trace.marker.opacity == 0
    ]
    assert click_traces
    assert click_traces[0].customdata is not None
