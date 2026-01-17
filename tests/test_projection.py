import numpy as np

from mondrian_map.data_processing import normalize_coords_to_canvas


def test_normalize_coords_to_canvas_bounds_and_padding():
    coords = np.array([[0.0, 0.0], [10.0, 10.0]])
    normalized = normalize_coords_to_canvas(
        coords, canvas_min=0.0, canvas_max=100.0, pad=10.0
    )
    assert normalized.min() >= 10.0
    assert normalized.max() <= 90.0
