"""
Visualization Module for Mondrian Maps

This module contains the visualization functions including the complex
line generation algorithms and Plotly figure creation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .core import (LINE_WIDTH, Block, Colors, Corner, CornerPos, GridSystem,
                   Line, LineDir, Point, adjust, adjust_d, blank_canvas,
                   euclidean_distance_point, get_line_direction)
from .data_processing import get_relations, prepare_pathway_data


def get_closest_corner(block_a: Block, block_b: Block) -> Corner:
    """Identify the nearest corner of one block to another block's centroid.

    Uses Manhattan distance to locate the corner of block_a that minimizes
    the distance to block_b's center point, enabling optimal line routing.

    Args:
        block_a: Source pathway block
        block_b: Target pathway block for connection

    Returns:
        Corner instance of block_a closest to block_b's center
    """
    min_distance = float("inf")
    closest_corner = None

    for corner in [
        block_a.top_left,
        block_a.top_right,
        block_a.bottom_right,
        block_a.bottom_left,
    ]:
        distance = abs(corner.point.x - block_b.center.x) + abs(
            corner.point.y - block_b.center.y
        )
        if distance < min_distance:
            min_distance = distance
            closest_corner = corner

    return closest_corner


def get_furthest_connector(cp1: Corner, cp2: Corner, center: Point) -> Point:
    """Select the optimal connector point for pathways with maximum separation.

    Evaluates two candidate connector points (L-shaped junctions) and selects
    the one that maximizes distance from the center, reducing visual clutter.

    Args:
        cp1: First corner of potential connection
        cp2: Second corner of potential connection
        center: Central reference point for layout

    Returns:
        Point instance representing the selected connector location
    """
    p = Point(cp1.point.x, cp2.point.y)
    q = Point(cp2.point.x, cp1.point.y)
    if euclidean_distance_point(
        (p.x, p.y), (center.x, center.y)
    ) > euclidean_distance_point((q.x, q.y), (center.x, center.y)):
        return p
    else:
        return q


def get_manhattan_line_color(block_a: Block, block_b: Block) -> Colors:
    """Determine connection line color based on block pair regulation state.

    Assigns consistent coloring to pathway relationship edges reflecting the
    combined regulation state: red for co-upregulated, blue for co-downregulated,
    yellow for mixed or neutral interactions.

    Args:
        block_a: First pathway block
        block_b: Second pathway block

    Returns:
        Colors enum value for the inter-block connection line
    """
    if block_a.color == Colors.RED and block_b.color == Colors.RED:
        return Colors.RED
    elif block_a.color == Colors.BLUE and block_b.color == Colors.BLUE:
        return Colors.BLUE
    else:
        return Colors.YELLOW


def get_manhattan_lines_2(
    corner_a: Corner, corner_b: Corner, connector: Point, color: Colors
) -> List[Line]:
    """Generate orthogonal (Manhattan distance) connection lines between pathway blocks.

    Constructs a two-segment path connecting corners using axis-aligned routing,
    with adaptive line placement based on corner positions to minimize overlap
    and maintain visual clarity in dense pathway relationship networks.

    Args:
        corner_a: Starting corner of the connection
        corner_b: Ending corner of the connection
        connector: Intermediate junction point for complex pathways
        color: Color assignment for the connection line

    Returns:
        List of Line objects representing the pathway relationship visualization
    """
    if corner_a.point.x == corner_b.point.x and corner_a.point.y != corner_b.point.y:
        if corner_a.position in [CornerPos.TOP_LEFT, CornerPos.BOTTOM_LEFT]:
            line_v = Line(
                Point(corner_a.point.x + adjust, corner_a.point.y),
                Point(corner_b.point.x + adjust, corner_b.point.y),
                get_line_direction(corner_a.point, corner_b.point),
                color=color,
            )
        if corner_a.position in [CornerPos.TOP_RIGHT, CornerPos.BOTTOM_RIGHT]:
            line_v = Line(
                Point(corner_a.point.x - adjust, corner_a.point.y),
                Point(corner_b.point.x - adjust, corner_b.point.y),
                get_line_direction(corner_a.point, corner_b.point),
                color=color,
            )
        corner_a.line = line_v
        corner_b.line = line_v
        return [line_v]

    elif corner_a.point.x != corner_b.point.x and corner_a.point.y == corner_b.point.y:
        if corner_a.position == CornerPos.TOP_LEFT:
            line_h = Line(
                Point(corner_a.point.x - adjust_d, corner_a.point.y + adjust),
                Point(corner_b.point.x, corner_b.point.y + adjust),
                get_line_direction(corner_a.point, corner_b.point),
                color=color,
            )
        elif corner_a.position == CornerPos.TOP_RIGHT:
            line_h = Line(
                Point(corner_a.point.x + adjust_d, corner_a.point.y + adjust),
                Point(corner_b.point.x, corner_b.point.y + adjust),
                get_line_direction(corner_a.point, corner_b.point),
                color=color,
            )
        elif corner_a.position == CornerPos.BOTTOM_LEFT:
            line_h = Line(
                Point(corner_a.point.x - adjust_d, corner_a.point.y - adjust),
                Point(corner_b.point.x, corner_b.point.y - adjust),
                get_line_direction(corner_a.point, corner_b.point),
                color=color,
            )
        elif corner_a.position == CornerPos.BOTTOM_RIGHT:
            line_h = Line(
                Point(corner_a.point.x + adjust_d, corner_a.point.y - adjust),
                Point(corner_b.point.x, corner_b.point.y - adjust),
                get_line_direction(corner_a.point, corner_b.point),
                color=color,
            )
        corner_a.line = line_h
        corner_b.line = line_h
        return [line_h]

    # Complex case with connector point
    line_a_dir = get_line_direction(corner_a.point, connector)
    line_b_dir = get_line_direction(connector, corner_b.point)

    # Create line A
    if corner_a.position in [
        CornerPos.TOP_LEFT,
        CornerPos.TOP_RIGHT,
    ] and line_a_dir in [LineDir.LEFT, LineDir.RIGHT]:
        if corner_a.position == CornerPos.TOP_RIGHT and line_a_dir == LineDir.RIGHT:
            line_a = Line(
                Point(corner_a.point.x + adjust_d, corner_a.point.y + adjust),
                Point(connector.x, connector.y + adjust),
                line_a_dir,
                color=color,
            )
        elif corner_a.position == CornerPos.TOP_LEFT and line_a_dir == LineDir.LEFT:
            line_a = Line(
                Point(corner_a.point.x - adjust_d, corner_a.point.y + adjust),
                Point(connector.x, connector.y + adjust),
                line_a_dir,
                color=color,
            )
        else:
            line_a = Line(
                Point(corner_a.point.x, corner_a.point.y + adjust),
                Point(connector.x, connector.y + adjust),
                line_a_dir,
                color=color,
            )
    elif corner_a.position in [
        CornerPos.BOTTOM_LEFT,
        CornerPos.BOTTOM_RIGHT,
    ] and line_a_dir in [LineDir.LEFT, LineDir.RIGHT]:
        if corner_a.position == CornerPos.BOTTOM_RIGHT and line_a_dir == LineDir.RIGHT:
            line_a = Line(
                Point(corner_a.point.x + adjust_d, corner_a.point.y - adjust),
                Point(connector.x, connector.y - adjust),
                line_a_dir,
                color=color,
            )
        elif corner_a.position == CornerPos.BOTTOM_LEFT and line_a_dir == LineDir.LEFT:
            line_a = Line(
                Point(corner_a.point.x - adjust_d, corner_a.point.y - adjust),
                Point(connector.x, connector.y - adjust),
                line_a_dir,
                color=color,
            )
        else:
            line_a = Line(
                Point(corner_a.point.x, corner_a.point.y - adjust),
                Point(connector.x, connector.y - adjust),
                line_a_dir,
                color=color,
            )
    elif corner_a.position in [
        CornerPos.TOP_LEFT,
        CornerPos.BOTTOM_LEFT,
    ] and line_a_dir in [LineDir.UP, LineDir.DOWN]:
        if corner_a.position == CornerPos.TOP_LEFT and line_a_dir == LineDir.UP:
            line_a = Line(
                Point(corner_a.point.x + adjust, corner_a.point.y - adjust_d),
                Point(connector.x + adjust, connector.y),
                line_a_dir,
                color=color,
            )
        elif corner_a.position == CornerPos.BOTTOM_LEFT and line_a_dir == LineDir.DOWN:
            line_a = Line(
                Point(corner_a.point.x + adjust, corner_a.point.y + adjust_d),
                Point(connector.x + adjust, connector.y),
                line_a_dir,
                color=color,
            )
        else:
            line_a = Line(
                Point(corner_a.point.x + adjust, corner_a.point.y),
                Point(connector.x + adjust, connector.y),
                line_a_dir,
                color=color,
            )
    elif corner_a.position in [
        CornerPos.TOP_RIGHT,
        CornerPos.BOTTOM_RIGHT,
    ] and line_a_dir in [LineDir.UP, LineDir.DOWN]:
        if corner_a.position == CornerPos.TOP_RIGHT and line_a_dir == LineDir.UP:
            line_a = Line(
                Point(corner_a.point.x - adjust, corner_a.point.y - adjust_d),
                Point(connector.x - adjust, connector.y),
                line_a_dir,
                color=color,
            )
        elif corner_a.position == CornerPos.BOTTOM_RIGHT and line_a_dir == LineDir.DOWN:
            line_a = Line(
                Point(corner_a.point.x - adjust, corner_a.point.y + adjust_d),
                Point(connector.x - adjust, connector.y),
                line_a_dir,
                color=color,
            )
        else:
            line_a = Line(
                Point(corner_a.point.x - adjust, corner_a.point.y),
                Point(connector.x - adjust, connector.y),
                line_a_dir,
                color=color,
            )
    else:
        line_a = Line(corner_a.point, connector, line_a_dir, color=color)

    # Create line B
    if corner_b.position in [
        CornerPos.TOP_LEFT,
        CornerPos.TOP_RIGHT,
    ] and line_b_dir in [LineDir.LEFT, LineDir.RIGHT]:
        if corner_b.position == CornerPos.TOP_LEFT and line_b_dir == LineDir.RIGHT:
            line_b = Line(
                Point(connector.x - adjust * 2, connector.y + adjust),
                Point(corner_b.point.x - adjust_d, corner_b.point.y + adjust),
                line_b_dir,
                color=color,
            )
        elif corner_b.position == CornerPos.TOP_RIGHT and line_b_dir == LineDir.LEFT:
            line_b = Line(
                Point(connector.x + adjust * 2, connector.y + adjust),
                Point(corner_b.point.x + adjust_d, corner_b.point.y + adjust),
                line_b_dir,
                color=color,
            )
        else:
            line_b = Line(
                Point(connector.x, connector.y + adjust),
                Point(corner_b.point.x, corner_b.point.y + adjust),
                line_b_dir,
                color=color,
            )
    elif corner_b.position in [
        CornerPos.BOTTOM_LEFT,
        CornerPos.BOTTOM_RIGHT,
    ] and line_b_dir in [LineDir.LEFT, LineDir.RIGHT]:
        if corner_b.position == CornerPos.BOTTOM_LEFT and line_b_dir == LineDir.RIGHT:
            line_b = Line(
                Point(connector.x - adjust * 2, connector.y - adjust),
                Point(corner_b.point.x - adjust_d, corner_b.point.y - adjust),
                line_b_dir,
                color=color,
            )
        elif corner_b.position == CornerPos.BOTTOM_RIGHT and line_b_dir == LineDir.LEFT:
            line_b = Line(
                Point(connector.x + adjust * 2, connector.y - adjust),
                Point(corner_b.point.x + adjust_d, corner_b.point.y - adjust),
                line_b_dir,
                color=color,
            )
        else:
            line_b = Line(
                Point(connector.x, connector.y - adjust),
                Point(corner_b.point.x, corner_b.point.y - adjust),
                line_b_dir,
                color=color,
            )
    elif corner_b.position in [
        CornerPos.TOP_LEFT,
        CornerPos.BOTTOM_LEFT,
    ] and line_b_dir in [LineDir.UP, LineDir.DOWN]:
        if corner_b.position == CornerPos.TOP_LEFT and line_b_dir == LineDir.DOWN:
            line_b = Line(
                Point(connector.x + adjust, connector.y - adjust * 2),
                Point(corner_b.point.x + adjust, corner_b.point.y - adjust_d),
                line_b_dir,
                color=color,
            )
        elif corner_b.position == CornerPos.BOTTOM_LEFT and line_b_dir == LineDir.UP:
            line_b = Line(
                Point(connector.x + adjust, connector.y + adjust * 2),
                Point(corner_b.point.x + adjust, corner_b.point.y + adjust_d),
                line_b_dir,
                color=color,
            )
        else:
            line_b = Line(
                Point(connector.x + adjust, connector.y),
                Point(corner_b.point.x + adjust, corner_b.point.y),
                line_b_dir,
                color=color,
            )
    elif corner_b.position in [
        CornerPos.TOP_RIGHT,
        CornerPos.BOTTOM_RIGHT,
    ] and line_b_dir in [LineDir.UP, LineDir.DOWN]:
        if corner_b.position == CornerPos.TOP_RIGHT and line_b_dir == LineDir.DOWN:
            line_b = Line(
                Point(connector.x - adjust, connector.y - adjust * 2),
                Point(corner_b.point.x - adjust, corner_b.point.y - adjust_d),
                line_b_dir,
                color=color,
            )
        elif corner_b.position == CornerPos.BOTTOM_RIGHT and line_b_dir == LineDir.UP:
            line_b = Line(
                Point(connector.x - adjust, connector.y + adjust * 2),
                Point(corner_b.point.x - adjust, corner_b.point.y + adjust_d),
                line_b_dir,
                color=color,
            )
        else:
            line_b = Line(
                Point(connector.x - adjust, connector.y),
                Point(corner_b.point.x - adjust, corner_b.point.y),
                line_b_dir,
                color=color,
            )
    else:
        line_b = Line(connector, corner_b.point, line_b_dir, color=color)

    corner_a.line = line_a
    corner_b.line = line_b
    return [line_a, line_b]


def create_smart_grid_lines(grid_system: GridSystem, blocks: List[Block]) -> List[Line]:
    """
    Legacy smart grid line generation.

    NOTE: This is retained for backward compatibility but is no longer used by the
    paper-style visualization, which instead renders sparse partition lines via shapes.
    """
    return []


def _quantize_positions(
    values: List[float], step: float, epsilon: float = 1.0
) -> List[float]:
    """Quantize and merge near-duplicate positions to stabilize sparse partitions."""
    quantized = [round(value / step) * step for value in values]
    quantized.sort()
    merged = []
    for value in quantized:
        if not merged or abs(value - merged[-1]) > epsilon:
            merged.append(value)
    return merged


def create_partition_line_shapes(
    blocks: List[Block],
    grid_step: float,
    max_lines: int,
    color: str,
    width: int,
    canvas_size: int = 1000,
) -> List[Dict]:
    """
    Create sparse, structural partition lines from tile edges.

    The filtering strategy keeps only coordinates that appear on multiple tile edges
    and then down-samples to the top-N most frequent positions to avoid a dense lattice.
    """
    edge_x = []
    edge_y = []
    for block in blocks:
        edge_x.extend([block.top_left_p[0], block.bottom_right_p[0]])
        edge_y.extend([block.top_left_p[1], block.bottom_right_p[1]])

    quantized_x = _quantize_positions(edge_x, grid_step)
    quantized_y = _quantize_positions(edge_y, grid_step)

    freq_x: Dict[float, int] = {}
    freq_y: Dict[float, int] = {}
    for value in edge_x:
        q = round(value / grid_step) * grid_step
        freq_x[q] = freq_x.get(q, 0) + 1
    for value in edge_y:
        q = round(value / grid_step) * grid_step
        freq_y[q] = freq_y.get(q, 0) + 1

    candidates_x = [x for x in quantized_x if freq_x.get(x, 0) >= 2]
    candidates_y = [y for y in quantized_y if freq_y.get(y, 0) >= 2]

    ranked_x = sorted(candidates_x, key=lambda x: freq_x.get(x, 0), reverse=True)
    ranked_y = sorted(candidates_y, key=lambda y: freq_y.get(y, 0), reverse=True)

    total_candidates = [("x", value) for value in ranked_x] + [
        ("y", value) for value in ranked_y
    ]
    if max_lines > 0:
        total_candidates = total_candidates[:max_lines]

    shapes = []
    for axis, value in total_candidates:
        if value <= 0 or value >= canvas_size:
            continue
        if axis == "x":
            shapes.append(
                dict(
                    type="line",
                    x0=value,
                    x1=value,
                    y0=0,
                    y1=canvas_size,
                    line=dict(color=color, width=width),
                    layer="below",
                )
            )
        else:
            shapes.append(
                dict(
                    type="line",
                    x0=0,
                    x1=canvas_size,
                    y0=value,
                    y1=value,
                    line=dict(color=color, width=width),
                    layer="below",
                )
            )

    return shapes


def _normalize_color_name(color_value: str) -> str:
    """Normalize color strings/hex values to red/blue/yellow/black labels."""
    normalized = str(color_value).lower()
    # Map hex values to color names
    color_map = {
        Colors.RED.value.lower(): "red",
        Colors.BLUE.value.lower(): "blue",
        Colors.YELLOW.value.lower(): "yellow",
        Colors.BLACK.value.lower(): "black",
    }
    return color_map.get(normalized, normalized)


def _get_block_bounds(block: Block) -> Dict[str, float]:
    return {
        "left": block.top_left_p[0],
        "right": block.bottom_right_p[0],
        "top": block.top_left_p[1],
        "bottom": block.bottom_right_p[1],
    }


def _get_block_center(block: Block) -> Tuple[float, float]:
    return (
        (block.top_left_p[0] + block.bottom_right_p[0]) / 2,
        (block.top_left_p[1] + block.bottom_right_p[1]) / 2,
    )


def _block_ports(
    block: Block, target_center: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """Return candidate ports on block edges ordered by preferred routing direction."""
    left = block.top_left_p[0]
    right = block.bottom_right_p[0]
    top = block.top_left_p[1]
    bottom = block.bottom_right_p[1]
    cx, cy = _get_block_center(block)
    tx, ty = target_center

    ports = {
        "left": (max(0, left - 3), cy),
        "right": (min(1000, right + 3), cy),
        "top": (cx, max(0, top - 3)),
        "bottom": (cx, min(1000, bottom + 3)),
    }

    dx = tx - cx
    dy = ty - cy
    if abs(dx) >= abs(dy):
        primary = "right" if dx >= 0 else "left"
        secondary = ["top", "bottom"]
        # Determine the opposite horizontal direction
        tertiary = "left" if primary == "right" else "right"
        ordered_keys = [primary] + secondary + [tertiary]
    else:
        primary = "bottom" if dy >= 0 else "top"
        secondary = ["left", "right"]
        # Determine the opposite vertical direction
        tertiary = "top" if primary == "bottom" else "bottom"
        ordered_keys = [primary] + secondary + [tertiary]

    return [ports[key] for key in ordered_keys]


def _segment_intersects_rect(
    a: Tuple[float, float],
    b: Tuple[float, float],
    rect: Dict[str, float],
    epsilon: float = 1e-6,
) -> bool:
    """Check if an axis-aligned segment crosses a rectangle interior."""
    left = rect["left"] + epsilon
    right = rect["right"] - epsilon
    top = rect["top"] + epsilon
    bottom = rect["bottom"] - epsilon

    if a[0] == b[0]:  # vertical
        x = a[0]
        if not (left < x < right):
            return False
        seg_top = min(a[1], b[1])
        seg_bottom = max(a[1], b[1])
        return seg_bottom > top and seg_top < bottom
    if a[1] == b[1]:  # horizontal
        y = a[1]
        if not (top < y < bottom):
            return False
        seg_left = min(a[0], b[0])
        seg_right = max(a[0], b[0])
        return seg_right > left and seg_left < right
    return True


def _path_hits_obstacles(
    path: List[Tuple[float, float]],
    obstacles: List[Dict[str, float]],
    ignore_rects: List[Dict[str, float]],
) -> bool:
    """Return True if any segment intersects obstacle interiors (excluding endpoints)."""
    # Convert ignore_rects to set of tuples for proper comparison
    ignore_set = {
        (rect["left"], rect["right"], rect["top"], rect["bottom"])
        for rect in ignore_rects
    }

    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        for rect in obstacles:
            # Check if this rect should be ignored using content comparison
            rect_tuple = (rect["left"], rect["right"], rect["top"], rect["bottom"])
            if rect_tuple in ignore_set:
                continue
            if _segment_intersects_rect(a, b, rect):
                return True
    return False


def _simplify_path(
    path: List[Tuple[float, float]], epsilon: float = 1e-6
) -> List[Tuple[float, float]]:
    """Remove duplicate consecutive points within epsilon threshold."""
    simplified = []
    for point in path:
        if not simplified or (
            abs(point[0] - simplified[-1][0]) > epsilon
            or abs(point[1] - simplified[-1][1]) > epsilon
        ):
            simplified.append(point)
    return simplified


def route_manhattan(
    src: Block,
    dst: Block,
    obstacles: List[Dict[str, float]],
    grid_step: float,
    canvas_size: int = 1000,
) -> List[Tuple[float, float]]:
    """Route orthogonal segments while avoiding tile interiors via offset bend search."""
    src_center = _get_block_center(src)
    dst_center = _get_block_center(dst)
    src_ports = _block_ports(src, dst_center)
    dst_ports = _block_ports(dst, src_center)

    src_rect = _get_block_bounds(src)
    dst_rect = _get_block_bounds(dst)
    ignore_rects = [src_rect, dst_rect]

    offsets = [0, grid_step, -grid_step, 2 * grid_step, -2 * grid_step]

    for src_port in src_ports:
        for dst_port in dst_ports:
            sx, sy = src_port
            dx, dy = dst_port

            for offset in offsets:
                # Candidate A: horizontal then vertical (offset y to avoid obstacles).
                mid_y = sy + offset
                if 0 <= mid_y <= canvas_size:
                    candidate = (
                        [(sx, sy), (dx, sy), (dx, dy)]
                        if offset == 0
                        else [(sx, sy), (sx, mid_y), (dx, mid_y), (dx, dy)]
                    )
                    candidate = _simplify_path(candidate)
                    if not _path_hits_obstacles(candidate, obstacles, ignore_rects):
                        return candidate

                # Candidate B: vertical then horizontal (offset x to avoid obstacles).
                mid_x = sx + offset
                if 0 <= mid_x <= canvas_size:
                    candidate = (
                        [(sx, sy), (sx, dy), (dx, dy)]
                        if offset == 0
                        else [(sx, sy), (mid_x, sy), (mid_x, dy), (dx, dy)]
                    )
                    candidate = _simplify_path(candidate)
                    if not _path_hits_obstacles(candidate, obstacles, ignore_rects):
                        return candidate

    return []


def add_paper_style_legend(fig: go.Figure, no_relations_color: str) -> None:
    """Add right-side legend entries matching the paper-style figure."""
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=12, color=Colors.RED.value, symbol="square"),
            name="Up-regulated pathways (p < 0.05)",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=12, color=Colors.BLUE.value, symbol="square"),
            name="Down-regulated pathways (p < 0.05)",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=12, color=Colors.YELLOW.value, symbol="square"),
            name="Neutrally perturbed pathways (p < 0.05)",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None, None],
            y=[None, None],
            mode="lines",
            line=dict(color=Colors.RED.value, width=3),
            name="crosstalk linking two up-regulated pathways",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None, None],
            y=[None, None],
            mode="lines",
            line=dict(color=Colors.BLUE.value, width=3),
            name="crosstalk linking two down-regulated pathways",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None, None],
            y=[None, None],
            mode="lines",
            line=dict(color=Colors.YELLOW.value, width=3),
            name="crosstalk linking different perturbed pathways",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None, None],
            y=[None, None],
            mode="lines",
            line=dict(color=no_relations_color, width=3, dash="solid"),
            name="No Relations",
            showlegend=True,
        )
    )


def get_meaningful_tile_edges(blocks: List[Block]) -> Dict:
    """Get tile edges that are structurally important"""
    edges = {"vertical": set(), "horizontal": set(), "tile_bounds": {}}

    for block in blocks:
        # Store tile boundaries for intersection checking
        edges["tile_bounds"][block.id] = {
            "left": block.top_left_p[0],
            "right": block.bottom_right_p[0],
            "top": block.top_left_p[1],
            "bottom": block.bottom_right_p[1],
        }

        # Add significant edges (not all edges, only structurally important ones)
        edges["vertical"].add(block.top_left_p[0])  # left edge
        edges["vertical"].add(block.bottom_right_p[0])  # right edge
        edges["horizontal"].add(block.top_left_p[1])  # top edge
        edges["horizontal"].add(block.bottom_right_p[1])  # bottom edge

    return edges


def create_meaningful_vertical_lines(
    grid_system: GridSystem, blocks: List[Block], tile_edges: Dict
) -> List[Line]:
    """Create vertical lines that serve structural purpose"""
    lines = []

    for x_pos in tile_edges["vertical"]:
        if x_pos <= 0 or x_pos >= 1000:  # Skip canvas boundaries
            continue

        # Find meaningful segments for this vertical line
        segments = find_structural_vertical_segments(x_pos, blocks, tile_edges)

        for start_y, end_y in segments:
            if end_y - start_y > 40:  # Only keep substantial segments
                lines.append(
                    Line(
                        Point(x_pos, start_y),
                        Point(x_pos, end_y),
                        LineDir.DOWN if end_y > start_y else LineDir.UP,
                        Colors.LIGHT_GRAY,
                        1,
                    )
                )

    return lines


def create_meaningful_horizontal_lines(
    grid_system: GridSystem, blocks: List[Block], tile_edges: Dict
) -> List[Line]:
    """Create horizontal lines that serve structural purpose"""
    lines = []

    for y_pos in tile_edges["horizontal"]:
        if y_pos <= 0 or y_pos >= 1000:  # Skip canvas boundaries
            continue

        # Find meaningful segments for this horizontal line
        segments = find_structural_horizontal_segments(y_pos, blocks, tile_edges)

        for start_x, end_x in segments:
            if end_x - start_x > 40:  # Only keep substantial segments
                lines.append(
                    Line(
                        Point(start_x, y_pos),
                        Point(end_x, y_pos),
                        LineDir.RIGHT if end_x > start_x else LineDir.LEFT,
                        Colors.LIGHT_GRAY,
                        1,
                    )
                )

    return lines


def find_structural_vertical_segments(
    x_pos: float, blocks: List[Block], tile_edges: Dict
) -> List[Tuple[float, float]]:
    """Find vertical line segments that serve structural purpose"""
    segments = []

    # Find tiles that this vertical line would intersect or touch
    intersecting_tiles = []
    touching_tiles = []

    for block in blocks:
        bounds = tile_edges["tile_bounds"][block.id]

        # Check if line intersects tile interior
        if bounds["left"] < x_pos < bounds["right"]:
            intersecting_tiles.append((bounds["top"], bounds["bottom"]))
        # Check if line touches tile edge
        elif bounds["left"] == x_pos or bounds["right"] == x_pos:
            touching_tiles.append((bounds["top"], bounds["bottom"]))

    # Only create segments where line serves structural purpose
    if not touching_tiles and not intersecting_tiles:
        return []  # No structural purpose

    # Sort all occupied ranges
    all_occupied = sorted(intersecting_tiles + touching_tiles)

    # Merge overlapping ranges
    merged = []
    for start, end in all_occupied:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Create segments only in meaningful gaps
    if not merged:
        return []

    # Only create segments that connect to tile edges or canvas
    current_y = 0
    for start, end in merged:
        # Segment before tile (only if it connects to something meaningful)
        if current_y < start and (
            current_y == 0
            or has_structural_purpose_vertical(x_pos, current_y, start, blocks)
        ):
            segments.append((current_y, start))
        current_y = max(current_y, end)

    # Final segment to canvas edge (only if meaningful)
    if current_y < 1000 and has_structural_purpose_vertical(
        x_pos, current_y, 1000, blocks
    ):
        segments.append((current_y, 1000))

    return segments


def find_structural_horizontal_segments(
    y_pos: float, blocks: List[Block], tile_edges: Dict
) -> List[Tuple[float, float]]:
    """Find horizontal line segments that serve structural purpose"""
    segments = []

    # Find tiles that this horizontal line would intersect or touch
    intersecting_tiles = []
    touching_tiles = []

    for block in blocks:
        bounds = tile_edges["tile_bounds"][block.id]

        # Check if line intersects tile interior
        if bounds["top"] < y_pos < bounds["bottom"]:
            intersecting_tiles.append((bounds["left"], bounds["right"]))
        # Check if line touches tile edge
        elif bounds["top"] == y_pos or bounds["bottom"] == y_pos:
            touching_tiles.append((bounds["left"], bounds["right"]))

    # Only create segments where line serves structural purpose
    if not touching_tiles and not intersecting_tiles:
        return []  # No structural purpose

    # Sort all occupied ranges
    all_occupied = sorted(intersecting_tiles + touching_tiles)

    # Merge overlapping ranges
    merged = []
    for start, end in all_occupied:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Create segments only in meaningful gaps
    if not merged:
        return []

    # Only create segments that connect to tile edges or canvas
    current_x = 0
    for start, end in merged:
        # Segment before tile (only if it connects to something meaningful)
        if current_x < start and (
            current_x == 0
            or has_structural_purpose_horizontal(y_pos, current_x, start, blocks)
        ):
            segments.append((current_x, start))
        current_x = max(current_x, end)

    # Final segment to canvas edge (only if meaningful)
    if current_x < 1000 and has_structural_purpose_horizontal(
        y_pos, current_x, 1000, blocks
    ):
        segments.append((current_x, 1000))

    return segments


def has_structural_purpose_vertical(
    x_pos: float, start_y: float, end_y: float, blocks: List[Block]
) -> bool:
    """Check if a vertical line segment serves structural purpose"""
    # Check if segment endpoints align with tile edges
    start_touches_tile = any(
        (block.top_left_p[1] == start_y or block.bottom_right_p[1] == start_y)
        and (block.top_left_p[0] <= x_pos <= block.bottom_right_p[0])
        for block in blocks
    )

    end_touches_tile = any(
        (block.top_left_p[1] == end_y or block.bottom_right_p[1] == end_y)
        and (block.top_left_p[0] <= x_pos <= block.bottom_right_p[0])
        for block in blocks
    )

    # Has purpose if it connects tile edges or extends from tile to canvas
    return start_touches_tile or end_touches_tile or start_y == 0 or end_y == 1000


def has_structural_purpose_horizontal(
    y_pos: float, start_x: float, end_x: float, blocks: List[Block]
) -> bool:
    """Check if a horizontal line segment serves structural purpose"""
    # Check if segment endpoints align with tile edges
    start_touches_tile = any(
        (block.top_left_p[0] == start_x or block.bottom_right_p[0] == start_x)
        and (block.top_left_p[1] <= y_pos <= block.bottom_right_p[1])
        for block in blocks
    )

    end_touches_tile = any(
        (block.top_left_p[0] == end_x or block.bottom_right_p[0] == end_x)
        and (block.top_left_p[1] <= y_pos <= block.bottom_right_p[1])
        for block in blocks
    )

    # Has purpose if it connects tile edges or extends from tile to canvas
    return start_touches_tile or end_touches_tile or start_x == 0 or end_x == 1000


def create_authentic_mondrian_map(
    df: pd.DataFrame,
    dataset_name: str,
    maximize: bool = False,
    show_pathway_ids: bool = True,
    show_partitions: bool = True,
    edge_top_k: int = 2,
    edge_max_total: int = 30,
    seed: int = 0,
) -> go.Figure:
    """
    Create authentic Mondrian map using the exact algorithm from the notebooks
    """
    if len(df) == 0:
        return go.Figure()

    partition_max_lines = 20
    partition_color = "#D0D0D0"
    partition_width = 2
    show_insignificant_edges = False
    no_relations_color = "#CDB4DB"
    show_legend = False
    tile_area_scale = 0.97
    label_min_side = 30.0

    np.random.seed(seed)

    relations_df = df.attrs.get("relations_df")

    # Prepare data using the data processing module
    network_dir = Path("data/case_study/pathway_networks")
    data = prepare_pathway_data(df, dataset_name, network_dir)
    if isinstance(relations_df, pd.DataFrame):
        data["relations"] = get_relations(relations_df)
        data["network_data"] = relations_df

    center_points = data["center_points"]
    areas = [area * tile_area_scale for area in data["areas"]]
    colors = data["colors"]
    pathway_ids = data["pathway_ids"]
    relations = data["relations"]
    network_data = data.get("network_data")
    suffix_to_row: Dict[str, pd.Series] = {}
    # Build suffix-to-row mapping for pathway data lookup
    # Note: Duplicate GS_IDs (same ID appearing multiple times) will use the last occurrence
    # This only detects suffix collisions (different GS_IDs with same 4-char suffix)
    for _, row in df.iterrows():
        gs_id = str(row["GS_ID"])
        suffix = gs_id[-4:]
        existing_row = suffix_to_row.get(suffix)
        if existing_row is not None:
            existing_gs_id = str(existing_row["GS_ID"])
            if existing_gs_id != gs_id:
                raise ValueError(
                    f"Detected GS_ID suffix collision for suffix '{suffix}': "
                    f"{existing_gs_id} and {gs_id}. "
                    "GS_ID suffixes must be unique; please use full GS_IDs or "
                    "disambiguate the identifiers."
                )
        suffix_to_row[suffix] = row

    # Initialize canvas
    blank_canvas()
    grid_system = GridSystem(1001, 1001, 20, 20)

    # Sort data by area (largest first)
    sorted_data = sorted(zip(areas, center_points, colors, pathway_ids), reverse=True)
    areas_sorted, center_points_sorted, colors_sorted, pathway_ids_sorted = zip(
        *sorted_data
    )

    # Get rectangles from grid system
    rectangles = grid_system.plot_points_fill_blocks(
        center_points_sorted,
        areas_sorted,
        avoid_overlap=True,
        padding=LINE_WIDTH,
        snap_to_grid=True,
        nudge=True,
    )

    # STAGE 1: Create blocks
    all_blocks = []
    for idx, rect in enumerate(rectangles):
        b = Block(
            rect[0],
            rect[1],
            areas_sorted[idx],
            colors_sorted[idx],
            pathway_ids_sorted[idx],
        )
        all_blocks.append(b)

    # STAGE 2: Create relationship lines (Manhattan lines for PAG-to-PAG crosstalk)
    all_manhattan_paths: List[Tuple[List[Tuple[float, float]], str]] = []
    edge_strengths: Dict[Tuple[str, str], float] = {}
    if isinstance(network_data, pd.DataFrame) and len(network_data) > 0:
        col_a = "GS_A_ID" if "GS_A_ID" in network_data.columns else "GS_ID_A"
        col_b = "GS_B_ID" if "GS_B_ID" in network_data.columns else "GS_ID_B"
        for _, row in network_data.iterrows():
            gs_a = str(row[col_a])[-4:]
            gs_b = str(row[col_b])[-4:]
            key = tuple(sorted((gs_a, gs_b)))
            if "PVALUE" in row:
                # Use -log(p) so smaller p-values (more significant) get higher scores
                pval = float(row["PVALUE"])
                score = -np.log(max(pval, 1e-300))  # Avoid log(0)
            elif "SIMILARITY" in row:
                score = float(row["SIMILARITY"])
            else:
                score = 0.0
            if key not in edge_strengths or score > edge_strengths[key]:
                edge_strengths[key] = score

    obstacles = [_get_block_bounds(block) for block in all_blocks]

    sorted_edges = []
    for rel in relations:
        if rel[0] not in Block.instances or rel[1] not in Block.instances:
            continue
        key = tuple(sorted(rel))
        sorted_edges.append((rel[0], rel[1], edge_strengths.get(key, 0.0)))

    sorted_edges.sort(key=lambda item: item[2], reverse=True)

    edge_counts: Dict[str, int] = {}
    selected_edges = []
    for source_id, target_id, score in sorted_edges:
        if edge_counts.get(source_id, 0) >= edge_top_k:
            continue
        if edge_counts.get(target_id, 0) >= edge_top_k:
            continue
        selected_edges.append((source_id, target_id, score))
        edge_counts[source_id] = edge_counts.get(source_id, 0) + 1
        edge_counts[target_id] = edge_counts.get(target_id, 0) + 1
        if len(selected_edges) >= edge_max_total:
            break

    for source_id, target_id, _ in selected_edges:
        s = Block.instances[source_id]
        b = Block.instances[target_id]
        s_color = _normalize_color_name(
            s.color.value if hasattr(s.color, "value") else str(s.color)
        )
        b_color = _normalize_color_name(
            b.color.value if hasattr(b.color, "value") else str(b.color)
        )

        # Determine edge color based on endpoint significance and colors
        if s_color == "red" and b_color == "red":
            edge_color = Colors.RED.value
        elif s_color == "blue" and b_color == "blue":
            edge_color = Colors.BLUE.value
        elif s_color != "black" and b_color != "black":
            # Both significant but different types
            edge_color = Colors.YELLOW.value
        elif show_insignificant_edges:
            edge_color = no_relations_color
        else:
            continue

        # Route Manhattan path while avoiding tile interiors with offset bends.
        path = route_manhattan(s, b, obstacles, grid_system.block_width)
        if len(path) < 2:
            continue
        all_manhattan_paths.append((path, edge_color))

    # Convert to Plotly traces
    # IMPORTANT: Trace order determines z-order (rendering layers)
    # 1. Partition lines via layout shapes (bottom layer, layer="below")
    # 2. Colored tiles (middle layer)
    # 3. Black tile borders (part of tiles)
    # 4. Connecting (crosstalk) lines (above tiles)
    # 5. Labels and click-target markers (topmost)
    traces = []
    tile_traces = []
    click_traces = []
    line_traces = []
    annotations = []

    # Style constants to match paper-style spec
    EDGE_WIDTH = 4  # connecting line width
    # Tile border width scales with maximize flag (full-size vs overview)
    tile_line_width = LINE_WIDTH if maximize else max(3, LINE_WIDTH - 1)

    # Add blocks as filled rectangles
    for block in all_blocks:
        # Rectangle coordinates
        x_coords = [
            block.top_left_p[0],
            block.bottom_right_p[0],
            block.bottom_right_p[0],
            block.top_left_p[0],
            block.top_left_p[0],
        ]
        y_coords = [
            block.top_left_p[1],
            block.top_left_p[1],
            block.bottom_right_p[1],
            block.bottom_right_p[1],
            block.top_left_p[1],
        ]

        pathway_row = suffix_to_row.get(block.id)
        # Handle missing pathway data (shouldn't occur in normal operation, but provides
        # graceful degradation if block.id doesn't match any GS_ID suffix in dataframe)
        row = pathway_row if pathway_row is not None else {}
        payload = {
            "name": row.get("NAME", ""),
            "pathway_id": row.get("GS_ID", ""),
            "fold_change": float(row.get("wFC", np.nan)),
            "pvalue": float(row.get("pFDR", np.nan)),
            "ontology": row.get("Ontology", ""),
            "disease": row.get("Disease", ""),
            "description": row.get("Description", ""),
        }

        # Convert Colors enum to string for Plotly
        fill_color = (
            str(block.color.value)
            if hasattr(block.color, "value")
            else str(block.color)
        )

        tile_traces.append(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                fill="toself",
                fillcolor=fill_color,
                # Tile borders are strong (paper-style)
                line=dict(color=str(Colors.BLACK.value), width=tile_line_width),
                mode="lines",
                hoverinfo="none",
                name="",
                showlegend=False,
                customdata=[payload] * len(x_coords),
                meta={
                    "dataset": dataset_name,
                    "pathway_id": payload["pathway_id"],
                },
            )
        )

        cx = (block.top_left_p[0] + block.bottom_right_p[0]) / 2
        cy = (block.top_left_p[1] + block.bottom_right_p[1]) / 2
        width = abs(block.bottom_right_p[0] - block.top_left_p[0])
        height = abs(block.bottom_right_p[1] - block.top_left_p[1])
        min_side = min(width, height)

        # Scale marker size based on tile dimensions to prevent overflow beyond boundaries
        # min_side is in data coordinates (0-1000 range), but marker size is in pixels
        # Formula: 60% of tile's minimum dimension, capped between 6-18 pixels
        # Examples: 20-unit tile → 12px marker, 200-unit tile → 18px marker (capped)
        marker_size = max(6, min(18, int(min_side * 0.6)))

        # Attach click metadata via invisible markers for Streamlit selection.
        # Marker size already bounded [6,18] from calculation above
        click_traces.append(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers",
                marker=dict(size=marker_size, opacity=0),
                customdata=[payload],
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if show_pathway_ids and min_side >= label_min_side:
            normalized_color = _normalize_color_name(fill_color)
            text_color = (
                "white" if normalized_color in ["red", "blue", "black"] else "black"
            )
            # Use tile center to avoid clipping near edges
            label_y = cy
            font_size = max(8, min(12, int(min_side / 6)))
            annotations.append(
                dict(
                    x=cx,
                    y=label_y,
                    text=block.id,
                    showarrow=False,
                    font=dict(size=font_size, color=text_color),
                    bgcolor=fill_color,
                    bordercolor="black",
                    borderwidth=1,
                )
            )

    # Add Manhattan relationship lines (PAG-to-PAG crosstalk)
    # These should be rendered BEFORE tiles so they appear behind
    for path, edge_color in all_manhattan_paths:
        x_vals = [point[0] for point in path]
        y_vals = [point[1] for point in path]
        line_traces.append(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color=edge_color, width=EDGE_WIDTH),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Combine traces in correct order: tiles first, connecting lines above tiles, clicks on top
    traces = tile_traces + line_traces + click_traces

    # Create figure
    fig = go.Figure(data=traces)

    # Add shapes: partition lines (gray grid) and border
    shapes = []
    if show_partitions:
        # Add partition lines first (bottom layer)
        partition_shapes = create_partition_line_shapes(
            all_blocks,
            grid_system.block_width,
            partition_max_lines,
            partition_color,
            partition_width,
        )
        shapes.extend(partition_shapes)

    # Add black border on top of everything
    shapes.append(
        dict(
            type="rect",
            x0=0,
            y0=0,
            x1=1000,
            y1=1000,
            line=dict(color="black", width=8),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )
    )

    if shapes:
        fig.update_layout(shapes=shapes)

    if annotations:
        fig.update_layout(annotations=annotations)

    if show_legend:
        add_paper_style_legend(fig, no_relations_color)

    # Set figure size based on maximize option
    if maximize:
        height = 1000
        width = 1000
        title_size = 24
    else:
        height = 600
        width = 600
        title_size = 16

    fig.update_layout(
        title=dict(
            text=f"Authentic Mondrian Map: {dataset_name}", font=dict(size=title_size)
        ),
        xaxis=dict(
            range=[0, 1000],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            constrain="domain",
        ),
        yaxis=dict(
            range=[0, 1000],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="white",
        height=height,
        width=width,
        showlegend=show_legend,
        legend=dict(
            orientation="v",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            font=dict(size=12),
        ),
        margin=dict(l=20, r=220 if show_legend else 20, t=60, b=20),
        clickmode="event+select",
        hovermode=False,  # Final, definitive hover disabling
    )

    return fig


def create_canvas_grid(
    df_list: List[pd.DataFrame],
    dataset_names: List[str],
    canvas_rows: int,
    canvas_cols: int,
    show_pathway_ids: bool = True,
) -> go.Figure:
    """Create the canvas grid that holds multiple Mondrian maps"""
    fig = make_subplots(
        rows=canvas_rows,
        cols=canvas_cols,
        subplot_titles=dataset_names[: canvas_rows * canvas_cols],
        specs=[
            [{"type": "xy"} for _ in range(canvas_cols)] for _ in range(canvas_rows)
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    all_shapes = []

    # Add each Mondrian map to its canvas cell
    for idx, (df, name) in enumerate(
        zip(
            df_list[: canvas_rows * canvas_cols],
            dataset_names[: canvas_rows * canvas_cols],
        )
    ):
        row = idx // canvas_cols + 1
        col = idx % canvas_cols + 1

        # Create individual Mondrian map for this dataset
        mondrian_fig = create_authentic_mondrian_map(
            df,
            name,
            maximize=False,
            show_pathway_ids=show_pathway_ids,
        )

        # Add traces to subplot
        for trace in mondrian_fig.data:
            fig.add_trace(trace, row=row, col=col)

        # Add shapes (partition lines and borders) to subplot
        # Transfer shapes to the correct subplot by adjusting xref/yref
        if hasattr(mondrian_fig, "layout") and hasattr(mondrian_fig.layout, "shapes"):
            for shape in mondrian_fig.layout.shapes:
                # Convert Plotly shape object to dictionary
                if hasattr(shape, "to_plotly_json"):
                    shape_copy = shape.to_plotly_json()
                else:
                    shape_copy = shape

                # Set the shape to apply to the specific subplot
                if canvas_rows == 1 and canvas_cols == 1:
                    # Single subplot, use default x/y refs
                    shape_copy["xref"] = "x"
                    shape_copy["yref"] = "y"
                else:
                    # Multiple subplots, need to specify which axis
                    axis_suffix = "" if idx == 0 else str(idx + 1)
                    shape_copy["xref"] = f"x{axis_suffix}"
                    shape_copy["yref"] = f"y{axis_suffix}"
                all_shapes.append(shape_copy)

        # Configure subplot axes
        fig.update_xaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, 1000],
            row=row,
            col=col,
        )
        fig.update_yaxes(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[0, 1000],
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Canvas Grid: Authentic Mondrian Maps",
        showlegend=False,
        plot_bgcolor="white",
        height=200 * canvas_rows + 100,
        width=1200,
        margin=dict(l=50, r=50, t=100, b=50),
        hovermode=False,  # Disable hover completely
        shapes=all_shapes,  # Add all shapes to the figure
    )

    return fig


def create_color_legend() -> go.Figure:
    """Create Mondrian color legend"""
    fig = go.Figure()

    colors = [
        (Colors.WHITE, "Non-significant (p > 0.05)"),
        (Colors.BLACK, "Neutral (|FC| < 0.5)"),
        (Colors.YELLOW, "Moderate (0.5 ≤ |FC| < 1.0)"),
        (Colors.RED, "Up-regulated (FC ≥ 1.0)"),
        (Colors.BLUE, "Down-regulated (FC ≤ -1.0)"),
    ]

    for i, (color, desc) in enumerate(colors):
        # Add colored rectangle
        fig.add_shape(
            type="rect",
            x0=0,
            y0=i * 1.2,
            x1=1,
            y1=i * 1.2 + 1,
            fillcolor=color,
            line=dict(color="black", width=2),
        )

        # Add text description
        fig.add_annotation(
            x=1.5,
            y=i * 1.2 + 0.5,
            text=desc,
            showarrow=False,
            font=dict(size=11, color="black"),
            xanchor="left",
        )

    fig.update_layout(
        xaxis=dict(range=[-0.2, 5], showticklabels=False, showgrid=False),
        yaxis=dict(
            range=[-0.5, len(colors) * 1.2], showticklabels=False, showgrid=False
        ),
        plot_bgcolor="white",
        height=300,
        width=400,
        title="Mondrian Color Scheme",
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode=False,  # Disable hover completely
    )

    return fig
