"""
Visualization Module for Mondrian Maps

This module contains the visualization functions including the complex
line generation algorithms and Plotly figure creation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .core import (
    LINE_WIDTH,
    THIN_LINE_WIDTH,
    Block,
    Colors,
    Corner,
    CornerPos,
    GridSystem,
    Line,
    LineDir,
    Point,
    adjust,
    adjust_d,
    adjust_e,
    blank_canvas,
    euclidean_distance_point,
    get_line_direction,
)
from .data_processing import get_mondrian_color_description, prepare_pathway_data


def get_closest_corner(block_a: Block, block_b: Block) -> Corner:
    """Find the closest corner of block_a to block_b's center"""
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
    """Get the furthest connector point between two corners"""
    p = Point(cp1.point.x, cp2.point.y)
    q = Point(cp2.point.x, cp1.point.y)
    if euclidean_distance_point(
        (p.x, p.y), (center.x, center.y)
    ) > euclidean_distance_point((q.x, q.y), (center.x, center.y)):
        return p
    else:
        return q


def get_manhattan_line_color(block_a: Block, block_b: Block) -> Colors:
    """Determine the color of Manhattan lines between blocks"""
    if block_a.color == Colors.RED and block_b.color == Colors.RED:
        return Colors.RED
    elif block_a.color == Colors.BLUE and block_b.color == Colors.BLUE:
        return Colors.BLUE
    else:
        return Colors.YELLOW


def get_manhattan_lines_2(
    corner_a: Corner, corner_b: Corner, connector: Point, color: Colors
) -> List[Line]:
    """Generate Manhattan-style connection lines between corners"""
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
    Two-step grid line algorithm:
    1. Create full Manhattan grid
    2. Remove lines that go canvas edge to edge without hitting tile corners
    3. Trim line segments that cross tile middles, stopping at tile corners
    """
    smart_lines = []

    # Get all tile corner positions
    tile_corners = set()
    for block in blocks:
        tile_corners.add((block.top_left_p[0], block.top_left_p[1]))  # top-left
        tile_corners.add((block.bottom_right_p[0], block.top_left_p[1]))  # top-right
        tile_corners.add((block.top_left_p[0], block.bottom_right_p[1]))  # bottom-left
        tile_corners.add(
            (block.bottom_right_p[0], block.bottom_right_p[1])
        )  # bottom-right

    # Get all unique x and y positions from corners
    x_positions = set(corner[0] for corner in tile_corners)
    y_positions = set(corner[1] for corner in tile_corners)

    # STEP 1: Create full Manhattan grid lines
    full_grid_lines = []

    # Create full vertical lines
    for x_pos in x_positions:
        if 0 < x_pos < 1000:  # Skip canvas boundaries
            full_grid_lines.append(
                {"type": "vertical", "position": x_pos, "start": 0, "end": 1000}
            )

    # Create full horizontal lines
    for y_pos in y_positions:
        if 0 < y_pos < 1000:  # Skip canvas boundaries
            full_grid_lines.append(
                {"type": "horizontal", "position": y_pos, "start": 0, "end": 1000}
            )

    # STEP 2: Remove lines that go canvas edge to edge without hitting tile corners
    valid_lines = []
    for line in full_grid_lines:
        hits_corner = False

        if line["type"] == "vertical":
            x_pos = line["position"]
            # Check if this vertical line hits any tile corner
            for corner_x, corner_y in tile_corners:
                if corner_x == x_pos:
                    hits_corner = True
                    break
        else:  # horizontal
            y_pos = line["position"]
            # Check if this horizontal line hits any tile corner
            for corner_x, corner_y in tile_corners:
                if corner_y == y_pos:
                    hits_corner = True
                    break

        if hits_corner:
            valid_lines.append(line)

    # STEP 3: Create line segments from canvas edge to tile borders (stop at first tile encounter)
    for line in valid_lines:
        if line["type"] == "vertical":
            x_pos = line["position"]

            # Find all tile boundaries that this vertical line would encounter
            tile_boundaries = []
            for block in blocks:
                if block.top_left_p[0] <= x_pos <= block.bottom_right_p[0]:
                    tile_boundaries.append(
                        (block.top_left_p[1], block.bottom_right_p[1], "tile")
                    )

            # Sort boundaries by y position
            tile_boundaries.sort()

            # Create line from top canvas edge to first tile boundary
            if tile_boundaries:
                first_tile_top = tile_boundaries[0][0]
                if first_tile_top > 0:
                    smart_lines.append(
                        Line(
                            Point(x_pos, 0),
                            Point(x_pos, first_tile_top),
                            LineDir.DOWN,
                            Colors.LIGHT_GRAY,
                            1,
                        )
                    )

                # Create lines between tiles (in gaps)
                for i in range(len(tile_boundaries) - 1):
                    gap_start = tile_boundaries[i][1]  # End of current tile
                    gap_end = tile_boundaries[i + 1][0]  # Start of next tile

                    if gap_end - gap_start > 10:  # Minimum gap size
                        smart_lines.append(
                            Line(
                                Point(x_pos, gap_start),
                                Point(x_pos, gap_end),
                                LineDir.DOWN,
                                Colors.LIGHT_GRAY,
                                1,
                            )
                        )

                # Create line from last tile boundary to bottom canvas edge
                last_tile_bottom = tile_boundaries[-1][1]
                if last_tile_bottom < 1000:
                    smart_lines.append(
                        Line(
                            Point(x_pos, last_tile_bottom),
                            Point(x_pos, 1000),
                            LineDir.DOWN,
                            Colors.LIGHT_GRAY,
                            1,
                        )
                    )
            else:
                # No tiles intersect this line, draw full line
                smart_lines.append(
                    Line(
                        Point(x_pos, 0),
                        Point(x_pos, 1000),
                        LineDir.DOWN,
                        Colors.LIGHT_GRAY,
                        1,
                    )
                )

        else:  # horizontal line
            y_pos = line["position"]

            # Find all tile boundaries that this horizontal line would encounter
            tile_boundaries = []
            for block in blocks:
                if block.top_left_p[1] <= y_pos <= block.bottom_right_p[1]:
                    tile_boundaries.append(
                        (block.top_left_p[0], block.bottom_right_p[0], "tile")
                    )

            # Sort boundaries by x position
            tile_boundaries.sort()

            # Create line from left canvas edge to first tile boundary
            if tile_boundaries:
                first_tile_left = tile_boundaries[0][0]
                if first_tile_left > 0:
                    smart_lines.append(
                        Line(
                            Point(0, y_pos),
                            Point(first_tile_left, y_pos),
                            LineDir.RIGHT,
                            Colors.LIGHT_GRAY,
                            1,
                        )
                    )

                # Create lines between tiles (in gaps)
                for i in range(len(tile_boundaries) - 1):
                    gap_start = tile_boundaries[i][1]  # End of current tile
                    gap_end = tile_boundaries[i + 1][0]  # Start of next tile

                    if gap_end - gap_start > 10:  # Minimum gap size
                        smart_lines.append(
                            Line(
                                Point(gap_start, y_pos),
                                Point(gap_end, y_pos),
                                LineDir.RIGHT,
                                Colors.LIGHT_GRAY,
                                1,
                            )
                        )

                # Create line from last tile boundary to right canvas edge
                last_tile_right = tile_boundaries[-1][1]
                if last_tile_right < 1000:
                    smart_lines.append(
                        Line(
                            Point(last_tile_right, y_pos),
                            Point(1000, y_pos),
                            LineDir.RIGHT,
                            Colors.LIGHT_GRAY,
                            1,
                        )
                    )
            else:
                # No tiles intersect this line, draw full line
                smart_lines.append(
                    Line(
                        Point(0, y_pos),
                        Point(1000, y_pos),
                        LineDir.RIGHT,
                        Colors.LIGHT_GRAY,
                        1,
                    )
                )

    return smart_lines


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
    mem_df: Optional[pd.DataFrame] = None,
    maximize: bool = False,
    show_pathway_ids: bool = True,
) -> go.Figure:
    """
    Create authentic Mondrian map using the exact algorithm from the notebooks
    """
    if len(df) == 0:
        return go.Figure()

    # Prepare data using the data processing module
    network_dir = Path("data/case_study/pathway_networks")
    data = prepare_pathway_data(df, dataset_name, network_dir)

    center_points = data["center_points"]
    areas = data["areas"]
    colors = data["colors"]
    pathway_ids = data["pathway_ids"]
    relations = data["relations"]
    suffix_to_row = {str(row["GS_ID"])[-4:]: row for _, row in df.iterrows()}

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

    # Create border lines
    Line(Point(0, 0), Point(1000, 0), LineDir.RIGHT, Colors.GRAY, THIN_LINE_WIDTH)
    Line(Point(1000, 0), Point(1000, 1000), LineDir.DOWN, Colors.GRAY, THIN_LINE_WIDTH)
    Line(Point(1000, 1000), Point(0, 1000), LineDir.LEFT, Colors.GRAY, THIN_LINE_WIDTH)
    Line(Point(0, 1000), Point(0, 0), LineDir.UP, Colors.GRAY, THIN_LINE_WIDTH)

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

    # Create smart grid lines that avoid intersecting tiles (after blocks are created)
    smart_grid_lines = create_smart_grid_lines(grid_system, all_blocks)

    # STAGE 2: Create relationship lines (Manhattan lines for PAG-to-PAG crosstalk)
    all_manhattan_lines = []
    lines_to_extend = []
    for rel in relations:
        if rel[0] in Block.instances.keys() and rel[1] in Block.instances.keys():
            s = Block.instances[rel[0]]
            b = Block.instances[rel[1]]
        else:
            continue

        manhattan_line_color = get_manhattan_line_color(s, b)

        cp1 = get_closest_corner(s, b)
        cp2 = None
        dist = float("inf")

        for corner in [b.top_left, b.top_right, b.bottom_left, b.bottom_right]:
            if (
                s.top_left.point.x > corner.point.x
                or s.top_right.point.x < corner.point.x
            ) and (
                s.top_left.point.y > corner.point.y
                or s.bottom_left.point.y < corner.point.y
            ):
                d = euclidean_distance_point(
                    (s.center.x, s.center.y), (corner.point.x, corner.point.y)
                )
                if d < dist:
                    cp2 = corner
                    dist = d
        if cp2 == None:
            cp2 = get_closest_corner(b, s)
            con = get_furthest_connector(cp1, cp2, s.center)
        else:
            con = get_furthest_connector(cp1, cp2, b.center)

        lines = get_manhattan_lines_2(cp1, cp2, con, manhattan_line_color)

        if len(lines) == 1:
            all_manhattan_lines.append(lines[0])

        if len(lines) == 2:
            all_manhattan_lines.extend(lines)
            lines_to_extend.extend(lines)

    # Convert to Plotly traces
    traces = []

    # Add smart grid lines (lightest gray, thin lines)
    for line in smart_grid_lines:
        traces.append(
            go.Scatter(
                x=[line.point_a.x, line.point_b.x],
                y=[line.point_a.y, line.point_b.y],
                mode="lines",
                line=dict(color="#F5F5F5", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

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
        payload = {
            "name": pathway_row.get("NAME", "") if pathway_row is not None else "",
            "pathway_id": pathway_row.get("GS_ID", "") if pathway_row is not None else "",
            "fold_change": float(pathway_row.get("wFC", np.nan))
            if pathway_row is not None
            else float("nan"),
            "pvalue": float(pathway_row.get("pFDR", np.nan))
            if pathway_row is not None
            else float("nan"),
            "ontology": pathway_row.get("Ontology", "")
            if pathway_row is not None
            else "",
            "disease": pathway_row.get("Disease", "")
            if pathway_row is not None
            else "",
            "description": pathway_row.get("Description", "")
            if pathway_row is not None
            else "",
        }

        # Convert Colors enum to string for Plotly
        fill_color = (
            str(block.color.value)
            if hasattr(block.color, "value")
            else str(block.color)
        )

        traces.append(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=str(Colors.BLACK.value), width=LINE_WIDTH),
                mode="lines",
                hoverinfo="none",
                name="",
                showlegend=False,
                customdata=[payload],
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

        traces.append(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers",
                marker=dict(size=18, opacity=0),
                customdata=[payload],
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if show_pathway_ids and min_side >= 30:
            font_size = max(8, min(16, int(min_side / 6)))
            traces.append(
                go.Scatter(
                    x=[cx],
                    y=[cy],
                    mode="text",
                    text=[block.id],
                    textposition="middle center",
                    textfont=dict(size=font_size, color="black"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Add canvas border lines
    border_lines = [line for line in Line.instances if line.strength == THIN_LINE_WIDTH]

    for line in border_lines:
        traces.append(
            go.Scatter(
                x=[line.point_a.x, line.point_b.x],
                y=[line.point_a.y, line.point_b.y],
                mode="lines",
                line=dict(color="#808080", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add Manhattan relationship lines (PAG-to-PAG crosstalk)
    for line in all_manhattan_lines:
        line_color = (
            str(line.color.value) if hasattr(line.color, "value") else str(line.color)
        )
        traces.append(
            go.Scatter(
                x=[line.point_a.x, line.point_b.x],
                y=[line.point_a.y, line.point_b.y],
                mode="lines",
                line=dict(color=line_color, width=LINE_WIDTH),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Create figure
    fig = go.Figure(data=traces)

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
            range=[0, 1000], showticklabels=False, showgrid=False, zeroline=False
        ),
        yaxis=dict(
            range=[0, 1000], showticklabels=False, showgrid=False, zeroline=False
        ),
        plot_bgcolor="white",
        height=height,
        width=width,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
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
            df, name, mem_df=None, maximize=False, show_pathway_ids=show_pathway_ids
        )

        # Add traces to subplot
        for trace in mondrian_fig.data:
            fig.add_trace(trace, row=row, col=col)

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
