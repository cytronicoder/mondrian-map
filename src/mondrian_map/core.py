"""
Core Mondrian Map Algorithm Implementation

This module contains the authentic implementation of the Mondrian Map algorithm
as described in the bioRxiv paper: https://www.biorxiv.org/content/10.1101/2024.04.11.589093v2

The implementation follows the exact 3-stage process:
1. Grid System initialization
2. Block placement based on pathway data
3. Line generation for authentic Mondrian aesthetics
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

# Algorithm Constants
LINE_WIDTH = 5
THIN_LINE_WIDTH = 1
adjust = LINE_WIDTH // 2
adjust_e = adjust + 1
adjust_d = adjust_e - adjust
AREA_SCALAR = 4000

# Thresholds for regulation classification
up_th = 1.25
dn_th = abs(1 - (up_th - 1))


def rect_intersects(r1, r2, padding: float = 0.0) -> bool:
    """Check if two rectangles intersect, with optional padding."""
    (x0a, y0a), (x1a, y1a) = r1
    (x0b, y0b), (x1b, y1b) = r2

    minx1, maxx1 = sorted([x0a, x1a])
    miny1, maxy1 = sorted([y0a, y1a])
    minx2, maxx2 = sorted([x0b, x1b])
    miny2, maxy2 = sorted([y0b, y1b])

    minx1 -= padding
    maxx1 += padding
    miny1 -= padding
    maxy1 += padding
    minx2 -= padding
    maxx2 += padding
    miny2 -= padding
    maxy2 += padding

    if maxx1 <= minx2 or maxx2 <= minx1:
        return False
    if maxy1 <= miny2 or maxy2 <= miny1:
        return False
    return True


def count_overlaps(rects: list, padding: float = 0.0) -> int:
    """Count pairwise rectangle overlaps."""
    overlaps = 0
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if rect_intersects(rects[i], rects[j], padding=padding):
                overlaps += 1
    return overlaps


def snap_rect_to_grid(rect, grid: int = 20, bounds=(0, 1000)):
    """Snap a rectangle to the nearest grid within bounds."""
    (x0, y0), (x1, y1) = rect
    minx, maxx = sorted([x0, x1])
    miny, maxy = sorted([y0, y1])

    def _snap(val: float) -> float:
        snapped = round(val / grid) * grid
        return max(bounds[0], min(bounds[1], snapped))

    minx = _snap(minx)
    maxx = _snap(maxx)
    miny = _snap(miny)
    maxy = _snap(maxy)

    minx, maxx = sorted([minx, maxx])
    miny, maxy = sorted([miny, maxy])

    return [(minx, miny), (maxx, maxy)]


class Colors(str, Enum):
    """Authentic Mondrian color palette"""

    WHITE = "#FFFFFF"
    GRAY = "#3e3f39"
    LIGHT_GRAY = "#D3D3D3"
    BLACK = "#050103"
    BLACK_A = "#05010333"
    BLACK_AA = "#050103AA"
    RED = "#E70503"
    BLUE = "#0300AD"
    YELLOW = "#FDDE06"
    RED_A = "#E70503AA"
    BLUE_A = "#0300ADAA"
    YELLOW_A = "#FDDE06AA"


class CornerPos(int, Enum):
    """Corner position enumeration"""

    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3


class LineDir(str, Enum):
    """Line direction enumeration"""

    RIGHT = "left_to_right"
    LEFT = "right_to_left"
    DOWN = "up_to_down"
    UP = "down_to_up"


class Point:
    """Represents a 2D point"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({round(self.x, 2)}, {round(self.y, 2)})"


class Line:
    """Represents a line in the Mondrian map"""

    instances = []

    def __init__(
        self,
        point_a: Point,
        point_b: Point,
        direction: LineDir,
        color: Colors = Colors.BLACK,
        strength: int = LINE_WIDTH,
    ):
        self.point_a = point_a
        self.point_b = point_b
        self.direction = direction
        self.color = color
        self.strength = strength
        self.instances.append(self)

    def __str__(self):
        return f"({self.point_a.x}, {self.point_a.y}) to ({self.point_b.x}, {self.point_b.y})"


class Corner:
    """Represents a corner point of a block"""

    instances = []

    def __init__(self, point: Point, position: CornerPos, line: Line = None):
        self.point = point
        self.position = position
        self.line = line
        self.instances.append(self)

    def __str__(self):
        return f"{self.position}: ({round(self.point.x, 2)}, {round(self.point.y, 2)})"


class Block:
    """Represents a pathway block in the Mondrian map"""

    instances = {}

    def __init__(
        self,
        top_left: Tuple[float, float],
        bottom_right: Tuple[float, float],
        area: float,
        color: str,
        id: str,
    ):
        self.top_left_p = top_left
        self.bottom_right_p = bottom_right

        # Create corners with proper adjustments
        self.top_left = Corner(
            Point(self.top_left_p[0] - adjust, self.top_left_p[1] - adjust),
            CornerPos.TOP_LEFT,
        )
        self.top_right = Corner(
            Point(self.bottom_right_p[0] + adjust, self.top_left_p[1] - adjust),
            CornerPos.TOP_RIGHT,
        )
        self.bottom_left = Corner(
            Point(self.top_left_p[0] - adjust, self.bottom_right_p[1] + adjust),
            CornerPos.BOTTOM_LEFT,
        )
        self.bottom_right = Corner(
            Point(self.bottom_right_p[0] + adjust, self.bottom_right_p[1] + adjust),
            CornerPos.BOTTOM_RIGHT,
        )

        self.center = Point(
            (self.top_left.point.x + self.bottom_right.point.x) / 2,
            (self.top_left.point.y + self.bottom_right.point.y) / 2,
        )
        self.area = area
        self.color = self.get_color_map(color)
        self.id = id
        self.instances[id] = self

        # Create block boundary lines
        Line(
            Point(self.top_left.point.x, self.top_left.point.y + adjust),
            Point(self.top_right.point.x, self.top_right.point.y + adjust),
            LineDir.RIGHT,
        )
        Line(
            Point(self.top_right.point.x - adjust, self.top_right.point.y),
            Point(self.bottom_right.point.x - adjust, self.bottom_right.point.y),
            LineDir.DOWN,
        )
        Line(
            Point(self.bottom_right.point.x, self.bottom_right.point.y - adjust),
            Point(self.bottom_left.point.x, self.bottom_left.point.y - adjust),
            LineDir.LEFT,
        )
        Line(
            Point(self.bottom_left.point.x + adjust, self.bottom_left.point.y),
            Point(self.top_left.point.x + adjust, self.top_left.point.y),
            LineDir.UP,
        )

    def get_color_map(self, color: str) -> Colors:
        """Map color string to Colors enum"""
        color_map = {
            "red": Colors.RED,
            "blue": Colors.BLUE,
            "yellow": Colors.YELLOW,
            "black": Colors.BLACK,
            "gray": Colors.GRAY,
        }
        return color_map.get(color, Colors.BLACK)

    @property
    def height(self) -> float:
        return self.bottom_left.point.y - self.top_left.point.y

    @property
    def width(self) -> float:
        return self.top_right.point.x - self.top_left.point.x


class GridSystem:
    """Authentic grid system for Mondrian map generation"""

    def __init__(self, width: int, height: int, block_width: int, block_height: int):
        self.width = width
        self.height = height
        self.block_width = block_width
        self.block_height = block_height

        self.grid_lines_h = {}
        self.grid_lines_v = {}

        # Create horizontal grid lines
        for i in range(height // block_height + 1):
            self.grid_lines_h[f"h{i}"] = i * block_height

        # Create vertical grid lines
        for i in range(width // block_width + 1):
            self.grid_lines_v[f"v{i}"] = i * block_width

    def fill_blocks_around_point(
        self, point: Tuple[float, float], target_area: float
    ) -> Tuple[List[Tuple[float, float]], float]:
        """
        Creates a rectangle centered precisely on the given point (x, y)
        with an area as close as possible to the target_area.
        The rectangle's dimensions (width/height) are proportional to maintain a squarish shape.
        """
        x, y = point

        # Calculate width and height from area, maintaining a ~4:3 aspect ratio for aesthetics
        # This ratio can be adjusted if needed. The key is that width * height = target_area
        aspect_ratio = 4 / 3
        height = math.sqrt(target_area / aspect_ratio)
        width = aspect_ratio * height

        # Calculate top-left and bottom-right coordinates so (x, y) is the exact center
        top_left_x = x - width / 2
        top_left_y = y - height / 2
        bottom_right_x = x + width / 2
        bottom_right_y = y + height / 2

        # Ensure coordinates are within canvas bounds (0 to 1000)
        top_left_x = max(0, top_left_x)
        top_left_y = max(0, top_left_y)
        bottom_right_x = min(self.width, bottom_right_x)
        bottom_right_y = min(self.height, bottom_right_y)

        # The actual area might differ slightly due to snapping to bounds
        actual_area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
        area_diff = abs(target_area - actual_area)

        return [(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)], area_diff

    def approximate_grid_layout(self, nob: int) -> Tuple[int, int]:
        """Approximate the best grid layout for a given number of blocks"""
        if nob == 1:
            return 1, 1
        elif nob <= 4:
            return 2, 2
        else:
            sqrt_nob = int(math.sqrt(nob))
            return sqrt_nob, int(math.ceil(nob / sqrt_nob))

    def plot_points_fill_blocks(
        self,
        points: List[Tuple[float, float]],
        target_areas: List[float],
        avoid_overlap: bool = False,
        padding: float = LINE_WIDTH,
        max_scale_iters: int = 25,
        min_scale: float = 0.20,
        snap_to_grid: bool = True,
        nudge: bool = True,
        nudge_step: float = 10.0,
        nudge_radius: float = 120.0,
    ) -> List[List[Tuple[float, float]]]:
        """Plot points and fill blocks based on target areas"""
        if not avoid_overlap:
            rectangles = []
            area_diff = 0

            for point, target_area in zip(points, target_areas):
                rect, diff = self.fill_blocks_around_point(point, target_area)
                rectangles.append(rect)
                area_diff += diff

            return rectangles

        scale = 1.0
        rectangles = []
        areas_scaled = [area * scale for area in target_areas]

        for _ in range(max_scale_iters):
            areas_scaled = [area * scale for area in target_areas]
            rectangles = []
            for point, target_area in zip(points, areas_scaled):
                rect, _diff = self.fill_blocks_around_point(point, target_area)
                if snap_to_grid:
                    rect = snap_rect_to_grid(rect, grid=20, bounds=(0, 1000))
                rectangles.append(rect)

            if count_overlaps(rectangles, padding=padding) == 0:
                return rectangles

            scale *= 0.9
            if scale < min_scale:
                break

        if not nudge:
            return rectangles

        ordered_indices = sorted(
            range(len(areas_scaled)), key=lambda i: areas_scaled[i], reverse=True
        )
        accepted_rects = []
        final_rects = [None] * len(rectangles)

        def _offset_candidates():
            yield (0.0, 0.0)
            r = nudge_step
            while r <= nudge_radius:
                offsets = [
                    (r, 0),
                    (-r, 0),
                    (0, r),
                    (0, -r),
                    (r, r),
                    (r, -r),
                    (-r, r),
                    (-r, -r),
                ]
                for offset in offsets:
                    yield offset
                r += nudge_step

        for idx in ordered_indices:
            base_point = points[idx]
            base_area = areas_scaled[idx]
            base_rect = rectangles[idx]
            placed_rect = base_rect

            if any(
                rect_intersects(base_rect, accepted, padding=padding)
                for accepted in accepted_rects
            ):
                for dx, dy in _offset_candidates():
                    candidate_point = (base_point[0] + dx, base_point[1] + dy)
                    candidate_rect, _diff = self.fill_blocks_around_point(
                        candidate_point, base_area
                    )
                    if snap_to_grid:
                        candidate_rect = snap_rect_to_grid(
                            candidate_rect, grid=20, bounds=(0, 1000)
                        )
                    if not any(
                        rect_intersects(candidate_rect, accepted, padding=padding)
                        for accepted in accepted_rects
                    ):
                        placed_rect = candidate_rect
                        break

            accepted_rects.append(placed_rect)
            final_rects[idx] = placed_rect

        return final_rects


def blank_canvas():
    """Reset all instances for a fresh canvas"""
    Corner.instances = []
    Block.instances = {}
    Line.instances = []


def euclidean_distance_point(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    """Calculate Euclidean distance between two points"""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_line_direction(point_a: Point, point_b: Point) -> Optional[LineDir]:
    """Determine line direction between two points"""
    if abs(point_a.x - point_b.x) <= adjust:  # due to adjustment error
        if point_a.y < point_b.y:
            return LineDir.DOWN
        else:
            return LineDir.UP
    elif abs(point_a.y - point_b.y) <= adjust:
        if point_a.x < point_b.x:
            return LineDir.RIGHT
        else:
            return LineDir.LEFT
    else:
        return None
