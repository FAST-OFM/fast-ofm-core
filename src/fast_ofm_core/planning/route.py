"""Deterministic serpentine planning from a neutral physical-unit specification.

This implementation was written for the Fast OFM process protocol.  It does
not import or reproduce the OpenFlexure server planner.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


class PlanningError(ValueError):
    """A route request is invalid or unsupported."""


Point = tuple[float, float]


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanningError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PlanningError(f"{name} must be a finite number")
    return result


def _point(value: object, name: str) -> Point:
    if not isinstance(value, Mapping):
        raise PlanningError(f"{name} must be an object")
    return (_number(value.get("x_um"), f"{name}.x_um"), _number(value.get("y_um"), f"{name}.y_um"))


def _axis_centres(minimum: float, maximum: float, field: float, step: float) -> list[float]:
    """Return centres whose fields cover an interval, including both edges."""
    if maximum <= minimum:
        raise PlanningError("bounds maximum must exceed minimum")
    if field <= 0 or step <= 0:
        raise PlanningError("field size and route step must be positive")
    span = maximum - minimum
    if span <= field:
        return [(minimum + maximum) / 2.0]
    first = minimum + field / 2.0
    last = maximum - field / 2.0
    count = int(math.floor((last - first) / step)) + 1
    centres = [first + index * step for index in range(count)]
    if last - centres[-1] > max(1e-9, step * 1e-12):
        centres.append(last)
    return centres


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return true for points inside or on the boundary of a simple polygon."""
    x, y = point
    inside = False
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        x1, y1 = first
        x2, y2 = second
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross) <= 1e-9
            and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9
            and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9
        ):
            return True
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    values = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    if any(abs(value) <= 1e-9 for value in values):
        # Conservative edge inclusion is appropriate for tissue coverage.
        boxes_overlap = not (
            max(a[0], b[0]) < min(c[0], d[0]) - 1e-9
            or max(c[0], d[0]) < min(a[0], b[0]) - 1e-9
            or max(a[1], b[1]) < min(c[1], d[1]) - 1e-9
            or max(c[1], d[1]) < min(a[1], b[1]) - 1e-9
        )
        if boxes_overlap and all(abs(value) <= 1e-9 for value in values):
            return True
    return values[0] * values[1] <= 0 and values[2] * values[3] <= 0


def _field_intersects_polygon(centre: Point, field: Point, polygon: Sequence[Point]) -> bool:
    half_x, half_y = field[0] / 2.0, field[1] / 2.0
    x, y = centre
    corners = (
        (x - half_x, y - half_y),
        (x + half_x, y - half_y),
        (x + half_x, y + half_y),
        (x - half_x, y + half_y),
    )
    if any(_point_in_polygon(corner, polygon) for corner in corners):
        return True
    if any(x - half_x <= px <= x + half_x and y - half_y <= py <= y + half_y for px, py in polygon):
        return True
    rectangle_edges = tuple((corners[index], corners[(index + 1) % 4]) for index in range(4))
    polygon_edges = tuple(
        (polygon[index], polygon[(index + 1) % len(polygon)]) for index in range(len(polygon))
    )
    return any(
        _segments_intersect(*rectangle, *edge)
        for rectangle in rectangle_edges
        for edge in polygon_edges
    )


def _polygon(value: object, index: int) -> tuple[Point, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        raise PlanningError(f"region_polygons[{index}] must have at least three points")
    result = tuple(_point(point, f"region_polygons[{index}]") for point in value)
    if (
        abs(
            sum(
                result[i][0] * result[(i + 1) % len(result)][1]
                - result[(i + 1) % len(result)][0] * result[i][1]
                for i in range(len(result))
            )
        )
        <= 1e-9
    ):
        raise PlanningError(f"region_polygons[{index}] has zero area")
    return result


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise PlanningError(f"{name} must be a positive integer")
    return value


def _exact_grid(payload: Mapping[str, object]) -> tuple[tuple[str, list[Point]], ...]:
    """Expand a frozen physical grid without inferring camera geometry."""
    grid = payload.get("grid")
    if not isinstance(grid, Mapping):
        raise PlanningError("grid must be an object")
    origin = _point(grid.get("origin_um"), "grid.origin_um")
    step = _point(grid.get("step_um"), "grid.step_um")
    if step[0] == 0 or step[1] == 0:
        raise PlanningError("grid steps must be non-zero")
    columns = _positive_integer(grid.get("columns"), "grid.columns")
    rows = _positive_integer(grid.get("rows"), "grid.rows")
    points: list[Point] = []
    for row_index in range(rows):
        x_indices = range(columns - 1, -1, -1) if row_index % 2 else range(columns)
        for column_index in x_indices:
            points.append(
                (
                    origin[0] + column_index * step[0],
                    origin[1] + row_index * step[1],
                )
            )
    return (("component-1", points),)


def plan_route(payload: Mapping[str, object]) -> dict[str, Any]:
    """Plan deterministic physical positions for one or more scan components."""
    traversal = payload.get("traversal")
    if traversal == "exact_grid_serpentine":
        components = _exact_grid(payload)
        anchor_spacing = payload.get("focus_anchor_spacing_um")
        spacing = (
            None if anchor_spacing is None else _number(anchor_spacing, "focus_anchor_spacing_um")
        )
        if spacing is not None and spacing <= 0:
            raise PlanningError("focus_anchor_spacing_um must be positive")
        return _route_result(payload, components, spacing)

    bounds = payload.get("bounds")
    if not isinstance(bounds, Mapping):
        raise PlanningError("bounds must be an object")
    minimum = _point(bounds.get("minimum_um"), "bounds.minimum_um")
    maximum = _point(bounds.get("maximum_um"), "bounds.maximum_um")
    field = _point(payload.get("field_size_um"), "field_size_um")
    overlap = _number(payload.get("overlap_fraction"), "overlap_fraction")
    if not 0 <= overlap < 1:
        raise PlanningError("overlap_fraction must be in [0, 1)")
    if traversal not in {"serpentine", "component_serpentine"}:
        raise PlanningError("only serpentine and component_serpentine are implemented")
    if any(value <= 0 for value in field):
        raise PlanningError("field dimensions must be positive")

    x_values = _axis_centres(minimum[0], maximum[0], field[0], field[0] * (1 - overlap))
    y_values = _axis_centres(minimum[1], maximum[1], field[1], field[1] * (1 - overlap))
    raw_polygons = payload.get("region_polygons")
    polygons = (
        tuple(_polygon(value, index) for index, value in enumerate(raw_polygons))
        if isinstance(raw_polygons, Sequence) and not isinstance(raw_polygons, (str, bytes))
        else ()
    )
    if traversal == "component_serpentine" and not polygons:
        raise PlanningError("component_serpentine requires region_polygons")

    components: list[tuple[str, list[Point]]] = []
    source_components = polygons or ((),)
    for component_index, polygon in enumerate(source_components):
        points: list[Point] = []
        for row_index, y in enumerate(y_values):
            row = list(reversed(x_values)) if row_index % 2 else list(x_values)
            for x in row:
                point = (x, y)
                if not polygon or _field_intersects_polygon(point, field, polygon):
                    points.append(point)
        if points:
            components.append((f"component-{component_index + 1}", points))

    anchor_spacing = payload.get("focus_anchor_spacing_um")
    spacing = None if anchor_spacing is None else _number(anchor_spacing, "focus_anchor_spacing_um")
    if spacing is not None and spacing <= 0:
        raise PlanningError("focus_anchor_spacing_um must be positive")

    return _route_result(payload, tuple(components), spacing)


def _route_result(
    payload: Mapping[str, object],
    components: Sequence[tuple[str, list[Point]]],
    spacing: float | None,
) -> dict[str, Any]:
    """Serialise one deterministic route and its optional sparse-focus markers."""
    actions: list[dict[str, object]] = []
    action_index = 0
    for component_id, points in components:
        first = points[0]
        actions.append(
            {
                "index": action_index,
                "kind": "component_start",
                "position_um": {"x_um": first[0], "y_um": first[1]},
                "component_id": component_id,
            }
        )
        action_index += 1
        last_anchor: Point | None = None
        for point in points:
            kind = "visit"
            if spacing is not None and (
                last_anchor is None or math.dist(point, last_anchor) >= spacing
            ):
                kind = "focus_anchor"
                last_anchor = point
            actions.append(
                {
                    "index": action_index,
                    "kind": kind,
                    "position_um": {"x_um": point[0], "y_um": point[1]},
                    "component_id": component_id,
                }
            )
            action_index += 1

    digest = _canonical_digest(payload)
    return {
        "route_id": f"route-{digest[:16]}",
        "input_digest": digest,
        "actions": actions,
        "component_count": len(components),
    }
