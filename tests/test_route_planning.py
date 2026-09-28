from __future__ import annotations

from fast_ofm_core.planning import PlanningError, plan_route


def rectangular_request() -> dict[str, object]:
    return {
        "bounds": {
            "minimum_um": {"x_um": 0.0, "y_um": 0.0},
            "maximum_um": {"x_um": 2500.0, "y_um": 1500.0},
        },
        "field_size_um": {"x_um": 1000.0, "y_um": 1000.0},
        "overlap_fraction": 0.2,
        "traversal": "serpentine",
        "focus_anchor_spacing_um": 1500.0,
    }


def test_serpentine_covers_both_bounds_and_reverses_rows() -> None:
    result = plan_route(rectangular_request())
    visits = [action for action in result["actions"] if action["kind"] != "component_start"]
    assert result["component_count"] == 1
    assert [(item["position_um"]["x_um"], item["position_um"]["y_um"]) for item in visits] == [
        (500.0, 500.0),
        (1300.0, 500.0),
        (2000.0, 500.0),
        (2000.0, 1000.0),
        (1300.0, 1000.0),
        (500.0, 1000.0),
    ]


def test_disconnected_polygons_create_separate_components() -> None:
    request = rectangular_request()
    request["traversal"] = "component_serpentine"
    request["region_polygons"] = [
        [
            {"x_um": 0.0, "y_um": 0.0},
            {"x_um": 900.0, "y_um": 0.0},
            {"x_um": 900.0, "y_um": 900.0},
            {"x_um": 0.0, "y_um": 900.0},
        ],
        [
            {"x_um": 1600.0, "y_um": 600.0},
            {"x_um": 2500.0, "y_um": 600.0},
            {"x_um": 2500.0, "y_um": 1500.0},
            {"x_um": 1600.0, "y_um": 1500.0},
        ],
    ]
    result = plan_route(request)
    starts = [action for action in result["actions"] if action["kind"] == "component_start"]
    assert result["component_count"] == 2
    assert [item["component_id"] for item in starts] == ["component-1", "component-2"]


def test_exact_grid_preserves_signed_steps_and_frozen_field_count() -> None:
    result = plan_route(
        {
            "traversal": "exact_grid_serpentine",
            "grid": {
                "origin_um": {"x_um": 100.0, "y_um": 200.0},
                "step_um": {"x_um": 10.0, "y_um": -20.0},
                "columns": 5,
                "rows": 3,
            },
        }
    )
    visits = [action for action in result["actions"] if action["kind"] != "component_start"]
    assert [(item["position_um"]["x_um"], item["position_um"]["y_um"]) for item in visits] == [
        (100.0, 200.0),
        (110.0, 200.0),
        (120.0, 200.0),
        (130.0, 200.0),
        (140.0, 200.0),
        (140.0, 180.0),
        (130.0, 180.0),
        (120.0, 180.0),
        (110.0, 180.0),
        (100.0, 180.0),
        (100.0, 160.0),
        (110.0, 160.0),
        (120.0, 160.0),
        (130.0, 160.0),
        (140.0, 160.0),
    ]


def test_exact_grid_refuses_zero_step_and_non_integer_count() -> None:
    request = {
        "traversal": "exact_grid_serpentine",
        "grid": {
            "origin_um": {"x_um": 0.0, "y_um": 0.0},
            "step_um": {"x_um": 1.0, "y_um": 0.0},
            "columns": 1.5,
            "rows": 1,
        },
    }
    try:
        plan_route(request)
    except PlanningError as error:
        assert "non-zero" in str(error) or "positive integer" in str(error)
    else:
        raise AssertionError("invalid exact grid was accepted")


def test_digest_and_route_are_deterministic() -> None:
    first = plan_route(rectangular_request())
    second = plan_route(rectangular_request())
    assert first == second


def test_spiral_is_explicitly_refused_not_silently_changed() -> None:
    request = rectangular_request()
    request["traversal"] = "spiral"
    try:
        plan_route(request)
    except PlanningError as error:
        assert "only serpentine" in str(error)
    else:
        raise AssertionError("unsupported traversal was accepted")
