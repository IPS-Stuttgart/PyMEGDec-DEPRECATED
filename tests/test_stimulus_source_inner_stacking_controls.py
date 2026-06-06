from pymegdec.stimulus_source_inner_stacking import (
    DEFAULT_SOURCE_INNER_STACKER_WEIGHT_GRID,
    _build_parser,
    _normalize_stacker_weight_grid,
)


def test_source_inner_default_weight_grid_allows_pure_compact_fallback():
    assert DEFAULT_SOURCE_INNER_STACKER_WEIGHT_GRID[0] == 1.0
    assert 0.98 in DEFAULT_SOURCE_INNER_STACKER_WEIGHT_GRID
    assert 0.95 in DEFAULT_SOURCE_INNER_STACKER_WEIGHT_GRID


def test_normalized_stacker_weight_grid_accepts_pure_compact_weight():
    assert _normalize_stacker_weight_grid("1,0.95,0.5") == (1.0, 0.95, 0.5)


def test_source_inner_parser_exposes_latent_anti_collapse_controls():
    args = _build_parser().parse_args(
        [
            "--prediction-balance-weight", "0.03",
            "--prediction-balance-temperature", "0.05",
            "--label-smoothing", "0.05",
            "--balanced-batch-sampling",
            "--validation-prediction-balance-weight", "0.25",
        ]
    )
    assert args.prediction_balance_weight == 0.03
