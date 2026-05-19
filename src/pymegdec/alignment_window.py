"""Compatibility layer for alignment-window helpers now provided by NeuRepTrace."""

from reptrace.decoding.alignment_window import AlignmentWindow
from reptrace.decoding.alignment_window import WindowedFeatureSet
from reptrace.decoding.alignment_window import resolved_alignment_window
from reptrace.decoding.alignment_window import transform_with_alignment_projection
from reptrace.decoding.alignment_window import uses_separate_alignment_window
from reptrace.decoding.alignment_window import validate_paired_feature_sets

__all__ = [
    "AlignmentWindow",
    "WindowedFeatureSet",
    "resolved_alignment_window",
    "transform_with_alignment_projection",
    "uses_separate_alignment_window",
    "validate_paired_feature_sets",
]
