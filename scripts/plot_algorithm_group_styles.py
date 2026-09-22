"""Shared algorithm-group styling helpers for plotting scripts.

This module centralizes optional style overrides that visually group related
algorithms with marker shapes and color families.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt

# Toggle for enabling/disabling group-aware styling globally.
ENABLE_GROUP_STYLING: bool = True

# Editable algorithm grouping. Keys are group names, values are algorithm ids.
ALGORITHM_GROUPS: Dict[str, list[str]] = {
    "regularization": ["si", "ewc", "rwalk", "lwf", "ucl"],
    "replay": ["eralg4", "er_ring", "icarl"],
    "meta_learning": ["agem", "gem", "lamaml", "cmaml", "smaml"],
    "hybrid": ["la-er", "bcl_dual", "ctn"],
    "architectural": ["hat", "packnet"],
    # Reference baselines (joint/IID upper bound, naive fine-tuning lower
    # bound), not CL algorithms — always sorted last, one shared color,
    # split only by linestyle (iid2 solid, ft dashed).
    "baseline": ["iid2", "ft"],
}

# Group marker mapping used for line plots.
GROUP_MARKERS: Dict[str, str] = {
    "replay": "s",
    "regularization": "o",
    "meta_learning": "^",
    "hybrid": "D",
    "architectural": "P",
    "baseline": "*",
    "ungrouped": "X",
}

# Group linestyle mapping — adds a second visual channel on top of color and marker.
GROUP_LINESTYLES: Dict[str, Any] = {
    "regularization": "-",
    "replay": "--",
    "meta_learning": ":",
    "hybrid": "-.",
    "architectural": (0, (5, 1)),  # densely dashed
    "baseline": "-",
    "ungrouped": "-",
}

# Group colormap families used to assign related shades.
GROUP_COLOR_FAMILIES: Dict[str, str] = {
    "regularization": "Blues",
    "replay": "Purples",
    "meta_learning": "Oranges",
    "hybrid": "Greens",
    "architectural": "Greys",
    "baseline": "Greys",
    "ungrouped": "Greys",
}

# Optional per-group shade ranges to avoid overly light tones.
GROUP_SHADE_RANGES: Dict[str, tuple[float, float]] = {
    "replay": (0.55, 0.95),
}

_UNGROUPED: str = "ungrouped"

# Canonical group display order for sorting algorithm runs in plots.
# "baseline" (iid2, ft) is last so reference runs always trail the legend.
GROUP_ORDER: list[str] = [
    "regularization",
    "replay",
    "meta_learning",
    "hybrid",
    "architectural",
    "ungrouped",
    "baseline",
]


def _normalize_algorithm_name(algorithm_name: str) -> str:
    """Normalize an algorithm name for matching.

    Args:
        algorithm_name: Raw algorithm name.

    Returns:
        Normalized lowercase algorithm name.

    Usage:
        >>> _normalize_algorithm_name("A-GEM")
        'a-gem'
    """
    normalized_name = algorithm_name.strip().lower()
    # Collapse common separators so aliases like "A-GEM", "a_gem", and
    # "agem" resolve to the same canonical key.
    normalized_name = normalized_name.replace("-", "").replace("_", "").replace(" ", "")
    return normalized_name


def _build_algorithm_to_group_lookup() -> Dict[str, str]:
    """Build lookup from algorithm name to group name.

    Returns:
        Mapping of normalized algorithm names to group keys.

    Usage:
        >>> lookup = _build_algorithm_to_group_lookup()
        >>> lookup["agem"]
        'replay'
    """
    algorithm_to_group: Dict[str, str] = {}
    for group_name, algorithm_names in ALGORITHM_GROUPS.items():
        normalized_group_name = group_name.strip().lower()
        for algorithm_name in algorithm_names:
            algorithm_to_group[_normalize_algorithm_name(algorithm_name)] = (
                normalized_group_name
            )
    return algorithm_to_group


def _shade_for_group_position(position_in_group: int, group_size: int) -> float:
    """Return a stable colormap shade for an item in a group.

    Args:
        position_in_group: Zero-based position within the group.
        group_size: Number of algorithms in the group.

    Returns:
        Shade value in [0, 1] suitable for matplotlib colormaps.

    Usage:
        >>> _shade_for_group_position(0, 3)
        0.25
    """
    if group_size <= 1:
        return 0.72
    # Keep shades away from very light tones.
    return 0.35 + 0.55 * (position_in_group / float(group_size - 1))


def _group_shade_for_position(
    group_name: str, position_in_group: int, group_size: int
) -> float:
    """Resolve the colormap shade for an algorithm in a group.

    Args:
        group_name: Resolved group key.
        position_in_group: Zero-based position within group.
        group_size: Number of algorithms in group.

    Returns:
        Shade value in [0, 1] suitable for matplotlib colormaps.
    """
    shade_range = GROUP_SHADE_RANGES.get(group_name)
    if shade_range is None:
        return _shade_for_group_position(position_in_group, group_size)

    shade_min, shade_max = shade_range
    if group_size <= 1:
        return (shade_min + shade_max) / 2.0
    return shade_min + (shade_max - shade_min) * (
        position_in_group / float(group_size - 1)
    )


def _algorithm_group_position(
    algorithm_name: str, group_name: str, fallback_position: int
) -> int:
    """Resolve stable position for an algorithm within its configured group.

    Args:
        algorithm_name: Algorithm identifier.
        group_name: Resolved group key.
        fallback_position: Position used when not explicitly configured.

    Returns:
        Zero-based position within the group ordering.
    """
    declared_group_algorithms = ALGORITHM_GROUPS.get(group_name, [])
    normalized_declared_algorithms = [
        _normalize_algorithm_name(group_algorithm_name)
        for group_algorithm_name in declared_group_algorithms
    ]
    normalized_algorithm_name = _normalize_algorithm_name(algorithm_name)
    if normalized_algorithm_name in normalized_declared_algorithms:
        return normalized_declared_algorithms.index(normalized_algorithm_name)
    return len(normalized_declared_algorithms) + fallback_position


def build_group_color_map(
    algorithm_names: Iterable[str],
    fallback_colors: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build colors that preserve group-family relationships.

    Args:
        algorithm_names: Algorithms to assign colors for.
        fallback_colors: Optional default per-algorithm colors.

    Returns:
        Mapping from algorithm name to color value.

    Usage:
        >>> build_group_color_map(["agem", "gem"])
        {'agem': (...), 'gem': (...)}
    """
    fallback = dict(fallback_colors or {})
    if not ENABLE_GROUP_STYLING:
        return fallback

    algorithm_to_group = _build_algorithm_to_group_lookup()
    unique_algorithm_names = sorted(
        set(algorithm_names), key=lambda value: value.lower()
    )

    grouped_algorithms: Dict[str, list[str]] = {}
    for algorithm_name in unique_algorithm_names:
        normalized = _normalize_algorithm_name(algorithm_name)
        group_name = algorithm_to_group.get(normalized, _UNGROUPED)
        grouped_algorithms.setdefault(group_name, []).append(algorithm_name)

    resolved_colors: Dict[str, Any] = dict(fallback)
    for group_name, group_algorithms in grouped_algorithms.items():
        cmap_name = GROUP_COLOR_FAMILIES.get(
            group_name, GROUP_COLOR_FAMILIES[_UNGROUPED]
        )
        color_map = plt.get_cmap(cmap_name)
        sorted_group_algorithms = sorted(
            group_algorithms,
            key=lambda value: (
                _algorithm_group_position(value, group_name, fallback_position=0),
                _normalize_algorithm_name(value),
            ),
        )
        for group_position, algorithm_name in enumerate(sorted_group_algorithms):
            shade = _group_shade_for_position(
                group_name=group_name,
                position_in_group=group_position,
                group_size=len(sorted_group_algorithms),
            )
            resolved_colors[algorithm_name] = color_map(shade)

    return resolved_colors


def resolve_group_marker(
    algorithm_name: str, fallback_marker: str | None = None
) -> str | None:
    """Resolve marker style for an algorithm based on its group.

    Args:
        algorithm_name: Algorithm identifier.
        fallback_marker: Marker used when group styling is disabled.

    Returns:
        Marker string or ``None``.

    Usage:
        >>> resolve_group_marker("agem", fallback_marker=".")
        '.'
    """
    if not ENABLE_GROUP_STYLING:
        return fallback_marker

    algorithm_to_group = _build_algorithm_to_group_lookup()
    group_name = algorithm_to_group.get(
        _normalize_algorithm_name(algorithm_name), _UNGROUPED
    )
    return GROUP_MARKERS.get(group_name, GROUP_MARKERS[_UNGROUPED])


def resolve_group_linestyle(algorithm_name: str, fallback_linestyle: Any = "-") -> Any:
    """Resolve line style for an algorithm based on its group.

    Each group gets a distinct dash pattern so algorithms remain visually
    separable even when colors within a group are similar.

    Args:
        algorithm_name: Algorithm identifier.
        fallback_linestyle: Line style used when group styling is disabled.

    Returns:
        A matplotlib linestyle value (string or tuple).

    Usage:
        >>> resolve_group_linestyle("agem")
        '--'
        >>> resolve_group_linestyle("ewc")
        '-'
    """
    if not ENABLE_GROUP_STYLING:
        return fallback_linestyle

    algorithm_to_group = _build_algorithm_to_group_lookup()
    group_name = algorithm_to_group.get(
        _normalize_algorithm_name(algorithm_name), _UNGROUPED
    )
    return GROUP_LINESTYLES.get(group_name, "-")


def group_sort_key(algorithm_name: str) -> tuple[int, int, str]:
    """Return a sort key that orders algorithms by group then alphabetically within it.

    Use this to sort a list of runs so that all regularization algorithms appear
    first, then replay, meta-learning, hybrid, architectural, and ungrouped —
    matching the canonical ``GROUP_ORDER`` sequence.

    Args:
        algorithm_name: Algorithm identifier.

    Returns:
        Tuple ``(group_index, normalized_name)`` suitable for use as a sort key.

    Usage:
        >>> group_sort_key("ewc") < group_sort_key("agem")
        True
    """
    algorithm_to_group = _build_algorithm_to_group_lookup()
    group_name = algorithm_to_group.get(
        _normalize_algorithm_name(algorithm_name), _UNGROUPED
    )
    try:
        group_index = GROUP_ORDER.index(group_name)
    except ValueError:
        group_index = len(GROUP_ORDER)
    return (
        group_index,
        _algorithm_group_position(algorithm_name, group_name, fallback_position=0),
        _normalize_algorithm_name(algorithm_name),
    )
