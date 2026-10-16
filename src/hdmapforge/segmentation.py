"""Attribute segmentation in line chainage; geometry never comes from VLM pixels."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from shapely.ops import substring

from .line_types import JD_TYPES


@dataclass(frozen=True)
class Observation:
    s_m: float
    jd_type: int
    status: str = "ok"
    score: float = 0.0  # Model self-report, NOT a calibrated probability.


def sampling_positions(length_m, step_m=6.0):
    if length_m <= 0 or step_m <= 0:
        raise ValueError("Positive line length and step required")
    return np.linspace(0, length_m, max(2, int(np.ceil(length_m / step_m)) + 1)).tolist()


def refinement_positions(observations, tolerance_m=0.25):
    """Query disagreeing/mixed/unknown intervals until chainage resolution is sufficient."""
    if tolerance_m <= 0:
        raise ValueError("Positive refinement tolerance required")
    positions = []
    ordered = sorted(observations, key=lambda x: x.s_m)
    for left, right in pairwise(ordered):
        if right.s_m - left.s_m > tolerance_m and (
            left.status != "ok"
            or right.status != "ok"
            or left.jd_type != right.jd_type
            or left.jd_type == 0
        ):
            positions.append((left.s_m + right.s_m) / 2)
    return positions


def split_at_observations(line, observations, *, tolerance_m=0.25):
    """Split without smoothing away short states. Unresolved brackets remain review-only."""
    if line.geom_type != "LineString" or line.is_empty or line.length <= 0:
        raise ValueError("Expected nonempty LineString")
    if tolerance_m <= 0:
        raise ValueError("Positive tolerance required")
    ordered = sorted(observations, key=lambda x: x.s_m)
    if not ordered or len({o.s_m for o in ordered}) != len(ordered):
        raise ValueError("Unique observations required")
    for o in ordered:
        if not 0 <= o.s_m <= line.length or o.jd_type not in JD_TYPES:
            raise ValueError("Invalid chainage/type")
        if o.status not in {"ok", "mixed", "unknown"}:
            raise ValueError("Invalid observation status")
    bounds = [0.0] + [(a.s_m + b.s_m) / 2 for a, b in pairwise(ordered)]
    bounds += [line.length]
    pieces = []
    for i, o in enumerate(ordered):
        start, end = bounds[i : i + 2]
        if end <= start:
            continue
        neighbors = ordered[max(0, i - 1) : min(len(ordered), i + 2)]
        uncertain_gap = any(
            abs(n.s_m - o.s_m) > tolerance_m and (n.jd_type != o.jd_type or n.status != "ok")
            for n in neighbors
        )
        review = o.status != "ok" or o.jd_type == 0 or uncertain_gap
        code = o.jd_type if o.status == "ok" else 0
        if pieces and pieces[-1]["type"] == code and pieces[-1]["review"] == review:
            pieces[-1]["end_m"] = end
            pieces[-1]["geometry"] = substring(line, pieces[-1]["start_m"], end)
        else:
            pieces.append(
                {
                    "geometry": substring(line, start, end),
                    "type": code,
                    "start_m": start,
                    "end_m": end,
                    "review": review,
                }
            )
    return pieces
