"""Inference, local-chainage VLM attribution, and four-layer SHP export."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .crops import line_strip
from .segmentation import (
    Observation,
    refinement_positions,
    sampling_positions,
    split_at_observations,
)
from .shapefiles import export_layer
from .vectorize import mask_lines, mask_polygons
from .vlm import RequestBudgetExceeded


def predict_raster(source, destination, predictor, *, tile_size=512, overlap=64, window=None):
    """Disk-backed overlap averaging, preserving source local affine exactly."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Prediction raster already exists")
    if tile_size < 16 or not 0 <= overlap < tile_size:
        raise ValueError("Invalid inference tile/overlap")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        if src.count != 1 or src.dtypes[0] != "uint8":
            raise ValueError("Baseline input must be single-channel native uint8 intensity")
        if src.transform.is_identity:
            raise ValueError("A verified local affine/world file is required")
        xoff, yoff, width, height = window or (0, 0, src.width, src.height)
        if (
            any(type(v) is not int for v in (xoff, yoff, width, height))
            or min(xoff, yoff) < 0
            or min(width, height) < 8
            or xoff + width > src.width
            or yoff + height > src.height
        ):
            raise ValueError("ROI must be an integer pixel window inside the source raster")
        affine = src.window_transform(Window(xoff, yoff, width, height))
        with tempfile.TemporaryDirectory(prefix=".predict-", dir=destination.parent) as temp:
            if not Path(temp).resolve().is_relative_to(destination.parent):
                raise ValueError("Temporary path escaped output directory")
            accum = np.memmap(
                Path(temp) / "sum.dat", mode="w+", dtype="float32", shape=(4, height, width)
            )
            weights = np.memmap(
                Path(temp) / "weight.dat", mode="w+", dtype="float32", shape=(height, width)
            )
            accum[:] = 0
            weights[:] = 0
            try:
                step = tile_size - overlap
                for row in range(0, height, step):
                    for col in range(0, width, step):
                        h, w = min(tile_size, height - row), min(tile_size, width - col)
                        read_window = Window(xoff + col, yoff + row, w, h)
                        image = src.read(1, window=read_window).astype("float32") / 255
                        if min(h, w) < 8:
                            image = np.pad(
                                image, ((0, max(0, 8 - h)), (0, max(0, 8 - w))), mode="edge"
                            )
                        prediction = np.asarray(predictor(image), dtype="float32")[:, :h, :w]
                        if (
                            prediction.shape != (4, h, w)
                            or not np.isfinite(prediction).all()
                            or (prediction < 0).any()
                            or (prediction > 1).any()
                        ):
                            raise ValueError("Predictor returned invalid probabilities")
                        weight = np.outer(np.hanning(h + 2)[1:-1], np.hanning(w + 2)[1:-1])
                        weight = np.maximum(weight, 0.01).astype("float32")
                        accum[:, row : row + h, col : col + w] += prediction * weight
                        weights[row : row + h, col : col + w] += weight
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    count=4,
                    dtype="float32",
                    nodata=-1,
                    compress="deflate",
                    BIGTIFF="IF_SAFER",
                    width=width,
                    height=height,
                    transform=affine,
                )
                with rasterio.open(destination, "w", **profile) as dst:
                    for row in range(0, height, 512):
                        h = min(512, height - row)
                        result = np.asarray(accum[:, row : row + h]) / weights[row : row + h][None]
                        valid = src.read_masks(1, window=Window(xoff, yoff + row, width, h)) > 0
                        result[:, ~valid] = -1
                        dst.write(result, window=Window(0, row, width, h))
                    dst.update_tags(
                        channels="lane,arrow,crosswalk,stopline",
                        source_window=json.dumps([xoff, yoff, width, height]),
                        coordinate_frame="source_affine_no_reprojection",
                    )
            finally:
                accum.flush()
                weights.flush()
                accum._mmap.close()
                weights._mmap.close()
    return destination


def classify_line(
    dataset,
    line,
    classifier,
    *,
    step_m=2.0,
    tolerance_m=0.25,
    max_rounds=5,
    direction_confirmed=False,
):
    observations = {}
    errors = []

    def observe(s_m, span_m=1.0):
        try:
            crop = line_strip(dataset, line, s_m, center_span_m=span_m)
            reply = classifier.classify(crop, direction_confirmed=direction_confirmed)
            return Observation(s_m, reply["jd_type"], reply["status"], reply["score"])
        except RequestBudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 -- per-window failure is explicit unknown.
            errors.append({"chainage_m": s_m, "error": type(exc).__name__})
            return Observation(s_m, 0, "unknown")

    try:
        for s in sampling_positions(line.length, step_m):
            observations[s] = observe(s)
        for level in range(max_rounds):
            new = refinement_positions(observations.values(), tolerance_m)
            if not new:
                break
            for s in new:
                observations[s] = observe(s, max(0.09, min(1.0, step_m / 2 ** (level + 2))))
    except RequestBudgetExceeded:
        # Do not assign any unobserved tail using a single earlier label.
        errors.append({"error": "request_budget_exceeded"})
        for s in sampling_positions(line.length, step_m):
            observations.setdefault(s, Observation(s, 0, "unknown"))
    pieces = split_at_observations(line, observations.values(), tolerance_m=tolerance_m)
    return pieces, {
        "observations": [o.__dict__ for o in sorted(observations.values(), key=lambda o: o.s_m)],
        "errors": errors,
        "sampling_step_m": step_m,
        "boundary_tolerance_m": tolerance_m,
        "minimum_resolvable_span_note": "Transitions shorter than sampling step may be missed",
    }


def vectorize_prediction(
    probability_raster,
    source_raster,
    output_dir,
    *,
    output_crs,
    offset=(0, 0),
    threshold=0.5,
    classifier=None,
    sampling_step_m=2.0,
    boundary_tolerance_m=0.25,
    direction_confirmed=False,
    model_validated=False,
    max_vector_pixels=16_000_000,
):
    """Single-region baseline; larger rasters require a bounded metric inference ROI.

    This release never silently downsamples or pretends tile seams are globally merged.
    """
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    if not 0 < threshold < 1 or not math.isfinite(threshold):
        raise ValueError("Probability threshold must lie strictly between 0 and 1")
    output.mkdir(parents=True, exist_ok=True)
    with rasterio.open(probability_raster) as pred, rasterio.open(source_raster) as src:
        if pred.count != 4:
            raise ValueError("Four-channel probabilities required")
        pixel_origin = (~src.transform) @ (pred.transform.c, pred.transform.f)
        col, row = (round(v) for v in pixel_origin)
        expected = src.window_transform(Window(col, row, pred.width, pred.height))
        if (
            not pred.transform.almost_equals(expected)
            or pred.crs != src.crs
            or not 0 <= col <= src.width - pred.width
            or not 0 <= row <= src.height - pred.height
        ):
            raise ValueError("Prediction/source coordinate contract mismatch")
        if pred.width * pred.height > max_vector_pixels:
            raise ValueError(
                "Vectorization memory limit exceeded: use a bounded inference ROI; global stitching is pending"
            )
        masks = pred.read() > threshold
        boundary = mask_lines(masks[0], pred.transform)
        arrow = mask_polygons(masks[1], pred.transform, envelope=True)
        crosswalk = mask_polygons(masks[2], pred.transform)
        stopline = mask_lines(masks[3], pred.transform)
        typed, traces = [], []
        for parent, item in enumerate(boundary):
            if classifier:
                pieces, trace = classify_line(
                    src,
                    item["geometry"],
                    classifier,
                    step_m=sampling_step_m,
                    tolerance_m=boundary_tolerance_m,
                    direction_confirmed=direction_confirmed,
                )
                trace["parent_id"] = parent
                traces.append(trace)
                for piece in pieces:
                    piece["review"] |= item["review"] or not model_validated
                    typed.append(piece)
            else:
                typed.append({**item, "type": 0, "review": True})
        for item in arrow + crosswalk + stopline:
            item["review"] |= not model_validated
        layers = {
            "BOUNDARY": (typed, "line", True),
            "DIRECTION": (arrow, "polygon", False),
            "CROSSWALK": (crosswalk, "polygon", False),
            "STOPLINE": (stopline, "line", False),
        }
        for name, (items, family, with_type) in layers.items():
            export_layer(
                output / f"{name}.shp",
                items,
                geometry=family,
                crs=output_crs,
                offset=offset,
                typed=with_type,
            )
    summary = {
        "model_validated": model_validated,
        "counts": {k: len(v[0]) for k, v in layers.items()},
        "vlm_attempts": classifier.calls if classifier else 0,
        "vlm_usage": classifier.usage if classifier else {},
        "line_traces": traces,
        "limitations": [
            "No global multi-tile stitching or same-class instance separation",
            "REVIEW=1 features are not accepted production output",
        ],
    }
    (output / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
