"""Curvilinear intensity strips for runtime VLM requests; coordinates stay local."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from rasterio.windows import Window
from scipy.ndimage import map_coordinates


def line_strip(
    dataset, line, center_m, *, length_m=24.0, width_m=6.0, resolution_m=0.03, center_span_m=1.0
):
    """Return raw+annotated RGB PNGs, with central decision interval and full context.

    +x follows increasing line chainage; the top of the strip is its left side.
    The raw view is never covered by a synthetic continuous line.
    """
    if line.geom_type != "LineString" or line.length <= 0:
        raise ValueError("A positive-length LineString is required")
    if min(length_m, width_m, resolution_m, center_span_m) <= 0:
        raise ValueError("Positive strip dimensions required")
    if not 0 <= center_m <= line.length:
        raise ValueError("Chainage out of range")
    start, end = max(0, center_m - length_m / 2), min(line.length, center_m + length_m / 2)
    columns = max(2, int(np.ceil((end - start) / resolution_m)))
    rows = max(2, int(np.ceil(width_m / resolution_m)))
    if rows * columns > 4_000_000:
        raise ValueError("Strip exceeds configured pixel budget")
    chainage = np.linspace(start, end, columns)
    xy, normals = [], []
    for s in chainage:
        p = line.interpolate(s)
        before, after = (
            line.interpolate(max(0, s - 0.15)),
            line.interpolate(min(line.length, s + 0.15)),
        )
        direction = np.array([after.x - before.x, after.y - before.y])
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            raise ValueError("Degenerate local tangent")
        xy.append([p.x, p.y])
        normals.append([-direction[1] / norm, direction[0] / norm])
    points = (
        np.asarray(xy)[None]
        + np.linspace(width_m / 2, -width_m / 2, rows)[:, None, None] * np.asarray(normals)[None]
    )
    inverse = ~dataset.transform
    cols = inverse.a * points[..., 0] + inverse.b * points[..., 1] + inverse.c - 0.5
    rws = inverse.d * points[..., 0] + inverse.e * points[..., 1] + inverse.f - 0.5
    x0, x1 = int(np.floor(cols.min())) - 1, int(np.ceil(cols.max())) + 2
    y0, y1 = int(np.floor(rws.min())) - 1, int(np.ceil(rws.max())) + 2
    if (x1 - x0) * (y1 - y0) > 16_000_000:
        raise ValueError("Curved strip source window exceeds memory budget")
    window = Window(x0, y0, x1 - x0, y1 - y0)
    raw = dataset.read(1, window=window, boundless=True, fill_value=0)
    if raw.dtype != np.uint8:
        raise ValueError("This intensity-strip baseline requires native uint8 raster")
    gray = (
        map_coordinates(
            raw.astype("float32"), [rws - y0, cols - x0], order=1, mode="constant", cval=0
        )
        .clip(0, 255)
        .astype("uint8")
    )
    inside = (
        (cols >= -0.5) & (cols < dataset.width - 0.5) & (rws >= -0.5) & (rws < dataset.height - 0.5)
    )
    source_valid = dataset.read_masks(1, window=window, boundless=True)
    valid = (
        map_coordinates(source_valid, [rws - y0, cols - x0], order=0, mode="constant", cval=0) > 0
    )
    coverage = float((inside & valid).mean())
    if not gray.any():
        coverage = 0.0  # No visual evidence; this does not declare zero to be nodata.
    image = Image.fromarray(gray).convert("RGB")
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    middle = rows // 2
    for y in (middle - max(2, int(0.6 / resolution_m)), middle + max(2, int(0.6 / resolution_m))):
        draw.line([(0, y), (columns - 1, y)], fill=(255, 160, 0), width=1)
    xleft = max(0, round((center_m - center_span_m / 2 - start) / (end - start) * (columns - 1)))
    xright = min(
        columns - 1, round((center_m + center_span_m / 2 - start) / (end - start) * (columns - 1))
    )
    draw.rectangle([xleft, 0, xright, rows - 1], outline=(255, 0, 255), width=2)
    draw.line([(columns // 3, 8), (2 * columns // 3, 8)], fill=(0, 255, 255), width=2)
    draw.polygon(
        [(2 * columns // 3, 8), (2 * columns // 3 - 6, 4), (2 * columns // 3 - 6, 12)],
        fill=(0, 255, 255),
    )
    result = []
    for view in (image, annotated):
        buffer = BytesIO()
        view.save(buffer, format="PNG")
        result.append(buffer.getvalue())
    return {
        "images": result,
        "coverage": coverage,
        "start_m": start,
        "end_m": end,
        "center_m": center_m,
        "decision_span_m": center_span_m,
    }
