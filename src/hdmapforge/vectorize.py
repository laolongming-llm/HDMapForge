"""Baseline probability-to-vector conversion; junctions and seams remain explicit."""

import numpy as np
from rasterio.features import shapes
from shapely import make_valid
from shapely.geometry import LineString, Polygon, shape
from skimage.morphology import skeletonize


def mask_lines(mask, affine, minimum_m=0.5):
    skeleton = skeletonize(np.asarray(mask, dtype=bool))
    points = {tuple(map(int, p)) for p in np.argwhere(skeleton)}
    adjacency = {}
    for row, col in points:
        neighbors = []
        for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            neighbor = row + dr, col + dc
            if neighbor not in points:
                continue
            # Avoid diagonal shortcuts at right-angle stair steps.
            if dr and dc and ((row + dr, col) in points or (row, col + dc) in points):
                continue
            neighbors.append(neighbor)
        adjacency[row, col] = neighbors
    visited, results = set(), []

    def edge(a, b):
        return tuple(sorted((a, b)))

    starts = [p for p in sorted(points) if len(adjacency[p]) != 2]
    starts += [p for p in sorted(points) if len(adjacency[p]) == 2]
    for start in starts:
        for neighbor in adjacency[start]:
            if edge(start, neighbor) in visited:
                continue
            path = [start, neighbor]
            visited.add(edge(start, neighbor))
            while len(adjacency[path[-1]]) == 2:
                candidates = [p for p in adjacency[path[-1]] if p != path[-2]]
                nxt = candidates[0]
                if edge(path[-1], nxt) in visited:
                    break
                visited.add(edge(path[-1], nxt))
                path.append(nxt)
                if nxt == start:
                    break
            coords = [affine @ (col + 0.5, row + 0.5) for row, col in path]
            geometry = LineString(coords)
            if geometry.length >= minimum_m:
                touched_seam = any(
                    r in {0, mask.shape[0] - 1} or c in {0, mask.shape[1] - 1} for r, c in path
                )
                junction = len(adjacency[path[0]]) > 2 or len(adjacency[path[-1]]) > 2
                results.append(
                    {"geometry": geometry, "review": touched_seam or junction, "seam": touched_seam}
                )
    return results


def mask_polygons(mask, affine, *, envelope=False, minimum_area_m2=0.1):
    result = []
    frame = Polygon(
        [
            affine @ p
            for p in [
                (0, 0),
                (mask.shape[1], 0),
                (mask.shape[1], mask.shape[0]),
                (0, mask.shape[0]),
            ]
        ]
    )
    for geojson, value in shapes(
        np.asarray(mask, dtype="uint8"), mask=np.asarray(mask, dtype=bool), transform=affine
    ):
        if value != 1:
            continue
        geometry = make_valid(shape(geojson))
        parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        for part in parts:
            if part.geom_type != "Polygon" or part.area < minimum_area_m2:
                continue
            truncated = part.boundary.intersects(frame.boundary)
            # Arrow instances use a rectangle. Crosswalks keep their enclosing
            # region's concavities, including the gap between an L shape's arms.
            part = part.minimum_rotated_rectangle if envelope else Polygon(part.exterior)
            result.append({"geometry": part, "review": truncated})
    return result
