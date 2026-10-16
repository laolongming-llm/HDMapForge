"""JD-compatible layer families; arrow/stop/crosswalk have no type classification."""

import math
import shutil
import tempfile
from pathlib import Path

import shapefile
from pyproj import CRS
from shapely.affinity import translate
from shapely.geometry.polygon import orient

from .line_types import JD_TYPES


def export_layer(path, features, *, geometry, crs, offset=(0, 0), typed=False):
    path = Path(path).resolve().with_suffix(".shp")
    crs = CRS.from_user_input(crs)
    if not crs.is_projected or not all(a.unit_name == "metre" for a in crs.axis_info):
        raise ValueError("Export requires a verified projected metric CRS")
    if len(offset) != 2 or not all(math.isfinite(v) for v in offset):
        raise ValueError("A finite metric offset is required")
    features = list(features)
    family = {"line": "LineString", "polygon": "Polygon"}.get(geometry)
    if family is None:
        raise ValueError("geometry must be line or polygon")
    for f in features:
        g = f["geometry"]
        if g.geom_type != family or g.is_empty or not g.is_valid:
            raise ValueError("Invalid SHP output geometry")
        if typed and (type(f.get("type")) is not int or f["type"] not in JD_TYPES):
            raise ValueError("A valid integer JD TYPE is required")
    suffixes = (".shp", ".shx", ".dbf", ".prj", ".cpg")
    if any(path.with_suffix(s).exists() for s in suffixes):
        raise ValueError("Refusing to overwrite an existing SHP family")
    path.parent.mkdir(parents=True, exist_ok=True)
    published = []
    with tempfile.TemporaryDirectory(prefix=".shp-", dir=path.parent) as directory:
        temporary = Path(directory).resolve()
        if not temporary.is_relative_to(path.parent):
            raise ValueError("Staging directory escaped output root")
        staging = temporary / path.name
        with shapefile.Writer(
            str(staging),
            shapeType=shapefile.POLYLINE if geometry == "line" else shapefile.POLYGON,
            encoding="utf-8",
        ) as writer:
            writer.field("ID", "N", 18, 0)
            if typed:
                writer.field("TYPE", "N", 2, 0)
            writer.field("REVIEW", "N", 1, 0)
            for i, feature in enumerate(features):
                geom = translate(feature["geometry"], xoff=offset[0], yoff=offset[1])
                if geometry == "line":
                    writer.line([list(geom.coords)])
                else:
                    geom = orient(geom, sign=-1)
                    writer.poly(
                        [list(geom.exterior.coords)] + [list(r.coords) for r in geom.interiors]
                    )
                record = [i + 1]
                if typed:
                    record.append(feature["type"])
                writer.record(*record, int(bool(feature.get("review", False))))
        staging.with_suffix(".prj").write_text(crs.to_wkt(version="WKT1_ESRI"), encoding="utf-8")
        staging.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
        try:
            for suffix in (".cpg", ".prj", ".dbf", ".shx", ".shp"):
                dest = path.with_suffix(suffix)
                with dest.open("xb") as out:
                    published.append(dest)
                    with staging.with_suffix(suffix).open("rb") as src:
                        shutil.copyfileobj(src, out)
        except Exception:
            for dest in published:
                dest.unlink(missing_ok=True)
            raise
    return path
