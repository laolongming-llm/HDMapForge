"""Synthetic runtime regressions; no external API or road-accuracy claims."""

import json
from contextlib import contextmanager
from io import BytesIO

import httpx
import numpy as np
import pytest
import rasterio
import shapefile
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import LineString, Polygon, shape

from hdmapforge.crops import line_strip
from hdmapforge.pipeline import predict_raster, vectorize_prediction
from hdmapforge.segmentation import Observation, refinement_positions, split_at_observations
from hdmapforge.shapefiles import export_layer
from hdmapforge.vectorize import mask_polygons
from hdmapforge.vlm import RequestBudgetExceeded, VisionClassifier, validate_response


@pytest.fixture(autouse=True)
def isolated_runtime_environment(monkeypatch):
    # Other GIS installations may export an incompatible PROJ database globally.
    monkeypatch.delenv("PROJ_LIB", raising=False)
    monkeypatch.delenv("PROJ_DATA", raising=False)
    monkeypatch.setenv("HDMAPFORGE_TEST_KEY", "synthetic-not-a-real-secret")


@contextmanager
def memory_raster(image, *, nodata=None):
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=image.shape[1],
            height=image.shape[0],
            count=1,
            dtype="uint8",
            transform=from_origin(0, image.shape[0] * 0.1, 0.1, 0.1),
            crs="EPSG:4547",
            nodata=nodata,
        ) as writer:
            writer.write(image, 1)
        with memory.open() as reader:
            yield reader


def crop_payload(value=100, *, coverage=1.0):
    buffer = BytesIO()
    Image.new("RGB", (32, 16), (value, value, value)).save(buffer, format="PNG")
    return {
        "images": [buffer.getvalue(), buffer.getvalue()],
        "coverage": coverage,
        "decision_span_m": 1.0,
        "start_m": 0.0,
        "end_m": 12.0,
    }


def response_payload(code=1, status="ok"):
    return {
        "choices": [
            {"message": {"content": json.dumps({"jd_type": code, "status": status, "score": 0.9})}}
        ],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 30},
    }


def test_line_strip_counts_internal_nodata_and_blank_evidence():
    line = LineString([(2, 6.4), (10.8, 6.4)])
    image = np.full((128, 128), 100, dtype="uint8")
    image[:, 50:75] = 0
    with memory_raster(image, nodata=0) as dataset:
        crop = line_strip(dataset, line, 4.4, length_m=6, width_m=3, resolution_m=0.05)
        assert 0 < crop["coverage"] < 0.98
    for nodata in (None, 0):
        with memory_raster(np.zeros((128, 128), dtype="uint8"), nodata=nodata) as dataset:
            crop = line_strip(dataset, line, 4.4, length_m=6, width_m=3, resolution_m=0.05)
            assert crop["coverage"] == 0


def test_line_strip_preserves_unpainted_target_in_both_views():
    # A synthetic reference line must not paint a solid marking into the image.
    with memory_raster(np.full((128, 128), 100, dtype="uint8")) as dataset:
        crop = line_strip(
            dataset,
            LineString([(2, 6.4), (10.8, 6.4)]),
            4.4,
            length_m=6,
            width_m=3,
            resolution_m=0.05,
        )
    raw, annotated = [np.asarray(Image.open(BytesIO(data))) for data in crop["images"]]
    assert crop["coverage"] == 1
    assert np.all(raw == 100)
    middle, center = raw.shape[0] // 2, raw.shape[1] // 2
    assert np.array_equal(annotated[middle, center], raw[middle, center])
    # The locator is present off the target; the two PNGs are intentionally distinct.
    assert np.any(annotated != raw)
    assert np.count_nonzero(np.any(annotated[middle] != raw[middle], axis=1)) <= 8


def test_polygon_touching_region_boundary_requires_review():
    mask = np.zeros((50, 50), dtype=bool)
    mask[20:30, :10] = True
    mask[5:15, 20:30] = True
    features = mask_polygons(mask, from_origin(0, 5, 0.1, 0.1), envelope=True)
    assert len(features) == 2
    edge = next(item for item in features if item["geometry"].bounds[0] == 0)
    interior = next(item for item in features if item["geometry"].bounds[0] > 0)
    assert edge["review"] is True
    assert interior["review"] is False


def test_crosswalk_envelope_preserves_gap_between_l_shaped_arms():
    mask = np.zeros((30, 30), dtype=bool)
    mask[4:24, 4:9] = True
    mask[19:24, 4:24] = True
    # Small holes inside the crossing are enclosed, while its exterior concavity
    # must stay outside the exported crossing region.
    mask[10:12, 6:8] = False
    affine = from_origin(0, 3, 0.1, 0.1)
    crossing = mask_polygons(mask, affine)[0]["geometry"]
    expected = Polygon([(0.4, 2.6), (0.9, 2.6), (0.9, 1.1),
                        (2.4, 1.1), (2.4, 0.6), (0.4, 0.6)])
    assert crossing.symmetric_difference(expected).area == pytest.approx(0, abs=1e-10)
    rectangle = mask_polygons(mask, affine, envelope=True)[0]["geometry"]
    assert rectangle.area > crossing.area


def test_chainage_does_not_smooth_away_observed_short_state():
    line = LineString([(0, 0), (10, 0)])
    observations = [
        Observation(0, 1),
        Observation(4.9, 1),
        Observation(5, 3),
        Observation(5.1, 1),
        Observation(10, 1),
    ]
    pieces = split_at_observations(line, observations, tolerance_m=0.25)
    assert [item["type"] for item in pieces] == [1, 3, 1]
    assert pieces[1]["geometry"].length == pytest.approx(0.1)
    assert sum(item["geometry"].length for item in pieces) == pytest.approx(line.length)
    assert pieces[0]["geometry"].coords[-1] == pieces[1]["geometry"].coords[0]
    assert pieces[1]["geometry"].coords[-1] == pieces[2]["geometry"].coords[0]


@pytest.mark.parametrize("status", ["mixed", "unknown"])
def test_unresolved_chainage_stays_unknown_and_review_only(status):
    observations = [Observation(0, 1), Observation(5, 0, status), Observation(10, 1)]
    assert refinement_positions(observations, tolerance_m=0.25) == [2.5, 7.5]
    pieces = split_at_observations(LineString([(0, 0), (10, 0)]), observations)
    unresolved = [item for item in pieces if item["type"] == 0]
    assert unresolved and all(item["review"] for item in unresolved)
    assert sum(item["geometry"].length for item in pieces) == pytest.approx(10)


def test_api_cache_hits_do_not_consume_budget_and_survive_reopen(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert sum(c["type"] == "image_url" for c in body["messages"][1]["content"]) == 2
        return httpx.Response(200, json=response_payload())

    cache_path = tmp_path / "responses.sqlite"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        classifier = VisionClassifier(
            base_url="https://example.invalid/v1",
            model="synthetic-vlm",
            api_key_env="HDMAPFORGE_TEST_KEY",
            cache_path=cache_path,
            max_requests=1,
            client=client,
        )
        try:
            first = classifier.classify(crop_payload())
            assert classifier.classify(crop_payload()) == first
            assert classifier.calls == 1
            assert classifier.usage == {"prompt_tokens": 1000, "completion_tokens": 30}
            with pytest.raises(RequestBudgetExceeded):
                classifier.classify(crop_payload(101))
            assert len(requests) == 1
        finally:
            classifier.close()
        reopened = VisionClassifier(
            base_url="https://example.invalid/v1",
            model="synthetic-vlm",
            api_key_env="HDMAPFORGE_TEST_KEY",
            cache_path=cache_path,
            client=client,
        )
        try:
            assert reopened.classify(crop_payload()) == first
            assert reopened.calls == 0 and len(requests) == 1
        finally:
            reopened.close()


def test_invalid_api_type_is_not_cached_but_billable_usage_is_recorded(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response_payload(99))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        classifier = VisionClassifier(
            base_url="https://example.invalid/v1",
            model="synthetic-vlm",
            api_key_env="HDMAPFORGE_TEST_KEY",
            cache_path=tmp_path / "responses.sqlite",
            max_requests=2,
            client=client,
        )
        try:
            for _ in range(2):
                with pytest.raises(ValueError, match="Invalid JD type"):
                    classifier.classify(crop_payload())
            assert len(requests) == 2
            assert classifier.usage == {"prompt_tokens": 2000, "completion_tokens": 60}
            with pytest.raises(RequestBudgetExceeded):
                classifier.classify(crop_payload())
        finally:
            classifier.close()


def test_low_coverage_does_not_send_an_api_request(tmp_path):
    def forbidden_request(request):
        pytest.fail("Missing visual evidence must never invoke the external model")

    with httpx.Client(transport=httpx.MockTransport(forbidden_request)) as client:
        classifier = VisionClassifier(
            base_url="https://example.invalid/v1",
            model="synthetic-vlm",
            cache_path=tmp_path / "responses.sqlite",
            client=client,
        )
        try:
            assert classifier.classify(crop_payload(coverage=0)) == {
                "jd_type": 0,
                "status": "unknown",
                "score": 0.0,
            }
            assert classifier.calls == 0
        finally:
            classifier.close()


def test_retry_cannot_overrun_attempt_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("hdmapforge.vlm.time.sleep", lambda _: None)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(429, json={"error": "synthetic rate limit"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        classifier = VisionClassifier(
            base_url="https://example.invalid/v1",
            model="synthetic-vlm",
            api_key_env="HDMAPFORGE_TEST_KEY",
            cache_path=tmp_path / "responses.sqlite",
            max_requests=1,
            retries=2,
            client=client,
        )
        try:
            with pytest.raises(RequestBudgetExceeded):
                classifier.classify(crop_payload())
            assert classifier.calls == len(requests) == 1
        finally:
            classifier.close()


@pytest.mark.parametrize(
    "reply",
    [
        {"jd_type": True, "status": "ok", "score": 0.9},
        {"jd_type": 1, "status": "unknown", "score": 0.9},
        {"jd_type": 0, "status": "ok", "score": 0.9},
        {"jd_type": 1, "status": "ok", "score": float("nan")},
        {"jd_type": 5, "status": "ok", "score": 0.9},
    ],
)
def test_response_validation_rejects_untrusted_attributes(reply):
    with pytest.raises(ValueError):
        validate_response(reply)


def test_shapefile_layer_fields_offset_and_polygon_holes(tmp_path):
    boundary = export_layer(
        tmp_path / "BOUNDARY.shp",
        [{"geometry": LineString([(1, 2), (3, 4)]), "type": 1, "review": False}],
        geometry="line",
        crs="EPSG:4547",
        offset=(100000, 200000),
        typed=True,
    )
    with shapefile.Reader(str(boundary)) as reader:
        fields = {field.name: field for field in reader.fields[1:]}
        assert fields["TYPE"].decimal == 0
        assert type(reader.record(0).as_dict()["TYPE"]) is int
        assert reader.record(0).as_dict()["TYPE"] == 1
        assert reader.shape(0).points[0] == (100001, 200002)
    polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (1, 2), (2, 2), (2, 1)]])
    arrow = export_layer(
        tmp_path / "DIRECTION.shp",
        [{"geometry": polygon, "review": False}],
        geometry="polygon",
        crs="EPSG:4547",
    )
    with shapefile.Reader(str(arrow)) as reader:
        assert "TYPE" not in {field.name for field in reader.fields[1:]}
        assert reader.shapeType == shapefile.POLYGON
        restored = shape(reader.shape(0).__geo_interface__)
        assert restored.equals(polygon) and len(restored.interiors) == 1


@pytest.mark.parametrize("window", [None, (4, 4, 56, 56)])
def test_synthetic_raster_to_four_shapefiles_end_to_end(tmp_path, window):
    image = np.full((64, 64), 16, dtype="uint8")
    image[:2, :2] = 0
    image[10:12, 5:59] = 240  # Continuous lane centerline target.
    image[20:28, 8:16] = 180  # Arrow region.
    image[36:45, 20:42] = 120  # Whole crosswalk region, not individual stripes.
    image[52:54, 30:58] = 200  # Stopline.
    source, probability = tmp_path / "intensity.tif", tmp_path / "prediction.tif"
    affine = from_origin(100, 200, 0.1, 0.1)
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=1,
        dtype="uint8",
        transform=affine,
        crs="EPSG:4547",
        nodata=0,
    ) as writer:
        writer.write(image, 1)

    def fake_predictor(patch):
        # Intensity codes are a deterministic oracle for synthetic geometry only.
        return np.stack([np.isclose(patch, value / 255) for value in (240, 180, 120, 200)])

    predict_raster(source, probability, fake_predictor, tile_size=32, overlap=8, window=window)
    with rasterio.open(probability) as result:
        expected = affine if window is None else rasterio.windows.transform(Window(*window), affine)
        assert result.transform == expected and result.count == 4
        if window is None:
            assert np.all(result.read()[:, :2, :2] == -1)
    output = tmp_path / "shp"
    summary = vectorize_prediction(
        probability,
        source,
        output,
        output_crs="EPSG:4547",
        offset=(100000, 200000),
    )
    assert set(summary["counts"]) == {"BOUNDARY", "DIRECTION", "CROSSWALK", "STOPLINE"}
    assert all(count == 1 for count in summary["counts"].values())
    assert summary["vlm_attempts"] == 0 and summary["model_validated"] is False
    assert json.loads((output / "run.json").read_text(encoding="utf-8")) == summary
    for layer, count in summary["counts"].items():
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            assert (output / f"{layer}{suffix}").exists()
        with shapefile.Reader(str(output / f"{layer}.shp")) as reader:
            assert len(reader) == count
            assert all(record.as_dict()["REVIEW"] == 1 for record in reader.records())
            assert ("TYPE" in {field.name for field in reader.fields[1:]}) == (layer == "BOUNDARY")
            assert reader.bbox[0] > 100100 and reader.bbox[1] > 200190
            if layer == "BOUNDARY":
                assert reader.record(0).as_dict()["TYPE"] == 0
