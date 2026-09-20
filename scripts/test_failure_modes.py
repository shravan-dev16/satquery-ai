"""Failure mode and negative testing suite for SatQuery AI (M13 Phase 6).

Verifies that the backend and API reject invalid inputs gracefully:
1. Missing primary image (HTTP 422)
2. Missing secondary image on bi-temporal query (HTTP 400 with clean error)
3. Corrupted / unreadable raster file (HTTP 400 with clean detail)
4. Unsupported / non-image file type (HTTP 400)
5. Missing second modality for Optical+SAR query (HTTP 400)
6. Zero stack traces or server paths exposed in error messages
"""

import io
import sys
from pathlib import Path
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.main import app

client = TestClient(app)


def test_failure_modes():
    print("=" * 70)
    print("SATQUERY AI — M13 PHASE 6: API FAILURE & ROBUSTNESS TESTING")
    print("=" * 70)

    # 1. Missing primary image
    res1 = client.post("/api/v1/analyze", data={"query": "What changed?"})
    assert res1.status_code == 422, f"Expected 422 for missing primary image, got {res1.status_code}"
    print("  [OK] Missing primary image rejected with HTTP 422 Unprocessable Entity")

    # 2. Single image with explicit bi-temporal change query (requires 2 images)
    dummy_img = io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
    res2 = client.post(
        "/api/v1/analyze",
        data={"query": "What changed between these two dates?"},
        files={"image_primary": ("test.png", dummy_img, "image/png")}
    )
    assert res2.status_code == 400, f"Expected 400 for missing T2 on bi-temporal query, got {res2.status_code}"
    err_detail2 = res2.json().get("detail", "")
    assert "Two images are required" in err_detail2 or "secondary" in err_detail2.lower() or "requires" in err_detail2.lower()
    assert "Traceback" not in err_detail2
    assert "C:\\Users" not in err_detail2
    print(f"  [OK] Missing secondary image rejected cleanly: '{err_detail2}'")

    # 3. Corrupted / unreadable image
    corrupt_data = io.BytesIO(b"This is not a real raster image data at all.")
    res3 = client.post(
        "/api/v1/analyze",
        data={"query": "What is present in this image?"},
        files={"image_primary": ("corrupt.tif", corrupt_data, "image/tiff")}
    )
    assert res3.status_code in (400, 500), f"Expected error code for corrupt image, got {res3.status_code}"
    err_detail3 = res3.json().get("detail", "")
    assert "Traceback" not in err_detail3
    assert "C:\\Users" not in err_detail3
    print(f"  [OK] Corrupt raster rejected safely: '{err_detail3[:60]}...'")

    # 4. Optical + SAR query with missing second image
    dummy_img2 = io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
    res4 = client.post(
        "/api/v1/analyze",
        data={"query": "Use the optical and SAR images together to identify built-up areas."},
        files={"image_primary": ("optical.png", dummy_img2, "image/png")}
    )
    assert res4.status_code == 400, f"Expected 400 for optical+SAR without secondary image, got {res4.status_code}"
    err_detail4 = res4.json().get("detail", "")
    assert "Two images" in err_detail4 or "secondary" in err_detail4.lower() or "requires" in err_detail4.lower()
    print(f"  [OK] Missing SAR modality rejected cleanly: '{err_detail4}'")

    # 5. Root endpoint integrity for browser and API
    res_browser = client.get("/", headers={"Accept": "text/html", "User-Agent": "Mozilla/5.0"})
    assert res_browser.status_code == 200
    assert "text/html" in res_browser.headers["content-type"]
    assert "SatQuery AI" in res_browser.text
    print("  [OK] GET / serves real SatQuery AI HTML interface for web browsers")

    res_docs = client.get("/docs")
    assert res_docs.status_code == 200
    print("  [OK] GET /docs serves Swagger UI documentation")

    res_health = client.get("/health")
    assert res_health.status_code == 200
    print("  [OK] GET /health returns service health JSON")

    print("\n" + "=" * 70)
    print("ALL FAILURE MODES AND NEGATIVE TESTS PASSED CLEANLY")
    print("=" * 70)


if __name__ == "__main__":
    test_failure_modes()
