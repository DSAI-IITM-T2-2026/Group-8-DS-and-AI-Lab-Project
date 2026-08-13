from app.worker import safe_error_code


def test_worker_errors_are_classified_without_exposing_output():
    assert safe_error_code("Earth Engine init failed: credentials missing") == "cloud_authentication_failed"
    assert safe_error_code("CDS credentials not found. Set CDS_API_KEY") == "cds_authentication_failed"
    assert safe_error_code("Timed out waiting for Sentinel-5P") == "external_data_timeout"
    assert safe_error_code("quota exceeded 429") == "cloud_quota_exceeded"
    assert safe_error_code("unexpected stack trace") == "pipeline_failed"
