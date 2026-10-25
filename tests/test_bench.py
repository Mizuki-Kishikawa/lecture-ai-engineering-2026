"""第1回 宿題(発展): 計測スクリプトのテスト（LLMもAPIキーも使わない）。"""

from google.genai import errors

from app.bench import Record, load_csv, parse_rate_limit, save_csv, summarize


def test_parse_rate_limit_reads_quota_and_retry_delay():
    body = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "quotaValue": "15"}],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "51s"},
            ],
        }
    }
    quota, delay = parse_rate_limit(errors.APIError(429, body))
    assert quota == "GenerateRequestsPerMinutePerProjectPerModel-FreeTier=15"
    assert delay == 51.0


def test_csv_roundtrip_and_summary(tmp_path):
    records = [
        Record("m", "short", 1, "ok", 0.5, 1.0, 14, 50, 0, 64, 100),
        Record("m", "short", 2, "429", quota="X=15", retry_delay_s=51.0),
    ]
    path = tmp_path / "r.csv"
    save_csv(records, path)
    assert load_csv(path) == records
    assert "| m | short | 1/2 |" in summarize(records)
