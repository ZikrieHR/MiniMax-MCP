from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

from minimax_mcp.client import MinimaxAPIClient


class StubResponse:
    headers = {}

    def raise_for_status(self):
        pass

    def json(self):
        return {"base_resp": {"status_code": 0}}


def test_concurrent_json_and_multipart_requests_use_local_headers(monkeypatch):
    client = MinimaxAPIClient("test-api-key", "https://api.example.invalid")
    request_barrier = Barrier(2)
    observations = []
    observations_lock = Lock()

    def fake_request(method, url, **kwargs):
        with observations_lock:
            observations.append(
                {
                    "url": url,
                    "headers": kwargs["headers"].copy(),
                    "has_files": bool(kwargs.get("files")),
                }
            )
        request_barrier.wait(timeout=5)
        return StubResponse()

    monkeypatch.setattr(client.session, "request", fake_request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        json_request = executor.submit(
            client.post,
            "/v1/json",
            json={"prompt": "test"},
        )
        multipart_request = executor.submit(
            client.post,
            "/v1/files/upload",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={"purpose": "voice_clone"},
            headers={"content-type": "application/json"},
        )

        assert json_request.result()["base_resp"]["status_code"] == 0
        assert multipart_request.result()["base_resp"]["status_code"] == 0

    json_observation = next(item for item in observations if not item["has_files"])
    multipart_observation = next(item for item in observations if item["has_files"])

    assert json_observation["headers"]["Content-Type"] == "application/json"
    assert "Content-Type" not in multipart_observation["headers"]
    assert json_observation["headers"]["Authorization"] == "Bearer test-api-key"
    assert multipart_observation["headers"]["Authorization"] == "Bearer test-api-key"
    assert "Content-Type" not in client.session.headers
