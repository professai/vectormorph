import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectormorph.loopguard import LoopGuard
from vectormorph.vector_morph import VectorDatabase, app, db, loop_guard, metrics

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(autouse=True)
def configure(monkeypatch, tmp_path):
    monkeypatch.setenv("BEARER_TOKEN", TOKEN)
    fresh = VectorDatabase(data_dir=str(tmp_path))
    # Swap the module-level database for a fresh, tmp-backed one per test.
    for attr in ("index", "summary_vectors", "document_vectors", "metadata", "deleted", "dim", "data_dir"):
        setattr(db, attr, getattr(fresh, attr))
    metrics.reset()
    loop_guard.reset()
    yield


@pytest.fixture
def client():
    return TestClient(app)


def vec(*values):
    return list(map(float, values))


def add(client, summary, document):
    return client.post(
        "/add/", json={"summary_vector": summary, "document_vector": document}, headers=AUTH
    )


class TestVectorDatabase:
    def test_add_returns_sequential_indices(self):
        d = VectorDatabase()
        assert d.add_vector([1.0, 0.0], [1.0, 0.0]) == 0
        assert d.add_vector([0.0, 1.0], [0.0, 1.0]) == 1
        assert d.count == 2

    def test_add_rejects_dimension_mismatch(self):
        d = VectorDatabase()
        d.add_vector([1.0, 0.0], [1.0, 0.0])
        with pytest.raises(ValueError):
            d.add_vector([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    def test_add_rejects_empty_vector(self):
        d = VectorDatabase()
        with pytest.raises(ValueError):
            d.add_vector([], [])

    def test_index_grows_beyond_initial_capacity(self):
        d = VectorDatabase()
        d.INITIAL_CAPACITY = 4
        rng = np.random.default_rng(0)
        for _ in range(10):
            v = rng.random(3).tolist()
            d.add_vector(v, v)
        assert d.count == 10

    def test_update_replaces_vector(self):
        d = VectorDatabase()
        d.add_vector([1.0, 0.0], [1.0, 0.0])
        d.add_vector([0.0, 1.0], [0.0, 1.0])
        d.update_vector(0, [0.0, 1.0], [0.0, 1.0])
        results = d.search([0.0, 1.0], k=2)
        assert {r["index"] for r in results} == {0, 1}

    def test_update_missing_index_raises(self):
        d = VectorDatabase()
        d.add_vector([1.0, 0.0], [1.0, 0.0])
        with pytest.raises(KeyError):
            d.update_vector(5, [1.0, 0.0], [1.0, 0.0])

    def test_delete_removes_from_search(self):
        d = VectorDatabase()
        d.add_vector([1.0, 0.0], [1.0, 0.0])
        d.add_vector([0.0, 1.0], [0.0, 1.0])
        d.delete_vector(0)
        results = d.search([1.0, 0.0], k=2)
        assert 0 not in [r["index"] for r in results]
        assert d.count == 1

    def test_search_empty_raises(self):
        d = VectorDatabase()
        with pytest.raises(LookupError):
            d.search([1.0, 0.0])

    def test_save_and_load_round_trip(self, tmp_path):
        d = VectorDatabase(data_dir=str(tmp_path / "store"))
        d.add_vector([1.0, 0.0], [1.0, 0.0], metadata={"title": "first"})
        d.add_vector([0.0, 1.0], [0.0, 1.0])
        d.delete_vector(1)
        d.save()

        restored = VectorDatabase(data_dir=str(tmp_path / "store"))
        assert restored.load() is True
        assert restored.dim == 2
        assert restored.count == 1
        assert restored.metadata[0] == {"title": "first"}
        results = restored.search([1.0, 0.0], k=1)
        assert [r["index"] for r in results] == [0]

    def test_save_leaves_no_temp_files(self, tmp_path):
        d = VectorDatabase(data_dir=str(tmp_path / "store"))
        d.add_vector([1.0, 0.0], [1.0, 0.0])
        d.save()
        leftovers = [f for f in os.listdir(tmp_path / "store") if f.endswith(".tmp")]
        assert leftovers == []
        assert sorted(os.listdir(tmp_path / "store")) == [
            "document_vectors.npy", "index.bin", "metadata.json", "summary_vectors.npy",
        ]

    def test_info_snapshot(self):
        d = VectorDatabase()
        assert d.info() == {"count": 0, "total_slots": 0, "deleted": 0, "dim": None, "capacity": 0}
        d.add_vector([1.0, 0.0], [1.0, 0.0])
        d.add_vector([0.0, 1.0], [0.0, 1.0])
        d.delete_vector(1)
        info = d.info()
        assert info["count"] == 1
        assert info["total_slots"] == 2
        assert info["deleted"] == 1
        assert info["dim"] == 2
        assert info["capacity"] >= 2

    def test_add_batch(self):
        d = VectorDatabase()
        indices = d.add_vectors(
            [
                ([1.0, 0.0], [1.0, 0.0], {"n": 1}),
                ([0.0, 1.0], [0.0, 1.0], None),
            ]
        )
        assert indices == [0, 1]
        assert d.count == 2
        assert d.metadata == [{"n": 1}, None]

    def test_get_vector(self):
        d = VectorDatabase()
        d.add_vector([1.0, 0.0], [0.5, 0.5], metadata={"title": "doc"})
        summary, document, metadata = d.get_vector(0)
        assert summary == [1.0, 0.0]
        assert document == [0.5, 0.5]
        assert metadata == {"title": "doc"}
        with pytest.raises(KeyError):
            d.get_vector(1)

    def test_load_missing_returns_false(self, tmp_path):
        d = VectorDatabase(data_dir=str(tmp_path / "nothing"))
        assert d.load() is False


class TestLoopGuard:
    KEY = ("1.2.3.4", "GET", "/get/999")

    def test_blocks_after_repeated_failures(self):
        guard = LoopGuard(threshold=3, window_seconds=10, cooldown_seconds=30)
        for i in range(4):
            assert guard.retry_after(self.KEY, now=float(i)) == 0
            guard.record(self.KEY, 404, now=float(i))
        # Block was set at t=3 for 30s, so 29s remain at t=4
        assert guard.retry_after(self.KEY, now=4.0) == pytest.approx(29.0)
        # Block expires after the cooldown
        assert guard.retry_after(self.KEY, now=35.0) == 0

    def test_success_resets_failures(self):
        guard = LoopGuard(threshold=3, window_seconds=10, cooldown_seconds=30)
        for i in range(3):
            guard.record(self.KEY, 404, now=float(i))
        guard.record(self.KEY, 200, now=3.0)
        guard.record(self.KEY, 404, now=4.0)
        assert guard.retry_after(self.KEY, now=5.0) == 0

    def test_failures_outside_window_ignored(self):
        guard = LoopGuard(threshold=3, window_seconds=10, cooldown_seconds=30)
        for i in range(3):
            guard.record(self.KEY, 404, now=float(i))
        guard.record(self.KEY, 404, now=100.0)  # earlier failures aged out
        assert guard.retry_after(self.KEY, now=101.0) == 0

    def test_keys_are_independent(self):
        guard = LoopGuard(threshold=2, window_seconds=10, cooldown_seconds=30)
        other = ("5.6.7.8", "GET", "/get/999")
        for i in range(3):
            guard.record(self.KEY, 404, now=float(i))
        assert guard.retry_after(self.KEY, now=3.0) > 0
        assert guard.retry_after(other, now=3.0) == 0

    def test_disabled_guard_never_blocks(self):
        guard = LoopGuard(enabled=False, threshold=1, window_seconds=10, cooldown_seconds=30)
        for i in range(10):
            guard.record(self.KEY, 404, now=float(i))
        assert guard.retry_after(self.KEY, now=10.0) == 0

    def test_snapshot(self):
        guard = LoopGuard(threshold=2, window_seconds=10, cooldown_seconds=30)
        for i in range(3):
            guard.record(self.KEY, 404, now=float(i))
        guard.retry_after(self.KEY, now=3.0)  # one blocked request
        snap = guard.snapshot(now=3.0)
        assert snap["active_blocks"] == 1
        assert snap["loops_detected_total"] == 1
        assert snap["requests_blocked_total"] == 1
        assert snap["active"][0]["path"] == "/get/999"


class TestAPI:
    def test_health_requires_no_auth(self, client):
        response = client.get("/health/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_missing_token_rejected(self, client):
        response = client.post(
            "/add/", json={"summary_vector": [1.0], "document_vector": [1.0]}
        )
        # FastAPI's HTTPBearer returns 403 in older releases, 401 in newer ones.
        assert response.status_code in (401, 403)

    def test_wrong_token_rejected(self, client):
        response = client.post(
            "/add/",
            json={"summary_vector": [1.0], "document_vector": [1.0]},
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401

    def test_add_and_search(self, client):
        assert add(client, vec(1, 0, 0), vec(1, 0, 0)).status_code == 200
        assert add(client, vec(0, 1, 0), vec(0, 1, 0)).status_code == 200

        response = client.post(
            "/search/", json={"query_vector": vec(1, 0, 0), "k": 2}, headers=AUTH
        )
        assert response.status_code == 200
        body = response.json()
        assert body["results"][0]["index"] == 0
        assert body["results"][0]["similarity"] == pytest.approx(1.0)
        assert body["execution_time"] > 0

    def test_add_dimension_mismatch_is_422(self, client):
        add(client, vec(1, 0), vec(1, 0))
        response = add(client, vec(1, 0, 0), vec(1, 0, 0))
        assert response.status_code == 422

    def test_search_empty_database_is_404(self, client):
        response = client.post(
            "/search/", json={"query_vector": vec(1, 0)}, headers=AUTH
        )
        assert response.status_code == 404

    def test_update_and_delete(self, client):
        add(client, vec(1, 0), vec(1, 0))
        response = client.put(
            "/update/0",
            json={"summary_vector": vec(0, 1), "document_vector": vec(0, 1)},
            headers=AUTH,
        )
        assert response.status_code == 200

        response = client.delete("/delete/0", headers=AUTH)
        assert response.status_code == 200

        response = client.delete("/delete/0", headers=AUTH)
        assert response.status_code == 404

    def test_update_missing_index_is_404(self, client):
        add(client, vec(1, 0), vec(1, 0))
        response = client.put(
            "/update/9",
            json={"summary_vector": vec(0, 1), "document_vector": vec(0, 1)},
            headers=AUTH,
        )
        assert response.status_code == 404

    def test_stats(self, client):
        add(client, vec(1, 0), vec(1, 0))
        response = client.get("/stats/", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["count"] == 1
        assert response.json()["dim"] == 2

    def test_save_and_load_endpoints(self, client):
        add(client, vec(1, 0), vec(1, 0))
        assert client.post("/save/", headers=AUTH).status_code == 200
        assert client.post("/load/", headers=AUTH).status_code == 200

    def test_load_without_saved_data_is_404(self, client):
        assert client.post("/load/", headers=AUTH).status_code == 404

    def test_metadata_round_trips_through_search(self, client):
        client.post(
            "/add/",
            json={
                "summary_vector": vec(1, 0),
                "document_vector": vec(1, 0),
                "metadata": {"title": "hello", "tags": ["a", "b"]},
            },
            headers=AUTH,
        )
        response = client.post(
            "/search/", json={"query_vector": vec(1, 0), "k": 1}, headers=AUTH
        )
        assert response.status_code == 200
        assert response.json()["results"][0]["metadata"] == {"title": "hello", "tags": ["a", "b"]}

    def test_add_batch_endpoint(self, client):
        response = client.post(
            "/add_batch/",
            json={
                "items": [
                    {"summary_vector": vec(1, 0), "document_vector": vec(1, 0)},
                    {"summary_vector": vec(0, 1), "document_vector": vec(0, 1), "metadata": {"n": 2}},
                ]
            },
            headers=AUTH,
        )
        assert response.status_code == 200
        assert response.json()["indices"] == [0, 1]

    def test_add_batch_empty_is_422(self, client):
        response = client.post("/add_batch/", json={"items": []}, headers=AUTH)
        assert response.status_code == 422

    def test_get_endpoint(self, client):
        add(client, vec(1, 0), vec(0.5, 0.5))
        response = client.get("/get/0", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert body["summary_vector"] == [1.0, 0.0]
        assert body["document_vector"] == [0.5, 0.5]
        assert body["metadata"] is None

        assert client.get("/get/7", headers=AUTH).status_code == 404

    def test_metrics_endpoint(self, client):
        add(client, vec(1, 0), vec(1, 0))
        client.post("/search/", json={"query_vector": vec(1, 0)}, headers=AUTH)
        response = client.get("/metrics/", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert body["uptime_seconds"] >= 0
        assert body["totals"]["requests"] >= 2
        assert body["database"]["count"] == 1
        assert "POST /add/" in body["endpoints"]
        add_stats = body["endpoints"]["POST /add/"]
        assert add_stats["count"] >= 1
        assert add_stats["p95_ms"] >= 0

    def test_metrics_history(self, client):
        add(client, vec(1, 0), vec(1, 0))
        client.post("/search/", json={"query_vector": vec(1, 0)}, headers=AUTH)
        body = client.get("/metrics/", headers=AUTH).json()
        history = body["history"]
        assert len(history) == 120
        assert body["bucket_seconds"] == 5
        # Timestamps are contiguous 5-second buckets, oldest first
        assert all(b["t"] - a["t"] == 5 for a, b in zip(history, history[1:]))
        # The traffic we just generated lands in the newest bucket(s)
        recent = history[-2:]
        assert sum(b["requests"] for b in recent) >= 2
        active = [b for b in recent if b["requests"]]
        assert all(b["avg_ms"] > 0 and b["p95_ms"] >= b["avg_ms"] * 0.5 for b in active)
        # Empty buckets have null latency, not zero
        assert history[0]["requests"] == 0 and history[0]["avg_ms"] is None

    def test_metrics_requires_auth(self, client):
        response = client.get("/metrics/", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401

    def test_metrics_counts_errors(self, client):
        client.post("/search/", json={"query_vector": vec(1, 0)}, headers=AUTH)  # 404: empty db
        body = client.get("/metrics/", headers=AUTH).json()
        assert body["endpoints"]["POST /search/"]["errors_4xx"] == 1

    def test_doom_loop_gets_429_with_retry_after(self, client):
        # Hammer the same failing request past the threshold
        responses = [client.get("/get/999", headers=AUTH) for _ in range(loop_guard.threshold + 2)]
        assert responses[0].status_code == 404
        blocked = [r for r in responses if r.status_code == 429]
        assert blocked, "expected the loop to be blocked eventually"
        assert int(blocked[0].headers["Retry-After"]) >= 1
        assert "Doom loop" in blocked[0].json()["detail"]

        body = client.get("/metrics/", headers=AUTH).json()
        guard = body["loop_guard"]
        assert guard["enabled"] is True
        assert guard["loops_detected_total"] == 1
        assert guard["active_blocks"] == 1
        assert guard["active"][0]["path"] == "/get/999"

    def test_successful_requests_never_blocked(self, client):
        for _ in range(loop_guard.threshold + 5):
            response = client.get("/health/")
            assert response.status_code == 200
        for _ in range(loop_guard.threshold + 5):
            response = client.get("/stats/", headers=AUTH)
            assert response.status_code == 200

    def test_health_exempt_from_loop_guard(self, client):
        # Even unauthenticated failures on /health/ would be exempt; simulate
        # heavy failing traffic on a guarded route, then confirm /health/ still works
        for _ in range(loop_guard.threshold + 2):
            client.get("/get/999", headers=AUTH)
        assert client.get("/health/").status_code == 200

    def test_dashboard_served_without_auth(self, client):
        response = client.get("/dashboard/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "VectorMorph" in response.text

    def test_autoload_on_startup(self, tmp_path, monkeypatch):
        db.add_vector([1.0, 0.0], [1.0, 0.0], metadata={"title": "persisted"})
        db.save()

        # Simulate a fresh process: clear in-memory state, then run the
        # lifespan (which auto-loads from db.data_dir).
        db.index = None
        db.summary_vectors, db.document_vectors, db.metadata = [], [], []
        db.deleted, db.dim = set(), None

        monkeypatch.setenv("VECTORMORPH_AUTOLOAD", "1")
        with TestClient(app) as client:
            response = client.get("/stats/", headers=AUTH)
            assert response.json()["count"] == 1
            assert db.metadata[0] == {"title": "persisted"}
