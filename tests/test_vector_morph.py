import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectormorph.vector_morph import VectorDatabase, app, db

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(autouse=True)
def configure(monkeypatch, tmp_path):
    monkeypatch.setenv("BEARER_TOKEN", TOKEN)
    fresh = VectorDatabase(data_dir=str(tmp_path))
    # Swap the module-level database for a fresh, tmp-backed one per test.
    for attr in ("index", "summary_vectors", "document_vectors", "deleted", "dim", "data_dir"):
        setattr(db, attr, getattr(fresh, attr))
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
        indices, _ = d.search([0.0, 1.0], k=2)
        assert set(indices.flatten().tolist()) == {0, 1}

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
        indices, _ = d.search([1.0, 0.0], k=2)
        assert 0 not in indices.flatten().tolist()
        assert d.count == 1

    def test_search_empty_raises(self):
        d = VectorDatabase()
        with pytest.raises(LookupError):
            d.search([1.0, 0.0])

    def test_save_and_load_round_trip(self, tmp_path):
        d = VectorDatabase(data_dir=str(tmp_path / "store"))
        d.add_vector([1.0, 0.0], [1.0, 0.0])
        d.add_vector([0.0, 1.0], [0.0, 1.0])
        d.delete_vector(1)
        d.save()

        restored = VectorDatabase(data_dir=str(tmp_path / "store"))
        assert restored.load() is True
        assert restored.dim == 2
        assert restored.count == 1
        indices, _ = restored.search([1.0, 0.0], k=1)
        assert indices.flatten().tolist() == [0]

    def test_load_missing_returns_false(self, tmp_path):
        d = VectorDatabase(data_dir=str(tmp_path / "nothing"))
        assert d.load() is False


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
