import json
import os
import secrets
import signal
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional

import hnswlib
import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

__version__ = "0.2.0"

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bin")

app = FastAPI(
    title="VectorMorph",
    description="A lightweight hypervector or vector of vectors database.",
    version=__version__,
)

security = HTTPBearer()


def get_current_user(authorization: HTTPAuthorizationCredentials = Depends(security)):
    expected = os.environ.get("BEARER_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server is not configured: BEARER_TOKEN environment variable is not set.",
        )
    token = authorization.credentials
    if not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    return token


@contextmanager
def timer():
    """Context manager that measures wall-clock time of the enclosed block."""
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class VectorPair(BaseModel):
    summary_vector: List[float] = Field(
        ..., description="Vector used for indexing and nearest-neighbour lookup."
    )
    document_vector: List[float] = Field(
        ..., description="Vector stored alongside the summary vector, used for re-ranking."
    )


class SearchQuery(BaseModel):
    query_vector: List[float] = Field(..., description="Vector to search for.")
    k: int = Field(10, ge=1, description="Maximum number of results to return.")


class VectorDatabase:
    """An hnswlib-backed store of (summary_vector, document_vector) pairs.

    Summary vectors are indexed for approximate nearest-neighbour search;
    document vectors are kept alongside them for exact re-ranking.
    """

    INITIAL_CAPACITY = 1024

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self.index: Optional[hnswlib.Index] = None
        self.summary_vectors: List[List[float]] = []
        self.document_vectors: List[List[float]] = []
        self.deleted: set = set()
        self.dim: Optional[int] = None  # Dimensionality is set by the first vector
        self.lock = threading.Lock()

    # -- internal helpers -------------------------------------------------

    def _init_index(self, dim: int, max_elements: int = INITIAL_CAPACITY):
        self.dim = dim
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self.index.set_ef(200)

    def _ensure_capacity(self, needed: int):
        capacity = self.index.get_max_elements()
        if needed > capacity:
            self.index.resize_index(max(needed, capacity * 2))

    def _validate(self, summary_vector, document_vector):
        if len(summary_vector) != self.dim or len(document_vector) != self.dim:
            raise ValueError(f"Vectors must have dimensionality {self.dim}")

    # -- public API --------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self.document_vectors) - len(self.deleted)

    def add_vector(self, summary_vector, document_vector) -> int:
        with self.lock:
            if self.index is None:
                if len(summary_vector) == 0:
                    raise ValueError("Vectors must not be empty")
                self._init_index(len(summary_vector))
            self._validate(summary_vector, document_vector)

            idx = len(self.document_vectors)
            self._ensure_capacity(idx + 1)
            self.index.add_items(np.asarray([summary_vector]), np.asarray([idx]))
            self.summary_vectors.append(list(summary_vector))
            self.document_vectors.append(list(document_vector))
            return idx

    def update_vector(self, idx: int, summary_vector, document_vector) -> int:
        with self.lock:
            if self.index is None or not 0 <= idx < len(self.document_vectors):
                raise KeyError(idx)
            if idx in self.deleted:
                raise KeyError(idx)
            self._validate(summary_vector, document_vector)

            # hnswlib updates an element in place when re-adding an existing label.
            self.index.add_items(np.asarray([summary_vector]), np.asarray([idx]))
            self.summary_vectors[idx] = list(summary_vector)
            self.document_vectors[idx] = list(document_vector)
            return idx

    def delete_vector(self, idx: int):
        with self.lock:
            if self.index is None or not 0 <= idx < len(self.document_vectors):
                raise KeyError(idx)
            if idx in self.deleted:
                raise KeyError(idx)
            self.index.mark_deleted(idx)
            self.deleted.add(idx)

    def search(self, query_vector, k: int = 10):
        with self.lock:
            if self.index is None or self.count == 0:
                raise LookupError("The database is empty")
            if len(query_vector) != self.dim:
                raise ValueError(f"Vectors must have dimensionality {self.dim}")
            k = min(k, self.count)
            indices, distances = self.index.knn_query(np.asarray([query_vector]), k)
            return indices, distances

    def save(self):
        with self.lock:
            if self.index is None:
                raise LookupError("The database is empty")
            os.makedirs(self.data_dir, exist_ok=True)
            self.index.save_index(os.path.join(self.data_dir, "index.bin"))
            np.save(
                os.path.join(self.data_dir, "summary_vectors.npy"),
                np.asarray(self.summary_vectors),
            )
            np.save(
                os.path.join(self.data_dir, "document_vectors.npy"),
                np.asarray(self.document_vectors),
            )
            metadata = {"dim": self.dim, "deleted": sorted(self.deleted)}
            with open(os.path.join(self.data_dir, "metadata.json"), "w") as fh:
                json.dump(metadata, fh)

    def load(self) -> bool:
        with self.lock:
            index_path = os.path.join(self.data_dir, "index.bin")
            metadata_path = os.path.join(self.data_dir, "metadata.json")
            if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
                return False
            with open(metadata_path) as fh:
                metadata = json.load(fh)
            self.dim = metadata["dim"]
            self.deleted = set(metadata.get("deleted", []))
            self.summary_vectors = np.load(
                os.path.join(self.data_dir, "summary_vectors.npy")
            ).tolist()
            self.document_vectors = np.load(
                os.path.join(self.data_dir, "document_vectors.npy")
            ).tolist()
            self.index = hnswlib.Index(space="cosine", dim=self.dim)
            self.index.load_index(
                index_path, max_elements=max(len(self.document_vectors), self.INITIAL_CAPACITY)
            )
            self.index.set_ef(200)
            return True


db = VectorDatabase(data_dir=os.environ.get("VECTORMORPH_DATA_DIR", DEFAULT_DATA_DIR))


@app.get("/health/", tags=["Status"], summary="Health Check", description="Liveness probe; requires no authentication.")
async def health():
    return {"status": "ok", "version": __version__, "timestamp": utc_timestamp()}


@app.get("/stats/", tags=["Status"], summary="Database Stats", description="Return the number of stored vectors and their dimensionality.")
async def stats(user: str = Depends(get_current_user)):
    return {
        "count": db.count,
        "dim": db.dim,
        "timestamp": utc_timestamp(),
    }


@app.post("/add/", tags=["CRUD Operations"], summary="Add Vector", description="Add summary and document vectors to the database.")
async def add_vector(body: VectorPair, user: str = Depends(get_current_user)):
    with timer() as t:
        try:
            idx = db.add_vector(body.summary_vector, body.document_vector)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return {"index": idx, "timestamp": utc_timestamp(), "execution_time": t["elapsed"]}


@app.put("/update/{idx}", tags=["CRUD Operations"], summary="Update Vector", description="Update summary and document vectors at a specific index.")
async def update_vector(idx: int, body: VectorPair, user: str = Depends(get_current_user)):
    with timer() as t:
        try:
            updated_idx = db.update_vector(idx, body.summary_vector, body.document_vector)
        except KeyError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No vector at index {idx}")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return {"index": updated_idx, "timestamp": utc_timestamp(), "execution_time": t["elapsed"]}


@app.delete("/delete/{idx}", tags=["CRUD Operations"], summary="Delete Vector", description="Delete the vector at a specific index.")
async def delete_vector(idx: int, user: str = Depends(get_current_user)):
    with timer() as t:
        try:
            db.delete_vector(idx)
        except KeyError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No vector at index {idx}")
    return {
        "message": f"Vector at index {idx} has been deleted.",
        "timestamp": utc_timestamp(),
        "execution_time": t["elapsed"],
    }


@app.post("/search/", tags=["Search"], summary="Search Vectors", description="Search the database for similar vectors. Results are re-ranked by document-vector similarity.")
async def search(body: SearchQuery, user: str = Depends(get_current_user)):
    with timer() as t:
        try:
            indices, distances = db.search(body.query_vector, body.k)
        except LookupError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The database is empty")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

        query = np.asarray(body.query_vector)
        results = []
        for i, dist in zip(indices.flatten(), distances.flatten()):
            similarity = float(np.dot(db.document_vectors[int(i)], query))
            results.append(
                {"index": int(i), "similarity": similarity, "summary_distance": float(dist)}
            )
        results.sort(key=lambda r: -r["similarity"])
    return {"results": results, "timestamp": utc_timestamp(), "execution_time": t["elapsed"]}


@app.post("/save/", tags=["Persistence"], summary="Save Database", description="Persist the index and vectors to disk.")
async def save_database(user: str = Depends(get_current_user)):
    with timer() as t:
        try:
            db.save()
        except LookupError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The database is empty")
    return {"message": "Database saved.", "timestamp": utc_timestamp(), "execution_time": t["elapsed"]}


@app.post("/load/", tags=["Persistence"], summary="Load Database", description="Load a previously saved index and vectors from disk.")
async def load_database(user: str = Depends(get_current_user)):
    with timer() as t:
        loaded = db.load()
    if not loaded:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No saved database found")
    return {"message": "Database loaded.", "count": db.count, "timestamp": utc_timestamp(), "execution_time": t["elapsed"]}


def shutdown():
    os.kill(os.getpid(), signal.SIGTERM)


def reboot():
    os.execv(sys.executable, [sys.executable] + sys.argv)


@app.post("/shutdown/", tags=["Server Control"], summary="Shutdown Server", description="Shutdown the VectorMorph server.")
async def shutdown_server(background_tasks: BackgroundTasks, user: str = Depends(get_current_user)):
    background_tasks.add_task(shutdown)
    return {"message": "Shutting down..."}


@app.post("/reboot/", tags=["Server Control"], summary="Reboot Server", description="Reboot the VectorMorph server.")
async def reboot_server(background_tasks: BackgroundTasks, user: str = Depends(get_current_user)):
    background_tasks.add_task(reboot)
    return {"message": "Rebooting..."}


def main():
    import uvicorn

    host = os.environ.get("VECTORMORPH_HOST", "0.0.0.0")
    port = int(os.environ.get("VECTORMORPH_PORT", "4440"))
    uvicorn.run("vectormorph.vector_morph:app", host=host, port=port)


if __name__ == "__main__":
    main()
