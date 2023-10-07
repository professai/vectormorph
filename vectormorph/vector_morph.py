import hnswlib
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import numpy as np
import os
from typing import List
import time
from datetime import datetime

app = FastAPI(
    title="VectorMorph",
    description="A lightweight hypervector or vector of vectors database.",
    version="0.1.1",
)

security = HTTPBearer()

def get_current_user(authorization: HTTPAuthorizationCredentials = Depends(security)):
    token = authorization.credentials
    if token != os.environ.get("BEARER_TOKEN"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return token

def timer_dependency():
    start_time = time.time()
    yield
    execution_time = time.time() - start_time
    yield execution_time

class VectorDatabase:
    def __init__(self, dim, space='cosine'):
        self.space = space
        self.dim = dim
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=10000, ef_construction=200, M=16)
        self.summary_vectors = []
        self.document_vectors = []
        self.load()
    
    def add_vector(self, summary_vector, document_vector):
        idx = len(self.summary_vectors)
        self.index.add_items(summary_vector, idx)
        self.summary_vectors.append(summary_vector)
        self.document_vectors.append(document_vector)
        return idx

    def update_vector(self, idx, summary_vector, document_vector):
        self.summary_vectors[idx] = summary_vector
        self.document_vectors[idx] = document_vector
        self.index.set_ef(200)  # Setting ef to a higher value for more accurate search
        self.index.knn_query(summary_vector, 1)  # Query to update the index
        return idx

    def delete_vector(self, idx):
        # hnswlib doesn't support item deletion so we'll just zero out the vectors
        zero_vector = np.zeros((1, self.dim))
        self.update_vector(idx, zero_vector, zero_vector)

    def search(self, query_vector, k=10):
        indices, distances = self.index.knn_query(query_vector, k)
        return indices, distances

    def save(self):
        self.index.save_index('bin/index.bin')
        np.save('bin/summary_vectors.bin', self.summary_vectors)
        np.save('bin/document_vectors.bin', self.document_vectors)

    def load(self):
        if os.path.exists('bin/index.bin'):
            self.index.load_index('bin/index.bin')
            self.summary_vectors = np.load('bin/summary_vectors.bin').tolist()
            self.document_vectors = np.load('bin/document_vectors.bin').tolist()


db = VectorDatabase(dim=128)

@app.post("/add/", tags=["CRUD Operations"], summary="Add Vector", description="Add summary and document vectors to the database.")
async def add_vector(
        summary_vector: List[float],
        document_vector: List[float],
        user: str = Depends(get_current_user),
        execution_time: float = Depends(timer_dependency)):
    idx = db.add_vector(np.array([summary_vector]), np.array([document_vector]))
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    return {"index": idx, "timestamp": timestamp, "execution_time": execution_time}

@app.put("/update/{idx}", tags=["CRUD Operations"], summary="Update Vector", description="Update summary and document vectors at a specific index.")
async def update_vector(
        idx: int,
        summary_vector: List[float],
        document_vector: List[float],
        user: str = Depends(get_current_user),
        execution_time: float = Depends(timer_dependency)):
    updated_idx = db.update_vector(idx, np.array([summary_vector]), np.array([document_vector]))
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    return {"index": updated_idx, "timestamp": timestamp, "execution_time": execution_time}

@app.delete("/delete/{idx}", tags=["CRUD Operations"], summary="Delete Vector", description="Delete the vector at a specific index.")
async def delete_vector(
        idx: int,
        user: str = Depends(get_current_user),
        execution_time: float = Depends(timer_dependency)):
    db.delete_vector(idx)
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    return {"message": f"Vector at index {idx} has been deleted.", "timestamp": timestamp, "execution_time": execution_time}

@app.get("/search/", tags=["Search"], summary="Search Vectors", description="Search the database for similar vectors.")
async def search(
        query_vector: List[float],
        k: int = 10,
        user: str = Depends(get_current_user),
        execution_time: float = Depends(timer_dependency)):
    indices, distances = db.search(np.array([query_vector]), k)
    detailed_results = [(i, np.dot(db.document_vectors[i], np.array(query_vector))) for i in indices.flatten()]
    detailed_results.sort(key=lambda x: -x[1])  # Sort by similarity in descending order
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    return {"results": detailed_results, "timestamp": timestamp, "execution_time": execution_time}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4440)

if __name__ == "__main__":
    main()

