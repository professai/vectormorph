import hnswlib
import numpy as np

class VectorMorph:
    def __init__(self):
        self.index = None
        self.data_labels = None

    def create_index(self, vectors, dim):
        self.index = hnswlib.Index(space='l2', dim=dim)  
        self.index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
        self.index.add_items(vectors)
        self.data_labels = np.arange(len(vectors))

    def save_index_binary(self, filename):
        if self.index is None:
            raise ValueError("Index has not been created. Use 'create_index' method first.")
        self.index.save_index(filename)

    def save_vector_binary(self, vectors, filename):
        with open(filename, 'wb') as f:
            np.save(f, vectors)

    def load_index_from_binary(self, filename, dim):
        self.index = hnswlib.Index(space='l2', dim=dim)  
        self.index.load_index(filename)

    def load_vectors_from_binary(self, filename):
        with open(filename, 'rb') as f:
            vectors = np.load(f)
        return vectors

    def query(self, vector, k=1):
        if self.index is None:
            raise ValueError("Index has not been loaded or created.")
        return self.index.knn_query(vector, k=k)
