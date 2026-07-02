"""A small HTTP client for a running VectorMorph server."""

from typing import Any, Dict, List, Optional

import requests


class VectorMorphError(Exception):
    """Raised when the VectorMorph server returns an error response."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class VectorMorphClient:
    """Client for the VectorMorph REST API.

    Example:
        >>> client = VectorMorphClient("http://localhost:4440", token="secret")
        >>> idx = client.add([0.1, 0.2], [0.3, 0.4], metadata={"title": "Doc 1"})
        >>> client.search([0.1, 0.2], k=5)
    """

    def __init__(self, base_url: str, token: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        response = self.session.request(
            method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
        )
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise VectorMorphError(response.status_code, detail)
        return response.json()

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health/")

    def stats(self) -> Dict[str, Any]:
        return self._request("GET", "/stats/")

    def add(
        self,
        summary_vector: List[float],
        document_vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        body = {
            "summary_vector": summary_vector,
            "document_vector": document_vector,
            "metadata": metadata,
        }
        return self._request("POST", "/add/", json=body)["index"]

    def add_batch(self, items: List[Dict[str, Any]]) -> List[int]:
        """Add many vector pairs at once.

        Each item is a dict with ``summary_vector``, ``document_vector`` and
        optionally ``metadata`` keys.
        """
        return self._request("POST", "/add_batch/", json={"items": items})["indices"]

    def get(self, idx: int) -> Dict[str, Any]:
        return self._request("GET", f"/get/{idx}")

    def update(
        self,
        idx: int,
        summary_vector: List[float],
        document_vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        body = {
            "summary_vector": summary_vector,
            "document_vector": document_vector,
            "metadata": metadata,
        }
        return self._request("PUT", f"/update/{idx}", json=body)["index"]

    def delete(self, idx: int) -> None:
        self._request("DELETE", f"/delete/{idx}")

    def search(self, query_vector: List[float], k: int = 10) -> List[Dict[str, Any]]:
        body = {"query_vector": query_vector, "k": k}
        return self._request("POST", "/search/", json=body)["results"]

    def save(self) -> None:
        self._request("POST", "/save/")

    def load(self) -> int:
        return self._request("POST", "/load/")["count"]
