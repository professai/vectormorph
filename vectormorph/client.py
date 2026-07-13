"""A small HTTP client for a running VectorMorph server."""

import time
from typing import Any, Dict, List, Optional

import requests

RETRYABLE_STATUSES = {429, 502, 503, 504}


class VectorMorphError(Exception):
    """Raised when the VectorMorph server returns an error response."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class VectorMorphClient:
    """Client for the VectorMorph REST API.

    Transient failures (429 including doom-loop blocks, 502/503/504, and
    connection errors) are retried up to `retries` times with exponential
    backoff, honouring the server's Retry-After header when present.

    Example:
        >>> client = VectorMorphClient("http://localhost:4440", token="secret")
        >>> idx = client.add([0.1, 0.2], [0.3, 0.4], metadata={"title": "Doc 1"})
        >>> client.search([0.1, 0.2], k=5, filter={"title": "Doc 1"})
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        timeout: float = 30.0,
        retries: int = 2,
        max_retry_wait: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.max_retry_wait = max_retry_wait
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        last_error = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self._retry_wait(attempt, last_error))
            try:
                response = self.session.request(
                    method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
                )
            except requests.ConnectionError as exc:
                last_error = exc
                continue
            if response.ok:
                return response.json()
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            last_error = VectorMorphError(response.status_code, detail)
            last_error.retry_after = self._parse_retry_after(response)
            if response.status_code not in RETRYABLE_STATUSES:
                raise last_error

        if isinstance(last_error, VectorMorphError):
            raise last_error
        raise VectorMorphError(0, f"Connection failed after {self.retries + 1} attempts: {last_error}")

    @staticmethod
    def _parse_retry_after(response) -> Optional[float]:
        value = response.headers.get("Retry-After")
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    def _retry_wait(self, attempt: int, last_error) -> float:
        retry_after = getattr(last_error, "retry_after", None)
        wait = retry_after if retry_after is not None else 2 ** (attempt - 1)
        return min(wait, self.max_retry_wait)

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health/")

    def stats(self) -> Dict[str, Any]:
        return self._request("GET", "/stats/")

    def metrics(self) -> Dict[str, Any]:
        return self._request("GET", "/metrics/")

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

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        body = {"query_vector": query_vector, "k": k, "filter": filter}
        return self._request("POST", "/search/", json=body)["results"]

    def compact(self) -> Dict[str, Any]:
        return self._request("POST", "/compact/")

    def save(self) -> None:
        self._request("POST", "/save/")

    def load(self) -> int:
        return self._request("POST", "/load/")["count"]
