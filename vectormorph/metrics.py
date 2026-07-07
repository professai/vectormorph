"""In-memory request metrics for the VectorMorph server."""

import threading
import time
from collections import deque
from typing import Any, Dict, List


def _percentile(sorted_values, fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(len(sorted_values) * fraction), len(sorted_values) - 1)
    return sorted_values[index]


class EndpointMetrics:
    """Rolling stats for one route: counts, errors, and recent latencies."""

    WINDOW = 500  # latencies kept per endpoint for percentile estimates

    def __init__(self):
        self.count = 0
        self.errors_4xx = 0
        self.errors_5xx = 0
        self.total_time = 0.0
        self.latencies = deque(maxlen=self.WINDOW)

    def record(self, duration: float, status_code: int):
        self.count += 1
        self.total_time += duration
        self.latencies.append(duration)
        if 400 <= status_code < 500:
            self.errors_4xx += 1
        elif status_code >= 500:
            self.errors_5xx += 1

    def snapshot(self) -> Dict[str, Any]:
        ordered = sorted(self.latencies)
        return {
            "count": self.count,
            "errors_4xx": self.errors_4xx,
            "errors_5xx": self.errors_5xx,
            "avg_ms": round(self.total_time / self.count * 1000, 3) if self.count else 0.0,
            "p50_ms": round(_percentile(ordered, 0.50) * 1000, 3),
            "p95_ms": round(_percentile(ordered, 0.95) * 1000, 3),
        }


class MetricsRegistry:
    """Thread-safe registry of per-endpoint metrics, uptime, and a short
    time series of request activity for dashboard charts."""

    BUCKET_SECONDS = 5
    HISTORY_BUCKETS = 120  # 10 minutes of 5-second buckets
    BUCKET_LATENCY_CAP = 200  # latencies kept per bucket for p95

    def __init__(self):
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.endpoints: Dict[str, EndpointMetrics] = {}
        self.buckets: Dict[int, Dict[str, Any]] = {}

    def _bucket_start(self, timestamp: float) -> int:
        return int(timestamp // self.BUCKET_SECONDS) * self.BUCKET_SECONDS

    def record(self, route: str, duration: float, status_code: int):
        now = time.time()
        with self.lock:
            if route not in self.endpoints:
                self.endpoints[route] = EndpointMetrics()
            self.endpoints[route].record(duration, status_code)

            start = self._bucket_start(now)
            bucket = self.buckets.setdefault(
                start,
                {"requests": 0, "errors_4xx": 0, "errors_5xx": 0, "latencies": []},
            )
            bucket["requests"] += 1
            if 400 <= status_code < 500:
                bucket["errors_4xx"] += 1
            elif status_code >= 500:
                bucket["errors_5xx"] += 1
            if len(bucket["latencies"]) < self.BUCKET_LATENCY_CAP:
                bucket["latencies"].append(duration)

            horizon = start - self.HISTORY_BUCKETS * self.BUCKET_SECONDS
            for key in [k for k in self.buckets if k < horizon]:
                del self.buckets[key]

    def history(self) -> List[Dict[str, Any]]:
        """Contiguous per-bucket series covering the retention window,
        oldest first; empty buckets are zero-filled."""
        now_bucket = self._bucket_start(time.time())
        first = now_bucket - (self.HISTORY_BUCKETS - 1) * self.BUCKET_SECONDS
        with self.lock:
            series = []
            for start in range(first, now_bucket + self.BUCKET_SECONDS, self.BUCKET_SECONDS):
                bucket = self.buckets.get(start)
                if bucket and bucket["requests"]:
                    ordered = sorted(bucket["latencies"])
                    series.append(
                        {
                            "t": start,
                            "requests": bucket["requests"],
                            "errors_4xx": bucket["errors_4xx"],
                            "errors_5xx": bucket["errors_5xx"],
                            "avg_ms": round(sum(ordered) / len(ordered) * 1000, 3),
                            "p95_ms": round(_percentile(ordered, 0.95) * 1000, 3),
                        }
                    )
                else:
                    series.append(
                        {
                            "t": start,
                            "requests": 0,
                            "errors_4xx": 0,
                            "errors_5xx": 0,
                            "avg_ms": None,
                            "p95_ms": None,
                        }
                    )
            return series

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            endpoints = {route: m.snapshot() for route, m in sorted(self.endpoints.items())}
        totals = {
            "requests": sum(e["count"] for e in endpoints.values()),
            "errors_4xx": sum(e["errors_4xx"] for e in endpoints.values()),
            "errors_5xx": sum(e["errors_5xx"] for e in endpoints.values()),
        }
        return {
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "totals": totals,
            "endpoints": endpoints,
            "history": self.history(),
            "bucket_seconds": self.BUCKET_SECONDS,
        }

    def reset(self):
        with self.lock:
            self.endpoints.clear()
            self.buckets.clear()
            self.started_at = time.time()
