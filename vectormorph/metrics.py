"""In-memory request metrics for the VectorMorph server."""

import threading
import time
from collections import deque
from typing import Any, Dict


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
    """Thread-safe registry of per-endpoint metrics plus process uptime."""

    def __init__(self):
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.endpoints: Dict[str, EndpointMetrics] = {}

    def record(self, route: str, duration: float, status_code: int):
        with self.lock:
            if route not in self.endpoints:
                self.endpoints[route] = EndpointMetrics()
            self.endpoints[route].record(duration, status_code)

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
        }

    def reset(self):
        with self.lock:
            self.endpoints.clear()
            self.started_at = time.time()
