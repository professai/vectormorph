"""Doom-loop protection: detect clients stuck repeating a failing request.

An agent (or buggy retry logic) that keeps replaying the same failing call
can hammer the server indefinitely. The guard watches failures per
(client, method, path); when the same client repeats the same failing
request more than `threshold` times within `window_seconds`, that
combination is blocked for `cooldown_seconds` and receives 429 responses
with a Retry-After header. A successful request clears the counter.
"""

import os
import threading
import time
from collections import deque
from typing import Any, Dict, Tuple

Key = Tuple[str, str, str]  # (client, method, path)


class LoopGuard:
    MAX_TRACKED_KEYS = 1024

    def __init__(
        self,
        enabled: bool = True,
        threshold: int = 15,
        window_seconds: float = 30.0,
        cooldown_seconds: float = 30.0,
    ):
        self.enabled = enabled
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.lock = threading.Lock()
        self.failures: Dict[Key, deque] = {}
        self.blocked_until: Dict[Key, float] = {}
        self.last_seen: Dict[Key, float] = {}
        self.loops_detected_total = 0
        self.requests_blocked_total = 0

    @classmethod
    def from_env(cls) -> "LoopGuard":
        return cls(
            enabled=os.environ.get("VECTORMORPH_LOOPGUARD", "1") != "0",
            threshold=int(os.environ.get("VECTORMORPH_LOOPGUARD_THRESHOLD", "15")),
            window_seconds=float(os.environ.get("VECTORMORPH_LOOPGUARD_WINDOW", "30")),
            cooldown_seconds=float(os.environ.get("VECTORMORPH_LOOPGUARD_COOLDOWN", "30")),
        )

    def retry_after(self, key: Key, now: float = None) -> float:
        """Seconds the key is still blocked for; 0 if not blocked."""
        if not self.enabled:
            return 0.0
        now = time.time() if now is None else now
        with self.lock:
            until = self.blocked_until.get(key, 0.0)
            if until <= now:
                return 0.0
            self.requests_blocked_total += 1
            return until - now

    def record(self, key: Key, status_code: int, now: float = None):
        """Record the outcome of a handled request."""
        if not self.enabled:
            return
        now = time.time() if now is None else now
        with self.lock:
            self.last_seen[key] = now
            if status_code < 400:
                self.failures.pop(key, None)
                return

            history = self.failures.setdefault(key, deque())
            history.append(now)
            while history and history[0] < now - self.window_seconds:
                history.popleft()
            if len(history) > self.threshold:
                self.blocked_until[key] = now + self.cooldown_seconds
                self.loops_detected_total += 1
                history.clear()

            self._prune(now)

    def _prune(self, now: float):
        # Drop expired blocks, then cap tracked keys by evicting the stalest.
        for key in [k for k, until in self.blocked_until.items() if until <= now]:
            del self.blocked_until[key]
        if len(self.last_seen) > self.MAX_TRACKED_KEYS:
            stale = sorted(self.last_seen, key=self.last_seen.get)
            for key in stale[: len(self.last_seen) - self.MAX_TRACKED_KEYS]:
                if key not in self.blocked_until:
                    self.last_seen.pop(key, None)
                    self.failures.pop(key, None)

    def snapshot(self, now: float = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        with self.lock:
            active = [
                {
                    "client": key[0],
                    "method": key[1],
                    "path": key[2],
                    "retry_after": round(until - now, 1),
                }
                for key, until in sorted(self.blocked_until.items())
                if until > now
            ]
            return {
                "enabled": self.enabled,
                "threshold": self.threshold,
                "window_seconds": self.window_seconds,
                "cooldown_seconds": self.cooldown_seconds,
                "active_blocks": len(active),
                "loops_detected_total": self.loops_detected_total,
                "requests_blocked_total": self.requests_blocked_total,
                "active": active[:20],
            }

    def reset(self):
        with self.lock:
            self.failures.clear()
            self.blocked_until.clear()
            self.last_seen.clear()
            self.loops_detected_total = 0
            self.requests_blocked_total = 0
