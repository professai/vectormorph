# Changelog

## 0.6.0 (2026-07-13)

- Metadata-filtered search: `POST /search/` accepts a `filter` object; only
  vectors whose metadata contains every given key/value are returned (the knn
  fetch widens automatically until k matches are found or the whole database
  has been examined)
- `POST /compact/` rebuilds the index without deleted slots, reclaiming their
  space (surviving vectors are re-labelled sequentially in original order)
- Client: transient failures (429 including doom-loop blocks, 502/503/504,
  connection errors) are retried with exponential backoff, honouring the
  server's Retry-After header; new `metrics()` and `compact()` methods and a
  `filter` argument on `search()`
- docker-compose: healthcheck on the server container; client waits for it

## 0.5.0 (2026-07-13)

- Doom-loop protection: a client repeating the same failing request past a
  threshold is blocked for a cooldown with `429` + `Retry-After`
  (configure via `VECTORMORPH_LOOPGUARD*`; state in `/metrics/` and dashboard)

## 0.4.0 (2026-07-13)

- In-memory request metrics with per-endpoint counts, errors, and latency
  percentiles, plus a rolling time series (5-second buckets, 10 minutes)
- `GET /metrics/` endpoint and a `GET /dashboard/` health/control dashboard
  with live charts, tooltips, time-window filter, and save/load/reboot/
  shutdown controls
- Atomic saves (temp files + rename), all-or-nothing loads, search re-ranking
  moved inside the database lock

## 0.3.0 (2026-07-13)

- Arbitrary JSON metadata per vector pair, returned in search results
- `POST /add_batch/` and `GET /get/{idx}` endpoints
- Auto-load of a saved database on startup (`VECTORMORPH_AUTOLOAD=0` disables)
- `VectorMorphClient` Python client
- Example Docker image builds from repository source; ruff lint in CI

## 0.2.0 (2026-07-13)

- Rewrote the core database: index initialisation, updates, deletes,
  persistence, and JSON serialisation all fixed and covered by tests
- Bearer auth hardened (constant-time comparison); `POST /search/`;
  new `/health/`, `/stats/`, `/save/`, `/load/` endpoints
- Packaging moved to `pyproject.toml`; GitHub Actions CI

## 0.1.3 and earlier

- Initial releases
