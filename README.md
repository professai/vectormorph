# VectorMorph

A lightweight vector database that allows LLMs to efficiently search multiple loaded documents.

VectorMorph stores pairs of vectors: a **summary vector**, which is indexed with
[hnswlib](https://github.com/nmslib/hnswlib) for fast approximate nearest-neighbour
search, and a **document vector**, which is used to re-rank the candidates by exact
similarity. This two-stage "hypervector" scheme lets you index short summaries while
still ranking against full-document embeddings. Each pair can carry arbitrary JSON
metadata (document IDs, titles, source text, …) that is returned with search results.

## 🚀 Quick Install

```bash
pip install vectormorph
```

## 🏃 Running the Server

The server requires a bearer token for authentication:

```bash
export BEARER_TOKEN=your-secret-token
vectormorph
```

The API is now available at `http://localhost:4440`, with interactive OpenAPI docs at
`http://localhost:4440/docs` and a monitoring/control dashboard at
`http://localhost:4440/dashboard/` (paste the bearer token into the page to see
stats, request metrics, and use the save/load/reboot/shutdown controls).

### Configuration

| Environment variable   | Default              | Description                                  |
| ---------------------- | -------------------- | -------------------------------------------- |
| `BEARER_TOKEN`         | *(required)*         | Shared secret for authenticating requests    |
| `VECTORMORPH_HOST`     | `0.0.0.0`            | Host interface to bind                       |
| `VECTORMORPH_PORT`     | `4440`               | Port to listen on                            |
| `VECTORMORPH_DATA_DIR` | `<package dir>/bin`  | Directory used by `/save/` and `/load/`      |
| `VECTORMORPH_AUTOLOAD` | `1`                  | Load a previously saved database on startup (`0` disables) |
| `VECTORMORPH_LOOPGUARD` | `1`                 | Doom-loop protection (`0` disables)          |
| `VECTORMORPH_LOOPGUARD_THRESHOLD` | `15`      | Failures of the same request before blocking |
| `VECTORMORPH_LOOPGUARD_WINDOW` | `30`         | Seconds the failures must fall within        |
| `VECTORMORPH_LOOPGUARD_COOLDOWN` | `30`       | Seconds the offending request stays blocked  |

## 📖 API

All endpoints except `/health/` require an `Authorization: Bearer <token>` header.

| Method   | Path            | Description                                              |
| -------- | --------------- | -------------------------------------------------------- |
| `GET`    | `/health/`      | Liveness probe (no auth)                                 |
| `GET`    | `/dashboard/`   | Health & control dashboard (page is public; its data/controls need the token) |
| `GET`    | `/stats/`       | Vector count, deleted count, dimensionality, capacity    |
| `GET`    | `/metrics/`     | Uptime, request/error counts, latency percentiles per endpoint |
| `POST`   | `/add/`         | Add a summary/document vector pair (with optional metadata); returns its index |
| `POST`   | `/add_batch/`   | Add many vector pairs in one request                     |
| `GET`    | `/get/{idx}`    | Return the vectors and metadata at an index              |
| `PUT`    | `/update/{idx}` | Replace the vectors at an index                          |
| `DELETE` | `/delete/{idx}` | Delete the vectors at an index                           |
| `POST`   | `/search/`      | k-NN search on summary vectors, re-ranked by document vectors |
| `POST`   | `/save/`        | Persist the index and vectors to disk                    |
| `POST`   | `/load/`        | Restore a previously saved database                      |
| `POST`   | `/shutdown/`    | Shut down the server                                     |
| `POST`   | `/reboot/`      | Restart the server process                               |

The dimensionality of the database is set by the first vector added; all subsequent
vectors must match it.

### Doom-loop protection

LLM agents (and buggy retry logic) sometimes get stuck replaying the same failing
request forever. VectorMorph watches failures per client + method + path: when the
same client repeats the same failing request more than
`VECTORMORPH_LOOPGUARD_THRESHOLD` times within the window, that request is blocked
for the cooldown period and receives `429 Too Many Requests` with a `Retry-After`
header. A successful request clears the counter, `/health/` and `/dashboard/` are
exempt, and active blocks are shown on the dashboard and in `/metrics/`.

### Example

```bash
# Add a vector pair with metadata
curl -X POST http://localhost:4440/add/ \
  -H "Authorization: Bearer $BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"summary_vector": [0.1, 0.2, 0.3], "document_vector": [0.4, 0.5, 0.6], "metadata": {"title": "Doc 1"}}'

# Search
curl -X POST http://localhost:4440/search/ \
  -H "Authorization: Bearer $BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query_vector": [0.1, 0.2, 0.3], "k": 5}'
```

## 🐍 Python Client

The package ships with a small client for the REST API:

```python
from vectormorph import VectorMorphClient

client = VectorMorphClient("http://localhost:4440", token="your-secret-token")

idx = client.add(
    summary_vector=[0.1, 0.2, 0.3],
    document_vector=[0.4, 0.5, 0.6],
    metadata={"title": "Doc 1"},
)

results = client.search([0.1, 0.2, 0.3], k=5)
# [{"index": 0, "similarity": 0.32, "summary_distance": 0.0, "metadata": {"title": "Doc 1"}}]

client.save()  # persist to disk; auto-loaded on the next server start
```

## 🐳 Docker Compose Example

The repository contains an end-to-end example: a VectorMorph server plus a small
client service that embeds text with the OpenAI API and stores/searches it.

```bash
cp .env.example .env   # then edit .env with your token and OpenAI API key
docker compose up --build
```

- VectorMorph server: `http://localhost:4440/docs`
- Example client: `http://localhost:80/docs`

## 🧪 Development

```bash
pip install -e '.[dev]'
pytest
```

## 📄 License

Apache-2.0
