# VectorMorph

A lightweight vector database that allows LLMs to efficiently search multiple loaded documents.

VectorMorph stores pairs of vectors: a **summary vector**, which is indexed with
[hnswlib](https://github.com/nmslib/hnswlib) for fast approximate nearest-neighbour
search, and a **document vector**, which is used to re-rank the candidates by exact
similarity. This two-stage "hypervector" scheme lets you index short summaries while
still ranking against full-document embeddings.

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
`http://localhost:4440/docs`.

### Configuration

| Environment variable   | Default              | Description                                  |
| ---------------------- | -------------------- | -------------------------------------------- |
| `BEARER_TOKEN`         | *(required)*         | Shared secret for authenticating requests    |
| `VECTORMORPH_HOST`     | `0.0.0.0`            | Host interface to bind                       |
| `VECTORMORPH_PORT`     | `4440`               | Port to listen on                            |
| `VECTORMORPH_DATA_DIR` | `<package dir>/bin`  | Directory used by `/save/` and `/load/`      |

## 📖 API

All endpoints except `/health/` require an `Authorization: Bearer <token>` header.

| Method   | Path            | Description                                              |
| -------- | --------------- | -------------------------------------------------------- |
| `GET`    | `/health/`      | Liveness probe (no auth)                                 |
| `GET`    | `/stats/`       | Vector count and dimensionality                          |
| `POST`   | `/add/`         | Add a summary/document vector pair; returns its index    |
| `PUT`    | `/update/{idx}` | Replace the vectors at an index                          |
| `DELETE` | `/delete/{idx}` | Delete the vectors at an index                           |
| `POST`   | `/search/`      | k-NN search on summary vectors, re-ranked by document vectors |
| `POST`   | `/save/`        | Persist the index and vectors to disk                    |
| `POST`   | `/load/`        | Restore a previously saved database                      |
| `POST`   | `/shutdown/`    | Shut down the server                                     |
| `POST`   | `/reboot/`      | Restart the server process                               |

The dimensionality of the database is set by the first vector added; all subsequent
vectors must match it.

### Example

```bash
# Add a vector pair
curl -X POST http://localhost:4440/add/ \
  -H "Authorization: Bearer $BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"summary_vector": [0.1, 0.2, 0.3], "document_vector": [0.4, 0.5, 0.6]}'

# Search
curl -X POST http://localhost:4440/search/ \
  -H "Authorization: Bearer $BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query_vector": [0.1, 0.2, 0.3], "k": 5}'
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
