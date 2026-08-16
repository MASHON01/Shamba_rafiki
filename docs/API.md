# API Reference

The backend exposes a small HTTP API. Interactive OpenAPI docs are
served at `/docs` (Swagger UI) and `/redoc` when the app is running.

Base URL (default): `http://localhost:8000`

## Response envelope

Every response uses a standard envelope.

**Success:**
```json
{
  "success": true,
  "request_id": "uuid",
  "timestamp": "ISO-8601",
  "message": "Answer generated.",
  "data": {... },
  "metadata": {... }
}
```

**Error:**
```json
{
  "success": false,
  "request_id": "uuid",
  "timestamp": "ISO-8601",
  "error": { "code": "ERROR_CODE", "message": "...", "details": null }
}
```

Every request accepts and echoes an `X-Request-ID` header for tracing.

## Endpoints

### `GET /`
Liveness + identity. Returns app name, version, environment, status.

### `GET /health`
Deep health. Reports whether the LLM backend (llama-server) is
reachable via `data.llm_available`. Returns HTTP 200 even when the LLM
is down (the API itself is up); the payload carries the real status.

### `GET /version`
Returns the application version.

### `POST /chat`
The main endpoint, a farmer's question.

**Request body:**
```json
{
  "query": "How do I treat maize blight in Nakuru?",
  "language": "en",
  "session_id": "optional-session-id"
}
```
- `query` (string, required, min length 1)
- `language` (string, default `"en"`), `"en"` or `"sw"`
- `session_id` (string, optional), enables follow-up context within
  one farmer's visit; omit for a stateless question

**Response `data`:** `answer`, `language`, `intent`, `sources` (list of
grounding chunks with crop/county/score), `grounded` (bool),
`confidence` (`low`/`medium`/`high`), `verification_action`
(`approved`/`caveated`/`replaced`).

**Response `metadata`:** `session_id`, token counts, `latency_ms`,
`confidence_score`, `verification_flags`, extracted `entities`.

Errors: `422` on an empty/missing `query`; `LLM_UNAVAILABLE` (HTTP 503-
class, returned in the envelope) if llama-server can't be reached.

### `POST /retrieve`
Raw retrieval, no LLM, useful for debugging corpus quality.

**Request body:**
```json
{ "query": "maize blight fungicide", "top_k": 3 }
```
- `query` (string, required, min length 1)
- `top_k` (int, optional, 1, 20)

**Response `data`:** `index_loaded` (bool), `count`, and `results` (each
with `text`, `score`, `crop`, `county`, `source_filename`). If no index
is built yet, returns `index_loaded: false` with an explanatory message
rather than an error.

### `POST /upload`
Save a document to the corpus for **offline** indexing. Multipart file
upload. The file is validated (supported type, size within
`MAX_UPLOAD_BYTES`) and saved to `data/raw_documents/`; it is **not**
indexed live (indexing is a build-time step, run `scripts/build_index.py`).

Returns `saved_as`, `size_bytes`, `indexed: false`. Errors: `400`
(`UnsupportedFormatError`) for a bad type; `400` for an empty or
oversized file.

## Example

```bash
curl -s http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I treat maize blight?","language":"en"}'
```