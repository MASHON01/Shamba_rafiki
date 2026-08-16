# Shamba Rafiki

An offline agricultural advisory tool for smallholder maize and bean farmers in
Kenya. It runs on a modest 8 GB laptop with no internet at question time. A
farmer types a question in English or Kiswahili, or uploads a photo of an
affected leaf, and gets practical advice drawn from local reference material
(KALRO, AFA, KAMIS, Infonet-Biovision), with the sources shown.

Built for the Africa Deep Tech Challenge 2026, Agriculture track. For the full
technical writeup and benchmarks, see [REPORT.md](REPORT.md).

## What it does

- Answers farming questions from a local document corpus, not from the model's
  memory alone (retrieval before generation).
- Classifies a leaf photo (maize and bean diseases) with a small CPU vision
  model, and uses that result to fetch the right guidance.
- Works in English and Kiswahili.
- Checks each answer against its sources and, when nothing relevant is found,
  says so instead of guessing.
- Runs fully offline once the model and corpus are in place.

## How it works

```
Farmer (text or leaf photo)
        -> image classifier (for photos)
        -> language detection
        -> retrieval from the local corpus
        -> prompt building
        -> local model (llama.cpp)
        -> verification
        -> answer with sources
```

The backend is FastAPI. Text generation runs through llama.cpp (`llama-server`).
Embeddings use a multilingual MiniLM model; the vector store is plain numpy. The
image model is MobileNetV3-small exported to ONNX and run on the CPU. A single
static kiosk page is served at `/app`.

## Model

Llama-3.2-1B-Instruct, Q4_K_M GGUF. On a 4-core laptop it generates about 16
tokens per second and peaks near 1.4 GB of RAM, well under the 7 GB budget.
Model weights are not committed; `download_model.sh` fetches them.

## Setup

```bash
# 1. Get the model weights (public, no login)
bash download_model.sh

# 2. Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate            # Windows (use: source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt

# 3. Configure
copy .env.example .env            # Windows (use: cp on Linux/Mac), then edit if needed

# 4. Build llama.cpp and start the model server
#    llama.cpp/build/bin/llama-server -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
#        --host 127.0.0.1 --port 8080 -c 4096 -t 4

# 5. Build the offline corpus and index (one time)
make corpus

# 6. Run the backend
make run
```

Then open the kiosk at http://localhost:8000/app and the API docs at
http://localhost:8000/docs.

## Tech stack

| Component | Technology |
|---|---|
| API | FastAPI (Python 3.12) |
| Text model | llama.cpp, Llama-3.2-1B-Instruct Q4_K_M |
| Embeddings | Sentence-Transformers (multilingual MiniLM) |
| Vector store | numpy |
| Image model | MobileNetV3-small via ONNX Runtime |
| Config / logging | Pydantic v2, structlog |

## Tests

```bash
make test
```

## License

MIT
