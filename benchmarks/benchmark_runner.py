import json
import requests
import pathlib
import time
from datetime import datetime

SERVER = "http://127.0.0.1:8080/completion"

MODEL_NAME = "llama"

OUTPUT_DIR = pathlib.Path("results") / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open("prompts.json", encoding="utf-8") as f:
    prompts = json.load(f)

summary = []

for item in prompts:

    payload = {
        "prompt": item["prompt"],
        "temperature": 0.2,
        "n_predict": 150
    }

    start = time.time()

    response = requests.post(SERVER, json=payload)

    elapsed = time.time() - start

    response.raise_for_status()

    data = response.json()

    output = data["content"]

    with open(
        OUTPUT_DIR / f"{item['id']}.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(output)

    summary.append({

        "id": item["id"],

        "seconds": round(elapsed,2),

        "tokens_generated": data["tokens_predicted"]

    })

print()

print("Benchmark Complete")

print()

for row in summary:

    print(row)