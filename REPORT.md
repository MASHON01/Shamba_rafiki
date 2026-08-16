# Shamba Rafiki, ADTC 2026 Technical Report

**Track:** Agriculture. **Languages:** English and Kiswahili. **Target:** 8 GB RAM laptop, fully offline.
**Model:** Llama-3.2-1B-Instruct (GGUF Q4_K_M) via llama.cpp.
**Bonuses claimed:** Budget Laptop Profile, African Use Case.

## 1. Problem

Smallholder farmers in Kenya lose a large share of their maize and bean
harvests to pests and diseases they cannot easily diagnose. Extension officers
are few, and the authoritative guidance (KALRO manuals, AFA crop strategies,
KAMIS market data, Infonet-Biovision) is scattered across PDFs and websites
that a farmer with an entry-level phone, patchy connectivity, and a limited
data budget cannot practically consult in the field.

Shamba Rafiki ("farming friend") is an offline advisory tool that runs on a
modest 8 GB laptop, the kind of shared kiosk device a cooperative, agro-dealer,
or extension office can afford. A farmer asks a question in English or
Kiswahili, or uploads a photo of an affected leaf, and gets practical advice
drawn from local reference material, with the sources shown. No internet is
needed at question time.

The intended user is a maize or bean smallholder in counties such as Nakuru,
Kiambu, and Machakos. The scope is deliberately limited to maize and beans, and
to one or two counties we could source real data for, so the reference material
stays accurate instead of broad but shallow.

## 2. Design decisions

**Base model: Llama-3.2-1B-Instruct, Q4_K_M.** We picked the model from
measured numbers, not by assumption. On a 4-core CPU laptop the 1B generates
about 16 tokens per second and peaks at roughly 1.4 GB of RAM, leaving over
5.5 GB of headroom under the 7 GB ceiling. We also measured a 3B Q4_K_M as the
alternative: it writes richer answers but manages only about 5.8 tokens per
second and 3.4 GB of RAM on the same machine. Because the system retrieves
supporting text before it answers, the smaller model's thinner built-in
knowledge is offset by grounding each answer in the reference material, so the
1B keeps English answers practical while roughly doubling the combined speed
and efficiency contribution to the score. All inference goes through llama.cpp
(`llama-server`); no Python machine-learning stack stays resident while serving.

**Retrieval before generation.** The model does not answer from memory alone. A
local set of KALRO, AFA, KAMIS, and Infonet documents is processed offline into
text chunks, embedded with a multilingual MiniLM sentence-transformer
(384-dimensional), and stored in a plain numpy vector store. At question time
the app embeds the question, pulls the closest chunks (cosine similarity at or
above a threshold of 0.45, tuned against the real embedder), and builds a
prompt that asks the model to answer only from that material and to cite it.

**A verification step that catches confident nonsense.** Each answer is checked
against the retrieved sources for citation validity, evidence overlap, and
unsupported specific claims. The result is then approved, caveated, or replaced.
If nothing relevant was retrieved, the app says so and points the farmer to an
extension officer instead of inventing a recommendation. That matters here,
because a wrong pesticide suggestion has a real cost.

**Kiswahili support.** The language is detected automatically, and Swahili
questions get a glossary and a worked example added to the prompt so a small
model produces more usable Swahili. A LoRA fine-tuning pipeline is included as
an optional extension, kept off unless it clearly beats the prompt-only version
without eating the RAM headroom.

**Computer vision paired with language.** A MobileNetV3-small classifier,
trained on PlantVillage maize images and the iBean bean dataset and exported to
ONNX, turns a leaf photo into a crop and disease label. That label is added to
the retrieval query and the prompt, so a photo of a rust-infected bean leaf
pulls the bean-rust guidance and produces a grounded treatment answer. The
vision model is wired into the same retrieval-and-verification path as text,
not bolted on as a separate feature. It reached 95.8% validation accuracy
across seven classes (four maize, three beans), adds under 50 MB of RAM, and
classifies in a few milliseconds. When it is unsure, it contributes at most the
crop name to retrieval, so a weak guess cannot skew the answer.

**Built to survive a real kiosk.** Separate connect and read timeouts, bounded
retries, a circuit breaker, a small exact-match answer cache, and a startup
warm-up. If llama-server is unreachable, the app falls back to showing the
retrieved reference material instead of failing.

## 3. Constraints that shaped the approach

The 8 GB RAM, 4 vCPU, integrated-GPU, offline profile drove most choices.
Running out of memory is an automatic disqualification, so peak memory was
treated as a primary constraint: a small quantized model, a numpy vector store
instead of FAISS, ONNX instead of PyTorch at serve time, and a size-capped
cache all exist to stay under the 7 GB ceiling. Sustained CPU inference can
throttle, so we checked for it rather than assuming. Everything the farmer
touches runs locally; the only network use is the one-time, offline build of
the corpus and model. And because the advice has to trace back to real Kenyan
sources, grounding, citations, and verification are not optional.

## 4. System architecture

```
  Farmer (kiosk browser, English or Kiswahili)
        |  text  or  leaf photo
        v
  FastAPI backend  ->  MobileNetV3 classifier (ONNX, CPU)   [photo path]
        |                    | crop + disease label
        |<-------------------+
        v
  Language analysis -> Retrieval (numpy vector store) -> Prompt build
        v
  llama.cpp llama-server (Llama-3.2-1B Q4_K_M)  ->  Verification  ->  Answer + sources
```

Two local processes run side by side: the FastAPI backend on port 8000 and
llama-server on port 8080. A single static kiosk page is served at `/app`. The
corpus and its index are built ahead of time; the kiosk never embeds a corpus
or trains anything.

## 5. Benchmarks

Measured with the ADTC profiler in participant mode on the submission laptop
(an Intel Core, Family 6 Model 142, 4 cores, 15.8 GB RAM, no discrete GPU,
Windows 11). The laptop has more than 8 GB, but the model peaks near 1.4 GB, so
it fits the 8 GB profile with wide headroom; the authoritative audit runs in
ADTC's 7.5 GB-capped VM.

Throughput:

| Metric | Value |
|---|---|
| Generation throughput | 16.1 tokens/sec |
| First-token latency | 9602 ms (dominated by CPU processing of a 512-token prompt; steady generation is 16.1 tokens/sec) |

Memory:

| Metric | Value |
|---|---|
| Peak RSS | 1393.6 MB |
| Steady-state RSS | 1322.0 MB |
| Headroom below the 7168 MB ceiling | about 5774 MB |
| Within budget | Yes (about 20% of the ceiling used) |

Thermal:

| Metric | Value |
|---|---|
| CPU percent (p99) | 100.0 (fully used during the benchmark, as expected on CPU) |
| Core temperature peak | not exposed by this laptop's sensors (reported as null) |
| Throttled | No, so no thermal penalty |

Supporting accuracy evidence: the image classifier reached 95.8% validation
accuracy across seven classes; live answers cite retrieved KALRO and Infonet
sources; and questions with no supporting material are refused rather than
answered from imagination.

## 6. Self-reported profiler scores

The profiler writes raw telemetry to `submission.json`. The two self-reported
scores come from the official formulas:

- S_perf = min(tokens_per_second / 15, 1.0) × 100
- S_eff  = max(0, (7 − peak_rss_gb) / 7) × 100

| Score | Input | Value |
|---|---|---|
| S_perf (performance) | 16.1 tokens/sec | 100.0 |
| S_eff (efficiency) | 1393.6 MB peak | 80.6 |

For comparison, the 3B alternative on the same laptop scored S_perf 38.3
(5.75 tokens/sec) and S_eff 51.9 (3450.6 MB peak). We chose the 1B because it
roughly doubles the combined performance and efficiency contribution while
retrieval keeps the answer quality practical.

## 7. Scope and known limitations

The corpus and the image classifier cover maize and beans, not every crop or
disease. This is a deliberate choice to keep the reference material accurate for
the crops most Kenyan smallholders grow, and the design extends to more crops
by adding documents and image classes.

The classifier's two hardest classes are bean rust and angular leaf spot, which
look similar on a leaf; it occasionally confuses the two. The verification step
reflects this by lowering the confidence band when the supporting evidence is
weak, and the answer tells the farmer to confirm with an extension officer.

Swahili answers from a 1B model are usable but weaker than English, and the
model can occasionally mix in an English phrase. Retrieval grounding limits how
far off an answer can go, and the honest fallback covers the rest.

## 8. Reproducibility

```bash
# 1. Fetch the model (public, no credentials needed)
bash download_model.sh

# 2. Build llama.cpp, then start the server
#    ./llama.cpp/build/bin/llama-server -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
#        --host 127.0.0.1 --port 8080 -c 4096 -t 4

# 3. Build the offline corpus and index, then run the backend
make corpus
make run                # FastAPI on :8000, kiosk UI at /app

# 4. Profile
adtc-profiler run --submission . --mode participant --output submission.json --skip-accuracy
```

The test suite has 257 passing tests covering ingestion, retrieval, the LLM
resilience and cache layers, the image path, and the API. Model weights and the
trained ONNX classifier are kept out of git and are fetched or produced by the
scripts, in line with the submission rules.

## 9. Bonuses

**Budget Laptop Profile.** The system is designed around the 7 GB ceiling with
measured headroom: a numpy vector store, ONNX inference, and a size-capped cache
keep peak RAM near 1.4 GB, well under the limit.

**African Use Case.** Shamba Rafiki is built for Kenyan smallholder maize and
bean farmers, grounded in Kenyan sources (KALRO, AFA, KAMIS, Infonet-Biovision),
with Kiswahili support and an offline-first design for areas with little or no
connectivity.
