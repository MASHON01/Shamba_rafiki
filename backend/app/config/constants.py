"""
Application-wide constants.

Only immutable values belong here.
"""

APP_AUTHOR = "Shamba Rafiki Team"

API_PREFIX = "/api/v1"

DEFAULT_ENCODING = "utf-8"

DEFAULT_CHUNK_SIZE = 300

DEFAULT_CHUNK_OVERLAP = 50

MIN_DOCUMENT_TEXT_LENGTH = 50
"""Minimum characters of cleaned text for a document to be usable.
Shorter almost always means a failed extraction (blank scan, empty
file) rather than a genuinely short source document."""

MIN_CHUNK_TOKEN_COUNT = 5
"""Minimum tokens for a chunk to be worth indexing. Filters near-empty
chunks produced at document boundaries."""

OCR_FALLBACK_ENABLED = False
"""Whether the text extractor falls back to OCR when a PDF yields no
extractable text (an image-only / scanned PDF). Off for Phase 1 -
the OCR engine is a Phase 3 concern; ocr_interface.py currently
raises NotImplementedError. Flip to True once a real OCR backend is
wired in."""

OCR_MIN_TEXT_LENGTH = 20
"""If a PDF loader returns fewer than this many characters, the
extractor treats it as image-only and (when OCR_FALLBACK_ENABLED)
routes it to OCR. Kept below MIN_DOCUMENT_TEXT_LENGTH so the
extractor decides 'should I OCR?' before the validator decides
'is this usable?'."""

REQUIRED_CHUNK_METADATA_KEYS = ["crop", "county", "document_type", "language"]
"""Metadata keys every chunk should carry once metadata_generator.py
is wired in. Missing keys are logged, not fatal, at this stage."""

METADATA_UNKNOWN = "unknown"
"""Value used when the metadata generator can't confidently detect a
crop/county/etc. from a chunk. Keeps DocumentChunk.metadata as a
clean dict[str, str] rather than mixing in None."""

# Domain vocabulary for lightweight, dependency-free metadata tagging.
# Deliberately narrow: the build plan locks scope to maize, beans,
# tomato (plus cassava, which PlantVillage/KALRO cover and which the
# classifier corpus touches). Keys are canonical labels; values are
# lowercase surface forms (incl. Swahili) matched case-insensitively.
KNOWN_CROPS = {
    "maize": ["maize", "corn", "mahindi"],
    "beans": ["beans", "bean", "maharagwe"],
    "tomato": ["tomato", "tomatoes", "nyanya"],
    "cassava": ["cassava", "muhogo", "mihogo"],
}

# Target counties (scope: 1-2 deeply-sourced counties). Extend as the
# corpus grows; unmatched chunks fall back to METADATA_UNKNOWN.
KNOWN_COUNTIES = {
    "Nakuru": ["nakuru"],
    "Kiambu": ["kiambu"],
    "Machakos": ["machakos"],
    "Meru": ["meru"],
    "Uasin Gishu": ["uasin gishu", "uasin-gishu"],
}

# Coarse document-type inference from filename/source keywords.
DOCUMENT_TYPE_KEYWORDS = {
    "pest_disease_guide": ["pest", "disease", "blight", "rust", "control"],
    "crop_calendar": ["calendar", "season", "planting", "schedule"],
    "market_price": ["price", "market", "kamis", "cost"],
    "extension_manual": ["manual", "guide", "extension", "kalro", "handbook"],
    "strategy": ["strategy", "policy", "afa"],
}

DEFAULT_DOCUMENT_TYPE = "general"
"""Document type assigned when no DOCUMENT_TYPE_KEYWORDS match."""

DEFAULT_TOP_K = 5

# ---------------------------------------------------------------------------
# Embedding / retrieval (Output 4)
# ---------------------------------------------------------------------------

EMBEDDING_DIMENSION = 384
"""Output dimension of the default embedding model
(paraphrase-multilingual-MiniLM-L12-v2). An immutable fact of the
model, not a tunable - the vector store allocates arrays to this
width, so it must match whatever EMBEDDING_MODEL actually produces.
Change this only alongside a model swap."""

EMBEDDING_BATCH_SIZE = 32
"""How many chunk texts the embedder encodes per forward pass.
Batching keeps encoding fast without spiking RAM on the 8GB target
machine; 32 is a safe default for a MiniLM-class model on CPU."""

EMBEDDING_CACHE_VERSION = "v1"
"""Bumped when the cache key scheme or stored format changes, so a
stale on-disk cache is transparently ignored rather than
misinterpreted."""

VECTOR_STORE_BACKEND = "numpy"
"""Which vector-store implementation the retriever/indexer use by
default: 'numpy' (brute-force cosine, zero native deps) or 'faiss'.
numpy is the safe default - a few-thousand-chunk corpus searches
instantly and it can't fail to install on the 8GB target machine.
Flip to 'faiss' only if corpus size ever makes it worthwhile."""

VECTOR_INDEX_FILENAME = "index.faiss"
"""FAISS binary index file (faiss backend only)."""

VECTOR_VECTORS_FILENAME = "vectors.npy"
"""Raw embedding matrix (numpy backend only)."""

VECTOR_MAPPING_FILENAME = "mapping.json"
"""id -> chunk+metadata sidecar. Written by BOTH backends: without
it, an index's integer row ids point to nothing meaningful."""

VECTOR_STORE_META_FILENAME = "store_meta.json"
"""Store-level metadata (backend, dimension, count, metric) used to
validate an index on load before trusting it."""

DEFAULT_TIMEOUT = 60

MAX_UPLOAD_SIZE_MB = 20

SUPPORTED_DOCUMENT_TYPES = [
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
]

SUPPORTED_IMAGE_TYPES = [
    ".jpg",
    ".jpeg",
    ".png",
]

SUPPORTED_AUDIO_TYPES = [
    ".wav",
    ".mp3",
]

LANGUAGE_CODES = {
    "en": "English",
    "sw": "Kiswahili",
}

# ---------------------------------------------------------------------------
# Language intelligence (Output 5)
# ---------------------------------------------------------------------------

# High-signal function words for lightweight, offline language ID.
# These are common, short, and rarely overlap between the two
# languages - so counting them is a cheap, dependency-free detector.
SWAHILI_MARKERS = {
    "na",
    "ya",
    "wa",
    "za",
    "la",
    "kwa",
    "ni",
    "si",
    "yangu",
    "zangu",
    "wangu",
    "gani",
    "nini",
    "vipi",
    "ina",
    "zina",
    "ana",
    "mimi",
    "wewe",
    "yeye",
    "sisi",
    "nyinyi",
    "wao",
    "hii",
    "hiyo",
    "hizi",
    "kuna",
    "kwenye",
    "katika",
    "nataka",
    "naomba",
    "tafadhali",
    "asante",
    "shamba",
    "mmea",
    "mimea",
    "ugonjwa",
    "wadudu",
    "dawa",
    "mbegu",
}

ENGLISH_MARKERS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "my",
    "how",
    "what",
    "why",
    "when",
    "do",
    "does",
    "can",
    "should",
    "i",
    "you",
    "it",
    "this",
    "that",
    "there",
    "in",
    "on",
    "at",
    "with",
    "please",
    "thanks",
    "have",
    "has",
    "of",
    "to",
    "for",
    "and",
    "or",
    "but",
    "will",
    "would",
}

# Pest / disease / symptom / input vocabulary across both languages,
# canonical English label -> surface forms (lowercase). Used by the
# entity extractor and (later) the translator's dictionary fallback.
AGRI_TERMS = {
    "blight": ["blight", "ukungu", "baa"],
    "rust": ["rust", "kutu"],
    "wilt": ["wilt", "kunyauka", "mnyauko"],
    "rot": ["rot", "kuoza", "uozo"],
    "mosaic": ["mosaic", "batobato"],
    "pest": ["pest", "pests", "wadudu", "mdudu"],
    "disease": ["disease", "diseases", "ugonjwa", "magonjwa"],
    "weed": ["weed", "weeds", "magugu"],
    "fungicide": ["fungicide", "dawa ya ukungu"],
    "pesticide": ["pesticide", "dawa ya wadudu", "dawa"],
    "fertilizer": ["fertilizer", "fertiliser", "mbolea"],
    "seed": ["seed", "seeds", "mbegu"],
    "yield": ["yield", "harvest", "mavuno"],
    "leaf": ["leaf", "leaves", "jani", "majani"],
    "soil": ["soil", "udongo"],
}

# Intent keywords, canonical intent -> trigger words (both languages).
# First matching intent (in dict order) wins; nothing matches ->
# DEFAULT_INTENT. Order matters: more specific intents come first.
INTENT_KEYWORDS = {
    "diagnosis": [
        "disease",
        "pest",
        "sick",
        "dying",
        "spots",
        "yellow",
        "wilting",
        "ugonjwa",
        "wadudu",
        "madoa",
        "kunyauka",
        "njano",
        "nini kimeshika",
        "what is wrong",
        "whats wrong",
        "problem",
        # Disease/symptom terms are themselves strong diagnosis signals:
        # a query naming a disease is asking about that disease even if it
        # also says "what should I do" (a how_to trigger).
        "blight",
        "rust",
        "wilt",
        "rot",
        "mosaic",
        "ukungu",
        "baa",
        "kutu",
        "kuoza",
        "uozo",
        "batobato",
        "mnyauko",
    ],
    "price": [
        "price",
        "cost",
        "market",
        "sell",
        "buy",
        "worth",
        "profit",
        "bei",
        "soko",
        "gharama",
        "kuuza",
        "faida",
        "how much",
    ],
    "how_to": [
        "how",
        "how do i",
        "when should",
        "steps",
        "method",
        "apply",
        "plant",
        "grow",
        "vipi",
        "namna",
        "jinsi",
        "lini",
        "nifanye",
        "nipande",
        "kupanda",
    ],
}

DEFAULT_INTENT = "general"
"""Intent assigned when no INTENT_KEYWORDS match."""

ENTITY_TYPE_CROP = "crop"
ENTITY_TYPE_COUNTY = "county"
ENTITY_TYPE_AGRI_TERM = "agri_term"

# ---------------------------------------------------------------------------
# Prompt engine (Output 6)
# ---------------------------------------------------------------------------

# Token budget for the retrieved-context block inside the prompt.
# The context block must leave room for the system prompt, the
# conversation history, the question, and the model's own answer
# within MODEL_CONTEXT_SIZE. This is a conservative slice of a 4096
# context window; the context builder truncates to fit.
MAX_CONTEXT_TOKENS = 1800

# Character-per-token estimate used by the context builder to stay
# within MAX_CONTEXT_TOKENS without loading the LLM tokenizer at
# build time. Deliberately low (pessimistic) so we under-fill rather
# than overflow the window. The LLM's real tokenizer governs at
# generation time; this only bounds how much context we assemble.
CHARS_PER_TOKEN_ESTIMATE = 4

# Minimum similarity for a retrieved chunk to be worth putting in the
# prompt. Below this, a chunk is more likely to mislead than help, so
# the context builder drops it even if retrieval returned it.
MIN_CONTEXT_SIMILARITY = 0.30

# Shown to the model when retrieval returns nothing usable, so it
# answers from general knowledge with an explicit caveat rather than
# fabricating a citation.
NO_CONTEXT_PLACEHOLDER = "No specific reference material was found for this question."

# ---------------------------------------------------------------------------
# LLM generation (Output 6)
# ---------------------------------------------------------------------------

LLM_MAX_TOKENS = 512
"""Max tokens the model may generate per answer. Farmer answers are
short and actionable; capping generation keeps latency (the speed
score) and memory bounded on the 8GB target machine."""

LLM_TEMPERATURE = 0.3
"""Low temperature: advisory answers should be consistent and
grounded, not creative. Higher values invite the model to wander
from the reference material."""

LLM_TOP_P = 0.9

LLM_TIMEOUT_SECONDS = 120
"""How long to wait on llama-server before giving up. CPU generation
of a few hundred tokens on the target machine can be slow, so this
is generous; a hung server surfaces as a typed error, not a freeze."""

LLM_STOP_SEQUENCES = ["\nQuestion:", "\nReference material:"]
"""Stop generation if the model tries to hallucinate a new Q/A turn
or a new reference block - keeps the answer to just the answer."""

# ---------------------------------------------------------------------------
# Conversation memory (Output 6)
# ---------------------------------------------------------------------------

MEMORY_MAX_TURNS = 3
"""How many recent Q/A turns to keep per session. Small on purpose:
enough for a natural follow-up during one farmer's visit, without
bloating the prompt or the RAM budget. Older turns roll off."""

MEMORY_MAX_SESSIONS = 50
"""Cap on concurrently-remembered sessions; the least-recently-used
is evicted past this. Bounds kiosk memory over a full day of
walk-up users."""

# ---------------------------------------------------------------------------
# Verification & confidence (Output 7)
# ---------------------------------------------------------------------------

# Below this fraction of answer content-words appearing in the
# retrieved context, the answer has drifted away from its sources.
SEMANTIC_OVERLAP_MIN = 0.20

# Evidence strength blends retrieval similarity with source coverage.
# Weight on the top source's similarity vs. how many sources corroborate.
EVIDENCE_SIMILARITY_WEIGHT = 0.7
EVIDENCE_COVERAGE_WEIGHT = 0.3

# A "specific claim" is the risky kind of detail to hallucinate: a
# number, dosage, price, or percentage a farmer might act on. The
# hallucination detector flags any that don't appear in the sources.
SPECIFIC_CLAIM_PATTERN = (
    r"\b\d+(?:\.\d+)?\s?(?:%|percent|kg|g|ml|l|litres?|liters?|"
    r"ksh|kes|shillings?|bags?|acres?|hectares?|days?|weeks?|"
    r"months?|times?|/\s?acre|/\s?ha)?\b"
)

# Content words shorter than this are ignored in overlap/citation
# checks (drops "a", "of", "to" noise without a full stopword list).
MIN_CONTENT_WORD_LENGTH = 3

# How the four check scores combine into one confidence score. They
# sum to 1.0. Hallucination and citation carry the most weight because
# an unsupported specific or an uncited claim is the most direct
# hallucination signal; evidence and semantic drift are softer.
CONFIDENCE_WEIGHTS = {
    "hallucination": 0.35,
    "citation": 0.30,
    "evidence": 0.20,
    "semantic": 0.15,
}

# Confidence bands: a blended score at/above HIGH is HIGH, at/above
# MEDIUM is MEDIUM, else LOW.
CONFIDENCE_HIGH_THRESHOLD = 0.75
CONFIDENCE_MEDIUM_THRESHOLD = 0.45

# --- Decision policy (conservative / policy A) ---------------------------
# The policy caveats rather than rejects wherever it reasonably can, so
# the farmer still gets the model's best answer, clearly flagged when
# uncertain. It only hard-replaces an answer in the narrow case of a
# specific fabricated claim with essentially no grounding.

# A hard critical failure: an unsupported *specific* claim (a fabricated
# dose/price) on top of weak grounding. This is the one case policy A
# replaces the answer instead of caveating.
POLICY_REJECT_MAX_EVIDENCE = 0.35

CAVEAT_EN = (
    "\n\nNote: I'm not fully certain about this answer - please confirm "
    "with a local agricultural officer before acting on it."
)
CAVEAT_SW = (
    "\n\nKumbuka: Sina uhakika kamili na jibu hili - tafadhali thibitisha "
    "na afisa wa kilimo wa eneo lako kabla ya kulitekeleza."
)
SAFE_FALLBACK_EN = (
    "I don't have reliable reference material to answer this accurately. "
    "Please consult a local agricultural extension officer."
)
SAFE_FALLBACK_SW = (
    "Sina maelezo ya kuaminika ya kujibu swali hili kwa usahihi. Tafadhali "
    "wasiliana na afisa wa ugani wa kilimo wa eneo lako."
)

# ---------------------------------------------------------------------------
# Profiling & benchmarking (Output 9)
# ---------------------------------------------------------------------------

RAM_CEILING_MB = 7168
"""The hard memory budget: 7 GB (7168 MB). The whole system - model,
index, server, OS headroom - must stay under this on the 8GB target
machine. Exceeding it risks an OOM kill, which is a total failure, so
the profiler flags any peak that crosses this line."""

RAM_WARN_MB = 6144
"""Soft warning line (6 GB). Peaks above this are within budget but
leave little headroom; worth surfacing in the performance report."""

RAM_SAMPLE_INTERVAL_SECONDS = 0.05
"""How often the background RAM sampler reads RSS while an operation
runs. 50 ms is frequent enough to catch a transient spike without
adding measurable overhead."""

BENCHMARK_WARMUP_RUNS = 1
"""Untimed runs before measurement, so one-off costs (lazy model load,
cache warmup) don't skew the reported latency."""

BENCHMARK_MEASURED_RUNS = 5
"""Timed runs per benchmarked operation; the report summarizes these
as p50/p95/mean so a single slow outlier doesn't define the result."""

LOG_FILE_NAME = "backend.log"

REQUEST_ID_HEADER = "X-Request-ID"

HEALTH_ENDPOINT = "/health"

ROOT_ENDPOINT = "/"

VERSION_ENDPOINT = "/version"

# ---------------------------------------------------------------------------
# Vision / image classifier
# ---------------------------------------------------------------------------

CLASSIFIER_INPUT_SIZE = 224
CLASSIFIER_NORM_MEAN = (0.485, 0.456, 0.406)
CLASSIFIER_NORM_STD = (0.229, 0.224, 0.225)
CLASSIFIER_TOP_K = 3
CLASSIFIER_CONFIDENCE_HIGH = 0.75
CLASSIFIER_CONFIDENCE_MEDIUM = 0.45
CLASSIFIER_HEALTHY_CONDITION = "healthy"
CLASSIFIER_LABELS_SUFFIX = ".labels.json"

# ---------------------------------------------------------------------------
# Restored constants (referenced by the model/LLM/prompt/cache modules)
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_VERSION = "v1"

# GGUF file format magic bytes, checked by the validator.
GGUF_MAGIC = b"GGUF"

# Deterministic decoding for reproducible answers.
LLM_DETERMINISTIC_SEED = 42
LLM_MAX_TOKENS_CAP = 1024
LLM_REPEAT_PENALTY = 1.1
LLM_TOP_K = 40

# Retry/backoff for the resilient LLM client.
LLM_RETRY_BASE_DELAY_SECONDS = 0.5
LLM_RETRY_MAX_DELAY_SECONDS = 8.0
LLM_RETRY_BACKOFF_MULTIPLIER = 2.0

# llama-server readiness polling.
MODEL_SERVER_STARTUP_TIMEOUT_S = 120
MODEL_SERVER_POLL_INTERVAL_S = 1.0

# Fractional tolerance when checking a downloaded GGUF's size.
MODEL_SIZE_TOLERANCE = 0.15

# Prompt token budgeting.
PROMPT_ANSWER_RESERVE_TOKENS = 512
PROMPT_SAFETY_MARGIN_TOKENS = 128

# Response cache byte ceiling (kept small so it never threatens RAM).
RESPONSE_CACHE_MAX_BYTES = 8 * 1024 * 1024

# Swahili glossary injection cap.
SWAHILI_GLOSSARY_MAX_TERMS = 20

# Warm-up throwaway generation length.
WARMUP_MAX_TOKENS = 8
