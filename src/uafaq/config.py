from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"

DATASET = "PaDaS-Lab/webfaq-retrieval"
SEED = 2026
N_TEST = 1000
N_DEV = 200
POOL_SIZES = (2_000, 10_000)
DEPTH = 100

OLLAMA = "http://localhost:11434"
DENSE_MODELS = {
    "bge-m3": ("ollama", "bge-m3"),
    "mminilm": ("st", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
}
EXPANSION_MODEL = "qwen2.5:7b-instruct"
RRF_K = 60
BOOTSTRAP = 2000
