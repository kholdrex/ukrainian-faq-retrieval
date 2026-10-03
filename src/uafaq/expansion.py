"""Hypothetical-answer query expansion with a local LLM (HyDE-style)."""
import json
import time
import urllib.request

from .config import EXPANSION_MODEL, OLLAMA

SYSTEM = (
    "Ти допомагаєш пошуковій системі. На запитання користувача напиши коротку правдоподібну відповідь українською "
    "мовою, одне-два речення, так, як вона могла б виглядати на сторінці поширених запитань сайту. "
    "Не додавай пояснень і застережень."
)


def expand(query: str, model: str = EXPANSION_MODEL) -> dict:
    body = {"model": model, "stream": False, "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": 96, "seed": 0},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]}
    request = urllib.request.Request(f"{OLLAMA}/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    began = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        reply = json.load(response)
    return {"answer": reply["message"]["content"].strip(), "seconds": time.perf_counter() - began,
            "prompt_tokens": reply.get("prompt_eval_count"), "output_tokens": reply.get("eval_count")}
