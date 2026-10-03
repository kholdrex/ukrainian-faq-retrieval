# Changes after PROTOCOL.md

1. Exploratory analyses added after the planned results were seen (2026-10-03), in `uafaq exploratory`
   (`results/exploratory.json`). Untuned RRF of BM25 and BGE-M3 was worse than BGE-M3 alone, so fusion weights were
   tuned on the 200 development questions (weighted RRF and a convex combination of min–max normalised scores,
   dense weight 0–1 in steps of 0.1), also for the character n-gram retriever; and two other uses of the LLM
   expansion were tried: retrieval by the hypothetical answer alone and by the sum of the normalised query and answer
   vectors. All are evaluated on the full collection and reported as exploratory.
