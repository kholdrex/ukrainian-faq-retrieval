# Changes after PROTOCOL.md

1. Exploratory analyses added after the planned results were seen (2026-10-03), in `uafaq exploratory`
   (`results/exploratory.json`). Untuned RRF of BM25 and BGE-M3 was worse than BGE-M3 alone, so fusion weights were
   tuned on the 200 development questions (weighted RRF and a convex combination of min–max normalised scores,
   dense weight 0–1 in steps of 0.1), also for the character n-gram retriever; and two other uses of the LLM
   expansion were tried: retrieval by the hypothetical answer alone and by the sum of the normalised query and answer
   vectors. All are evaluated on the full collection and reported as exploratory.
2. Latency was measured on 100 rather than the planned 200 test questions (2026-10-03), because each expanded query
   also generates a hypothetical answer (about 2.4 s) and the measurement had to run on an otherwise idle shared
   machine; the expansion is now timed together with retrieval of the expanded query.
3. Caches of vectors are validated by a fingerprint of the exact list of texts and their number of rows in every
   command; stored expansions are checked against the question text.
