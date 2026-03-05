# Stage 1: Ideation & Novelty Check

## Research Questions

**Primary RQ**  
Does the compression-to-reasoning-accuracy curve follow a predictable shape (linear, step-function, or threshold-based), and does the shape vary by document genre?

**Secondary RQ**  
Which information types (named entities, causal connectives, temporal markers, numerical values) are disproportionately load-bearing for multi-hop reasoning relative to their token share?

## Proposed Metric

**Reasoning-Critical Information Density (RCID):** the fraction of a memory chunk that is strictly logically required to solve a downstream multi-hop reasoning task.

## Novelty Framing

Existing work has established two important anchors:
- **Size-Fidelity Paradox**: compression can improve efficiency while degrading factual fidelity.
- **Context poisoning / context rot**: long-context accumulation can degrade reasoning quality.

**Gap we target:** current literature does not cleanly quantify which information types are reasoning-critical and where the compression threshold causes abrupt reasoning collapse. RCID is designed to measure exactly that.

## Candidate Angles (Approved Direction)

1. **RCID Threshold Map (Primary):** compression ratio vs. reasoning accuracy across document genres.  
2. **Failure Anatomy (Primary):** identify which information losses (entity, causal, temporal, numeric) drive failure.  
3. **Compression Method Comparison (Substudy):** frequency/token pruning vs. semantic summarization (kept as ablation, not a primary angle).

## Differentiation Notes Against Closest Neighbors

### Paper 4 — Girish et al. (2024), *Fundamental Limits of Prompt Compression*
This work formalizes prompt compression as a **rate–distortion** problem for black-box LLMs and derives optimal distortion–rate bounds under query-aware/query-agnostic settings. Their distortion is defined over output fidelity (e.g., log-loss, 0/1 loss, ROUGE/BERTScore), and the core object is optimal compressed prompt selection under budget. Our RCID direction is different in target and granularity: RCID is not just a global compression-fidelity tradeoff; it explicitly measures **which information types inside memory chunks are reasoning-load-bearing** (entities, causal links, temporal markers, numeric facts) and how removing those types changes **reasoning-chain integrity** in multi-hop tasks. In short: Girish et al. optimize compression under distortion; RCID seeks a token/type-level causal map from information loss to reasoning failure.

### Paper 7 — Lee et al. (2025), *How Well do LLMs Compress Their Own Chain-of-Thought?*
This paper studies compression of **generated reasoning traces** and shows a universal length–accuracy tradeoff with a question-level token-complexity threshold. It is close in spirit but operates on CoT output length rather than upstream memory chunk content. Our RCID agenda focuses on **input memory compression before reasoning** and asks which retained information types are necessary for downstream multi-hop correctness. Their token complexity captures minimum response length to solve a question; RCID captures minimum reasoning-critical information retained from context to preserve chain validity. Both involve thresholds, but at different layers (output trajectory budget vs input memory content criticality).

## Stage 1 Decision

**Status: COMPLETE — NOVELTY GATE SATISFIED, READY FOR STAGE 2**

Notes:
- Exact-phrase Semantic Scholar checks executed:
  - "reasoning-critical"
  - "information criticality compression"
  - "compression threshold reasoning"
- Abstract-level manual novelty screen completed on 10 papers; evidence saved in `outputs/stage1_novelty_gate.md`.
- Full abstract + introduction review completed for closest neighbors (Paper 4 and Paper 7); differentiation notes added above.
- No direct RCID-equivalent metric found in this sample.
- Stage 2 will run full 40–50 query sweep with strict API pacing (1 request/second max).
