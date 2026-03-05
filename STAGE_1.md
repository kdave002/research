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

## Stage 1 Decision

**Status: COMPLETE — NOVELTY GATE SATISFIED, READY FOR STAGE 2**

Notes:
- Exact-phrase Semantic Scholar checks executed:
  - "reasoning-critical"
  - "information criticality compression"
  - "compression threshold reasoning"
- Abstract-level manual novelty screen completed on 10 papers; evidence saved in `outputs/stage1_novelty_gate.md`.
- No direct RCID-equivalent metric found in this sample.
- Stage 2 will run full 40–50 query sweep with strict API pacing (1 request/second max).
