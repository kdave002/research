# Experiment Design: Semantic-Preserving Memory Compression and RCID

## 1. Objective
To empirically determine the relationship between memory compression ratios and multi-hop reasoning performance, and to validate Reasoning-Critical Information Density (RCID) as a predictive metric for reasoning collapse.

## 2. Dataset Selection
We select three datasets specifically chosen for their structured support for multi-hop reasoning and annotated evidence:
1. **MuSiQue (Primary):** Selected for its "compositional" nature and high-quality human-annotated supporting facts, which provide a ground truth for RCID.
2. **HotpotQA:** The industry standard for multi-hop reasoning; provides a large-scale testbed for genre variance.
3. **2WikiMultihopQA:** Chosen to provide structured evidence and diverse reasoning paths (e.g., comparison, inference) to test the robustness of the compression-accuracy curve.

## 3. Experimental Matrix
We use a 3x3x5 grid (45 cells total):
- **Compressors (3):**
    - **Extractive:** LLMLingua (Lite variant for efficiency).
    - **Abstractive:** GPT-4o-driven prompted summarization.
    - **Baseline (Ablation):** TF-IDF sentence ranking.
- **Datasets (3):** MuSiQue, HotpotQA, 2WikiMultihopQA.
- **Compression Ratios (5):** 0.9, 0.7, 0.5, 0.3, 0.1 (Target tokens / Source tokens).

## 4. Model Roster
- **QA Evaluator:** Claude 3.5 Sonnet (selected for superior instruction following and reasoning fidelity).
- **Abstractive Compressor:** GPT-4o.
- **RCID Prober:** Claude 3.5 Sonnet (using the masking protocol).

## 5. Evaluation Metrics
- **Primary:** Exact Match (EM) and F1-score against ground truth answers.
- **Secondary:** RCID Correlation (Pearson/Spearman correlation between RCID and F1 drop).

## 6. Cost Sanity Check (Stage 4 Estimate)
- **QA Grid:** 45 cells × 100 QA pairs = 4,500 evaluations.
- **Evaluator Cost (Claude Sonnet 3.5):** ~$0.003 avg per call (input context dependent) ≈ $13.50.
- **RCID Probing:** 500-sample subset with sentence masking ≈ $15.00.
- **Total Estimated Stage 4 Spend:** **$28.50 - $35.00.**
- **Verdict:** Within the $35 target budget.

## 7. Quality Gates
- **Stage 3 Exit:** Design approved, RCID spec finalized, `src/` stubs pushed.
- **Stage 4 Exit:** 45-cell results collected, RCID values computed for subset, W&B logs verified.
