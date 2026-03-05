# RCID Specification: Reasoning-Critical Information Density

## 1. Definition
Reasoning-Critical Information Density (RCID) measures the proportion of information in a context chunk that is strictly necessary for a model to successfully complete a multi-hop reasoning task.

## 2. Formalism
Given a context $C$, a question $Q$, and a reasoning model $f$:
- Let $A = f(C, Q)$ be the baseline answer.
- Partition $C$ into $N$ semantic units (sentences) $\{s_1, s_2, ..., s_N\}$.
- For each $s_i$, compute the answer $A_{\setminus s_i} = f(C \setminus \{s_i\}, Q)$.
- Let $\Delta_i = \text{Distortion}(A, A_{\setminus s_i})$ where Distortion measures the drop in Exact-Match or F1 accuracy.

A sentence $s_i$ is **Reasoning-Critical** if $\Delta_i > \epsilon$, where $\epsilon$ is a sensitivity threshold (default: 0.1).

$$RCID(C, Q, \epsilon) = \frac{\sum_{i=1}^{N} \mathbb{1}(\Delta_i > \epsilon)}{N}$$

## 3. Computation Protocol (Pseudocode)
```python
def compute_rcid(context, question, ground_truth, model, epsilon=0.1):
    sentences = split_into_sentences(context)
    baseline_f1 = evaluate(model(context, question), ground_truth)
    
    critical_count = 0
    for s in sentences:
        masked_context = context.replace(s, "[MASKED]")
        masked_f1 = evaluate(model(masked_context, question), ground_truth)
        
        delta = baseline_f1 - masked_f1
        if delta > epsilon:
            critical_count += 1
            
    return critical_count / len(sentences)
```

## 4. Edge Cases & Sensitivity
- **Redundancy:** If the same fact appears twice, neither sentence may trigger the threshold individually. *Resolution:* Treat both as non-critical unless both are removed simultaneously; this includes a pairwise masking pass for sentences with identical named entities or paraphrased facts.
- **Threshold Sensitivity:** We will test $\epsilon \in \{0.05, 0.1, 0.2\}$ to ensure the RCID metric is stable.
- **Irrelevant Sentences:** Sentences with zero impact on accuracy contribute to the denominator, lowering the density as expected.
