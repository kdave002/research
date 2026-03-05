# Baseline Definitions (for Stage 4)

## Baseline 1: Query-Aware Prompt Compression Baseline
- **Definition:** Reproduce a query-aware hard-prompt compression policy that optimizes compression under output-fidelity distortion proxies.
- **Citation:** Adway Girish et al., 2024, *Fundamental Limits of Prompt Compression: A Rate-Distortion Framework for Black-Box Language Models* (https://www.semanticscholar.org/paper/0e2e91faa0409498f60b4c541eca87e41d7acd1e)
- **Why this baseline:** It is the strongest formal rate–distortion competitor and tests whether RCID adds explanatory power beyond global compression-fidelity optimization.

## Baseline 2: CoT Length Compression Baseline
- **Definition:** Apply prompt-driven compression of chain-of-thought outputs (length budgets / concise prompting) and measure the resulting accuracy-length tradeoff.
- **Citation:** Ayeong Lee et al., 2025, *How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach* (https://www.semanticscholar.org/paper/d6d561786eb06df93e29bcecfc569a3eeb95af10)
- **Why this baseline:** It captures a leading efficiency strategy adjacent to RCID and provides a strong comparison point for threshold behavior.

## Baseline 3: Memory Retrieval-First Baseline
- **Definition:** Use a retrieval-focused memory system that optimizes which memories are fetched, without explicit token-type criticality scoring.
- **Citation:** Xingbo Du et al., 2025, *MemR3: Memory Retrieval via Reflective Reasoning for LLM Agents* (https://www.semanticscholar.org/paper/d6dc1be9960db66b192c6c984adb205f1fe86ec2)
- **Why this baseline:** It represents practical agent memory systems and helps isolate the incremental benefit of RCID’s typed criticality modeling.
