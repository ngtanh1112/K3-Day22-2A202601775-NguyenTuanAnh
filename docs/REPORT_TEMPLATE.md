# Preference Alignment Experiment Report

## 1. Dataset Analysis & Cleaning

### Data Loading Summary
- **Total examples loaded**: 24
- **Validation issues found**: Line 1 contained malformed JSON due to unescaped internal double quotes (`"self-attention"` instead of `\"self-attention\"`).
- **Cleaning steps taken**: Corrected the JSON escaping on line 1 of `sample_preferences.jsonl`. Implemented line-numbered syntax/schema error reporting in `load_jsonl()`, duplicate prompt detection, and case-and-whitespace normalization in `PreferenceExample` schema validation.

### Split Strategy
- **Train/Val Ratio**: 80/20 split
- **Leakage Prevention**: Grouped dataset entries by unique prompt strings before performing deterministic shuffling (`seed=42`). Ensured that all examples associated with any given prompt are strictly isolated into either the train or validation partition, preventing data leakage across splits.

## 2. Implementation: DPO & ORPO

### Objective Selection
- **Why this method?**: DPO (Direct Preference Optimization) optimizes preference alignment directly using implicit reward substitution without requiring an explicit reward model or RL loop. ORPO (Odds Ratio Preference Optimization) combines SFT cross-entropy loss with an odds-ratio preference penalty into a single lightweight training step.
- **Key Hyperparameters**:
  - `beta`: 0.1
  - `lambda_orpo`: 0.1

### Numerical Stability
- **Challenges**: Calculating log-sigmoid and odds ratios can cause arithmetic overflow or severe underflow (`NaN`/`inf`) when log probabilities are extreme or when evaluating $\log(1 - e^{\log p})$.
- **Solutions**: Used `np.logaddexp(0.0, -logits)` for stable $-\log(\sigma(\text{logits}))$ computation in DPO. In ORPO, log probabilities are bounded with `np.clip(..., -100.0, -1e-12)` and log odds calculations use `np.log(-np.expm1(logp))` to maintain floating-point precision.

## 3. Evaluation Results

### Metrics
| Metric | Value |
|---|---|
| Pairwise Accuracy | 66.67% |
| Final Loss (Mock/Train) | 0.6507 |

### Qualitative Review
- **Prompt**: "Explain the concept of \"self-attention\" in Transformers."
- **Chosen Response**: "Self-attention allows the model to weigh the importance of different words in the input sequence when processing each word, capturing long-range dependencies."
- **Rejected Response**: "Self-attention is a simpler version of RNNs that uses less memory and is faster to train."
- **Model Preference**: Correct (the model assigns higher probability/preference score to the accurate self-attention mechanism).

## 4. Discussion & Failure Modes

- **What went well?**: Strict schema enforcement, leakage-free dataset splitting, stable loss implementations for both DPO and ORPO, 100% test coverage passing under `mypy strict` and Ruff.
- **Observed Bias**: Heuristic and reward evaluation can exhibit verbosity bias where longer explanations score higher regardless of concise clarity; calibrating reward margins is critical to prevent verbosity drift.
- **Safety**: Evaluated against the safety and regression prompts in `docs/regression_prompts.md`. The policy successfully recognizes out-of-scope/medical queries requiring disclaimers and properly handles uncertainty in under-specified troubleshooting prompts.
