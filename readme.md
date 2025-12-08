# Bilingual MWE-BERT: Fine-tuning with Head-Based Masking

## Project Overview

This repository contains the algorithm and implementation for the research paper: **"A bilingual study of Multi-Word Expressions in Journalistic Texts: Fine-tune BERT with Head-Based Masking Technique."**

The study introduces a novel **Head-Based Masking Technique** and a **4-Component Embedding Architecture** ($E_{token} + P_{intra} + E_{phrase} + P_{inter}$) to improve the numerical representation of Multi-Word Expressions (MWEs) in German financial and journalistic texts.

The goal is to solve the "Distributed Semantic" problem (e.g., Separable Verbs like *brach... ein*) by forcing the model to treat distant components as a single semantic unit.

## Visual Results

The evaluation script generates a PCA projection showing how the custom model "pulls" semantic pairs together compared to standard BERT.

**Figure 1:** PCA projection of financial MWE vector pairs. The left panel shows vector proximity in Standard BERT, while the right panel shows the proposed MWE-BERT. Note the tighter clustering of causal pairs (e.g., *Zuwächsen-eingependelt*) in the proposed model.

![PCA Visualization](paper_visualization.png)

## Directory Structure

```
├── MWEProcessor.py          # Pre-processing, Chunking, Attention Masks, Dual Masking
├── MWEBertEmbeddings.py     # Custom Neural Architecture (The 4-Component Sum)
├── main.py                  # 2-Phase Training Loop (Freeze -> Fine-tune)
├── evaluation.py            # Metric Calculation (Cosine/Euclidean) & Visualization
└── README.md                # Documentation
```

## Installation & Setup

### Environment
- Python 3.8+

### Dependencies

```bash
pip install torch transformers spacy datasets scipy numpy matplotlib scikit-learn
```

### Language Model

Download the large German model for spaCy (critical for dependency parsing):

```bash
python -m spacy download de_core_news_lg
```

## How to Run the Experiment

### 1. Training (main.py)

The training script automatically downloads the Europarl (DE-EL) dataset and filters for complex journalistic sentences.

It implements a **Two-Phase Training Strategy** to prevent Catastrophic Forgetting:

- **Phase 1 (Epoch 1):** The Base BERT model is **FROZEN**. Only the new Phrase Embeddings and CLS head are trained. This aligns the new architecture without destroying pre-trained knowledge.
- **Phase 2 (Epoch 2+):** The Base BERT is **UNFROZEN**. The entire model is fine-tuned using differential learning rates (Lower LR for BERT, Higher LR for Phrase Embeddings).

```bash
python main.py
```

**Output:** Saves the model weights, config, and tokenizer to `./financial_mwe_bert_output`.

### 2. Evaluation (evaluation.py)

This script loads the saved custom model and compares it against a standard `bert-base-german-cased`. It calculates:

- **Cosine Similarity Improvement:** (Higher is better)
- **Euclidean Distance Reduction:** (Lower is better)

It tests against a curated list of Financial Causality, Separable Verbs, and Fixed Expressions.

```bash
python evaluation.py
```

**Output:** Prints a metrics table to the console and generates `paper_visualization.png`.

## Evaluation Results

The evaluation script tests the model on a curated test set (defined in `evaluation.py` under the `TEST_SET` array) containing 14 MWE pairs across four categories:

1. **Financial Causality & Events** (3 pairs)
2. **Functional Verb Constructions** (5 pairs)
3. **Separable Verbs** (4 pairs)
4. **Journalistic Phrasing** (2 pairs)

### Results Table

| MWE Pair | Cos Sim (Std) | Cos Sim (MWE) | Improvement |
|----------|---------------|---------------|-------------|
| Inflation - erhöhen | 0.6114 | 0.7522 | +23.04% |
| Kurssturz - verloren | 0.5572 | 0.6790 | +21.86% |
| Insolvenz - anmelden | 0.6467 | 0.6554 | +1.35% |
| Verfahren - einleiten | 0.7293 | 0.6911 | -5.23% |
| Kauf - nehmen | 0.6117 | 0.8426 | +37.75% |
| Ausdruck - bringen | 0.6636 | 0.7368 | +11.03% |
| Verfügung - stellte | 0.6355 | 0.7234 | +13.83% |
| Kritik - üben | 0.6184 | 0.8511 | +37.63% |
| brach - ein | 0.5022 | 0.6714 | +33.70% |
| gab - bekannt | 0.6260 | 0.8088 | +29.20% |
| fangen - an | 0.4098 | 0.9451 | +130.64% |
| stimmt - ab | 0.4095 | 0.6958 | +69.92% |
| Zusammenhang - steht | 0.6405 | 0.8299 | +29.58% |
| Abschluss - bringen | 0.5972 | 0.7425 | +24.32% |
| **AVERAGE IMPROVEMENT** | | | **+32.76%** |

### Key Findings

- The MWE-BERT model achieves an **average improvement of +32.76%** in cosine similarity compared to standard BERT
- Separable verbs show the most dramatic improvements, with "fangen - an" achieving a **+130.64%** improvement
- The model successfully captures semantic relationships in distant word pairs, addressing the "Distributed Semantic" problem

## Technical Architecture Details

### MWEProcessor.py

- **Robust Chunking:** Identifies Noun Chunks via spaCy and treats remaining verbs/prepositions as single-token phrases to ensure 100% sentence coverage.
- **Attention Masks:** Generates masks to ensure BERT ignores the padding tokens used to enforce the fixed phrase window (10 tokens).
- **Dual Masking:** Implements the core hypothesis by masking the Governor and Dependent simultaneously (e.g., masking both "Inflation" and "erhöhen").

### MWEBertEmbeddings.py

- **Custom Forward Pass:** Overrides standard BERT embeddings.
- **Initialization:** Uses `nn.init.normal_` with a low standard deviation (0.01) for new embeddings to reduce initialization noise.
- **Safe Resizing:** Includes logic to resize the embedding matrix dynamically to accommodate special tokens (`[PHRASE_START]`, `[PHRASE_END]`).

### main.py

- **Manual Saving:** Uses `torch.save(model.state_dict())` because the custom `BilingualMWEBert` class wraps the Hugging Face model and requires manual weight serialization.
- **Differential Optimization:** Uses AdamW with parameter groups to apply aggressive learning rates (1e-3) to new layers and conservative rates (2e-5) to pre-trained layers.
