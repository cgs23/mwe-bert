# Bilingual MWE-BERT: Fine-tuning with Head-Based Masking

## Overview

This repository contains the implementation for the research paper: "A bilingual study of Multi-Word Expressions in Journalistic Texts: Fine-tune BERT with Head-Based Masking Technique."

The project modifies the standard BERT architecture to better handle Financial Multi-Word Expressions (MWEs) in German. It introduces a Head-Based Masking Technique and a 4-Component Embedding Summation (Token + Intra-Phrase + Phrase + Inter-Phrase) to capture the causal and dependency relationships between distant words in financial texts.

## Directory Structure

```
├── MWEProcessor.py          # Data preprocessing and Dual Masking logic
├── MWEBertEmbeddings.py     # Custom Neural Network architecture (Model definition)
├── main.py                  # Training loop with Differential Learning Rates
├── evaluation.py            # Proof-of-concept metric calculation (Cosine/Euclidean)
└── README.md                # Documentation
```

## File Descriptions & Functionality

### MWEProcessor.py (The Pre-processing Engine)

This script bridges the gap between linguistic theory (Dependency Parsing) and Neural Tensors.

**Functionality:**
- Uses spaCy (`de_core_news_lg`) to parse German sentences and extract syntactic phrases.
- Implements Fixed-Window Padding: Ensures every phrase block is normalized to 10 tokens.
- Injects Boundary Tokens: Adds `[PHRASE_START]` and `[PHRASE_END]` to the vocabulary.

**The Logic:** It converts raw text into three distinct tensors: `input_ids`, `intra_phrase_ids` (position inside the phrase), and `inter_phrase_ids` (position of the phrase in the sentence).

**Dual Masking:** Implements the research hypothesis by masking not just random words, but Dependency Pairs (The Head and the Dependent) simultaneously.

### MWEBertEmbeddings.py (The Custom Architecture)

This file defines the mathematical modifications to the BERT Encoder.

**Functionality:**
- Subclasses `BertEmbeddings` to override the standard vector lookup.
- **The Math:** Implements the formula: $E_{final} = E_{token} + P_{intra} + E_{phrase} + P_{inter}$.
- Contains the `BilingualMWEBert` wrapper class, which swaps the default embeddings of a pre-trained `bert-base-german-cased` model with your custom structure.

### main.py (The Training Loop)

The entry point for fine-tuning the model.

**Functionality:**
- Loads the financial corpus.
- Initializes the `FinancialMWEDataset`.
- **Differential Learning Rates:** Configures the optimizer to use a high learning rate (1e-3) for the new phrase-aware layers and a low rate (5e-5) for the pre-trained BERT layers to prevent catastrophic forgetting.
- Executes the training epochs and saves the fine-tuned model to `./financial_mwe_bert_output`.

### evaluation.py (The Empirical Proof)

Used to verify the research hypothesis.

**Functionality:**
- Loads both the Standard BERT (Baseline) and your Fine-Tuned MWE-BERT.
- Extracts vectors for specific financial word pairs (e.g., "Zuwächsen" and "eingependelt").
- Calculates Cosine Similarity (semantic alignment) and Euclidean Distance (vector clustering).
- Outputs a comparison table showing the percentage improvement in vector proximity.

## Setup & Installation

### Prerequisites

You need Python 3.8+ and the following libraries:

```bash
pip install torch transformers spacy scipy numpy
```

### Download the Language Model

You must download the large German model for spaCy to ensure accurate dependency parsing:

```bash
python -m spacy download de_core_news_lg
```

## How to Execute the Flow

### Step 1: Prepare Your Data

Open `main.py` and locate the `financial_corpus` list.
- For testing, you can use the dummy sentences provided.
- For the actual experiment, load your Europarl or ECB corpus text file into this list.

### Step 2: Run the Training

Execute the main script to start fine-tuning. This will take time depending on your GPU and corpus size.

```bash
python main.py
```

**Output:** You will see epoch loss logs.

**Result:** A new directory `./financial_mwe_bert_output` will be created containing your saved model weights (`pytorch_model.bin`).

### Step 3: Run the Evaluation

Once training is complete, run the evaluation script to prove your hypothesis.

```bash
python evaluation.py
```

**Configuration:** Inside `evaluation.py`, ensure `test_sentence` and the `word_pair` variable match a specific example you want to analyze (e.g., "Zuwächsen", "eingependelt").

**Output:** A console table displaying the Cosine Similarity improvement.

## Configuration (Hyperparameters)

You can adjust specific research parameters inside the files:

- **Phrase Length:** In `MWEProcessor.py`, change `max_phrase_len=10` if you want larger/smaller windows.
- **Masking Probability:** In `MWEProcessor.py`, change `masking_prob=0.15`.
- **Learning Rates:** In `main.py`, under the Optimizer section, adjust `1e-3` (Phrase Layers) or `5e-5` (BERT Base).
