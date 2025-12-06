import os
import random
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from MWEProcessor import MWEProcessor
from MWEBertEmbeddings import BilingualMWEBert


class MWEDataset(Dataset):
    """
    Custom Dataset that uses the MWEProcessor to generate
    inputs with Dual Masking on the fly.
    """

    def __init__(self, texts, processor):
        self.texts = texts
        self.processor = processor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 1. Process: Parse Syntax -> Chunk Phrases -> Pad
        processed_data = self.processor.process_sentence(text)

        # 2. Apply Dual Masking (Gov + Dep)
        # We perform masking dynamically every epoch for data augmentation
        input_ids, labels = self.processor.apply_dual_masking(
            processed_data, masking_prob=0.15)

        # 3. Return the exact tensors required by your model's forward()
        # Squeeze(0) is used because processor returns [1, Seq_Len]
        return {
            'input_ids': input_ids.squeeze(0),
            'intra_phrase_ids': processed_data['intra_phrase_ids'].squeeze(0),
            'inter_phrase_ids': processed_data['inter_phrase_ids'].squeeze(0),
            'labels': labels.squeeze(0)
        }


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move all tensors to GPU/Device
        input_ids = batch['input_ids'].to(device)
        intra_ids = batch['intra_phrase_ids'].to(device)
        inter_ids = batch['inter_phrase_ids'].to(device)
        labels = batch['labels'].to(device)

        # Zero Gradients
        optimizer.zero_grad()

        # --- FORWARD PASS ---
        # Note: We pass the extra arguments specific to your architecture
        outputs = model(
            input_ids=input_ids,
            intra_phrase_ids=intra_ids,
            inter_phrase_ids=inter_ids,
            labels=labels
        )

        # Extract Loss (MLM Loss calculated on Masked Tokens only)
        loss = outputs[0]
        total_loss += loss.item()

        # --- BACKWARD PASS ---
        loss.backward()

        # Clip gradients (standard BERT practice to prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f">> Epoch Complete. Average Loss: {avg_loss:.4f}")
    return avg_loss

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================


# ==========================================
# 1. SETUP DATA
# ==========================================
print("Downloading Europarl dataset (DE-EL)...")

# Load the dataset from Hugging Face
dataset_raw = load_dataset("europarl_bilingual", "de-el", split="train")

general_journalistic_corpus = []
seen_sentences = set()

print("Filtering for sentence complexity...")

# Scan sentences to build the corpus
for entry in dataset_raw.select(range(min(len(dataset_raw), 100000))):
    german_text = entry['translation']['de']

    # Clean up whitespace
    german_text = german_text.strip()

    # FILTER 1: Length (Skip very short or too long sentences)
    word_count = len(german_text.split())
    if word_count < 8 or word_count > 60:
        continue

    # FILTER 2: Duplicates
    if german_text in seen_sentences:
        continue

    # FILTER 3: Basic Quality Check (Capital letter start, punctuation end)
    if not german_text[0].isupper() or german_text[-1] not in ".!?":
        continue

    seen_sentences.add(german_text)
    general_journalistic_corpus.append(german_text)

    # Stop after collecting 10,000 sentences for the experiment
    if len(general_journalistic_corpus) >= 10000:
        break

print(
    f"Dataset prepared. Loaded {len(general_journalistic_corpus)} sentences.")
random.shuffle(general_journalistic_corpus)

# IMPORTANT: Assign to the variable name expected by the Dataset Class below
mwe_corpus = general_journalistic_corpus

# 2. Initialize Processor and Model
print("Initializing Processor and Model...")
# Ensure you have 'de_core_news_lg' installed
processor = MWEProcessor(model_name='bert-base-german-cased')
model = BilingualMWEBert(
    model_name='bert-base-german-cased', tokenizer=processor.tokenizer)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Create DataLoader
dataset = MWEDataset(mwe_corpus, processor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 4. CONFIGURE DIFFERENTIAL LEARNING RATES
# This is the implementation of your request to train phrase layers faster.

# Identify the parameters for the new embedding layers
phrase_params = list(model.bert.embeddings.intra_phrase_embeddings.parameters()) + \
    list(model.bert.embeddings.inter_phrase_embeddings.parameters())

# Identify the parameters for the rest of the model (Base BERT)
base_params = [p for n, p in model.named_parameters()
               if "intra_phrase" not in n and "inter_phrase" not in n]

optimizer = AdamW([
    # Group A: High Learning Rate for NEW Phrase Embeddings
    {'params': phrase_params, 'lr': 1e-3},

    # Group B: Low Learning Rate for PRE-TRAINED BERT (Fine-tuning)
    {'params': base_params, 'lr': 5e-5}
])

# 5. Scheduler
epochs = 3
total_steps = len(dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 6. Start Training
print("Starting Training...")
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_one_epoch(model, dataloader, optimizer, scheduler, device)

# 7. Save the Fine-Tuned Model

# 7. Save the Fine-Tuned Model (Manual Save for Custom Module)
output_dir = "./financial_mwe_bert_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the model weights
torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

# Save the configuration (needed to reload the inner BERT)
model.config.save_pretrained(output_dir)

# Save the tokenizer (crucial because we added special tokens)
processor.tokenizer.save_pretrained(output_dir)

print(f"Model, config, and tokenizer saved to {output_dir}")
