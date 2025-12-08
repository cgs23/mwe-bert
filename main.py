import torch
import os
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset

# Import your custom modules
from MWEBertEmbeddings import BilingualMWEBert
from MWEProcessor import MWEProcessor

# ==========================================
# 1. CUSTOM DATASET CLASS
# ==========================================


class FinancialMWEDataset(Dataset):
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

        # 1. Process: Parse Syntax -> Chunk Phrases -> Pad -> Generate Mask
        # Note: Ensure your MWEProcessor.py is updated to return 'attention_mask'
        processed_data = self.processor.process_sentence(text)

        # 2. Apply Dual Masking (Gov + Dep)
        # We perform masking dynamically every epoch for data augmentation
        input_ids, labels = self.processor.apply_dual_masking(
            processed_data, masking_prob=0.15)

        # 3. Return the exact tensors required by your model's forward()
        return {
            'input_ids': input_ids.squeeze(0),
            # <--- CRITICAL UPDATE
            'attention_mask': processed_data['attention_mask'].squeeze(0),
            'intra_phrase_ids': processed_data['intra_phrase_ids'].squeeze(0),
            'inter_phrase_ids': processed_data['inter_phrase_ids'].squeeze(0),
            'labels': labels.squeeze(0)
        }


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch_idx):
    model.train()
    total_loss = 0

    print(f"  > Starting Epoch {epoch_idx+1}...")

    for batch_idx, batch in enumerate(dataloader):
        # Move all tensors to GPU/Device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(
            device)  # <--- CRITICAL UPDATE
        intra_ids = batch['intra_phrase_ids'].to(device)
        inter_ids = batch['inter_phrase_ids'].to(device)
        labels = batch['labels'].to(device)

        # Zero Gradients
        optimizer.zero_grad()

        # --- FORWARD PASS ---
        # Note: We pass the extra arguments specific to your architecture
        # We MUST pass attention_mask so BERT ignores the padding
        outputs = model(
            input_ids=input_ids,
            intra_phrase_ids=intra_ids,
            inter_phrase_ids=inter_ids,
            labels=labels
        )

        # Extract Loss (MLM Loss calculated on Masked Tokens only)
        # outputs tuple is (loss, logits) because we passed labels
        loss = outputs[0]
        total_loss += loss.item()

        # --- BACKWARD PASS ---
        loss.backward()

        # Clip gradients (standard BERT practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        if scheduler:
            scheduler.step()

        if batch_idx % 50 == 0 and batch_idx > 0:
            print(f"    Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"  >> Epoch {epoch_idx+1} Complete. Average Loss: {avg_loss:.4f}")
    return avg_loss


# ==========================================
# 2. MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":

    # --- SETUP DEVICE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- STEP A: LOAD DATA ---
    print("\n[1/5] Downloading Europarl dataset (DE-EL)...")
    try:
        dataset_raw = load_dataset(
            "europarl_bilingual", "de-el", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please run 'pip install datasets' and ensure internet connection.")
        exit()

    general_journalistic_corpus = []
    seen_sentences = set()

    print("      Filtering for sentence complexity...")
    # Scan sentences to build the corpus
    for entry in dataset_raw.select(range(min(len(dataset_raw), 100000))):
        german_text = entry['translation']['de']
        german_text = german_text.strip()

        # FILTER 1: Length (Skip very short or too long sentences)
        word_count = len(german_text.split())
        if word_count < 8 or word_count > 60:
            continue

        # FILTER 2: Duplicates
        if german_text in seen_sentences:
            continue

        # FILTER 3: Basic Quality Check
        if not german_text[0].isupper() or german_text[-1] not in ".!?":
            continue

        seen_sentences.add(german_text)
        general_journalistic_corpus.append(german_text)

        # Stop after collecting enough sentences
        if len(general_journalistic_corpus) >= 10000:
            break

    print(
        f"      Dataset prepared. Loaded {len(general_journalistic_corpus)} sentences.")
    random.shuffle(general_journalistic_corpus)
    financial_corpus = general_journalistic_corpus

    # --- STEP B: INIT PROCESSOR & MODEL ---
    print("\n[2/5] Initializing Processor and Model...")
    processor = MWEProcessor(model_name='bert-base-german-cased')

    # We pass the tokenizer so the model resizes its embeddings for [PHRASE_START], etc.
    model = BilingualMWEBert(
        model_name='bert-base-german-cased', tokenizer=processor.tokenizer)
    model.to(device)

    dataset = FinancialMWEDataset(financial_corpus, processor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # --- STEP C: TRAINING PHASES (The Fix for Forgetting) ---
    print("\n[3/5] Configuring Training Strategy...")
    epochs = 3

    # PHASE 1: FREEZE BASE BERT (Epoch 0)
    # We only train the NEW Phrase Embeddings and the Head.
    # This prevents the random initialization of phrase embeddings from destroying BERT's knowledge.

    print("      Locking BERT Base layers (Phase 1)...")
    for param in model.bert.parameters():
        param.requires_grad = False

    # Unlock ONLY the new embedding layers
    for param in model.bert.embeddings.intra_phrase_embeddings.parameters():
        param.requires_grad = True
    for param in model.bert.embeddings.inter_phrase_embeddings.parameters():
        param.requires_grad = True
    # Unlock the CLS head (to learn the new special tokens)
    for param in model.cls.parameters():
        param.requires_grad = True

    # Optimizer for Phase 1 (Aggressive Learning Rate)
    optimizer = AdamW(filter(lambda p: p.requires_grad,
                      model.parameters()), lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(dataloader))

    print("\n[4/5] Starting Training Loop...")

    for epoch in range(epochs):

        # --- PHASE SWITCH CHECK ---
        if epoch == 1:
            print("\n      >>> UNLOCKING BERT (Phase 2) <<<")
            print("      Fine-tuning entire model with differential learning rates...")

            # Unfreeze Everything
            for param in model.parameters():
                param.requires_grad = True

            # Re-configure Optimizer:
            # Low LR for BERT (2e-5), Medium LR for New Embeddings (1e-4)
            phrase_params = list(model.bert.embeddings.intra_phrase_embeddings.parameters()) + \
                list(model.bert.embeddings.inter_phrase_embeddings.parameters())

            base_params = [p for n, p in model.named_parameters()
                           if "intra_phrase" not in n and "inter_phrase" not in n]

            optimizer = AdamW([
                {'params': phrase_params, 'lr': 1e-4},
                {'params': base_params, 'lr': 2e-5}
            ])

            # New Scheduler for remaining epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=len(dataloader)*(epochs-1))

        # Run Epoch
        train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch)

    # --- STEP D: SAVE ---
    print("\n[5/5] Saving Model...")
    output_dir = "./financial_mwe_bert_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Manual Save (since it's a custom Module)
    torch.save(model.state_dict(), os.path.join(
        output_dir, "pytorch_model.bin"))
    model.config.save_pretrained(output_dir)
    processor.tokenizer.save_pretrained(output_dir)

    print(f"      Model saved to {output_dir}")
    print("      Ready for evaluation.")
