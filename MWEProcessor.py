import torch
import spacy
import numpy as np
from transformers import BertTokenizer


class MWEProcessor:
    def __init__(self, model_name='bert-base-german-cased', max_phrase_len=10, max_seq_len=128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # Add special boundary tokens to the vocabulary
        special_tokens = {'additional_special_tokens': [
            '[PHRASE_START]', '[PHRASE_END]']}
        self.tokenizer.add_special_tokens(special_tokens)

        # Load spaCy for dependency parsing
        print("Loading spaCy model...")
        self.nlp = spacy.load("de_core_news_lg")

        self.max_phrase_len = max_phrase_len
        self.max_seq_len = max_seq_len

        # IDs for special tokens
        self.phrase_start_id = self.tokenizer.convert_tokens_to_ids(
            '[PHRASE_START]')
        self.phrase_end_id = self.tokenizer.convert_tokens_to_ids(
            '[PHRASE_END]')
        self.pad_id = self.tokenizer.pad_token_id
        self.mask_id = self.tokenizer.mask_token_id

    def _pad_phrase(self, token_ids):
        """
        Ensures every phrase is exactly 10 tokens long (including boundaries).
        Format: [PHRASE_START] + tokens + [PHRASE_END] + padding
        """
        # Truncate if too long (reserving 2 spots for Start/End)
        limit = self.max_phrase_len - 2
        tokens = token_ids[:limit]

        # Construct the phrase block
        phrase_block = [self.phrase_start_id] + tokens + [self.phrase_end_id]

        # Apply padding if shorter than 10
        padding_needed = self.max_phrase_len - len(phrase_block)
        if padding_needed > 0:
            phrase_block += [self.pad_id] * padding_needed

        return phrase_block

    def process_sentence(self, text):
        """
        Main pipeline: Syntax Parse -> Phrase Extraction -> Padding -> ID Generation
        """
        doc = self.nlp(text)

        final_input_ids = []
        final_intra_ids = []
        final_inter_ids = []

        # We track phrases to alignment for masking later
        phrase_tracker = []

        # 1. Chunking Strategy (Refined based on Levy & Goldberg)
        # We use spaCy's yield_doc_slices or noun_chunks, but for full coverage
        # we often iterate recursively. Here we use a simplified head-based chunker.
        # We treat the subtree of every major head as a 'phrase' roughly.

        # For this implementation, we will iterate over spaCy's specific Noun Chunks
        # and Verbal clusters. Any word not in a chunk gets treated as a mini-phrase.

        # Simplified: Linear iteration grouping by syntactic head clusters

        # Let's assume we have extracted a list of spans (phrases) from the doc.
        # E.g. [Span("Nach diesen starken Zuwächsen"), Span("haben sich...")]
        # (In a production script, you would write a dedicated recursive tree walker here)
        phrases = [chunk for chunk in doc.noun_chunks]
        # Note: You would need to add verbal phrases here manually as spaCy noun_chunks excludes verbs.

        phrase_idx_counter = 0

        for phrase in phrases:
            # Tokenize phrase text
            # We use the text of the chunk
            sub_tokens = self.tokenizer.tokenize(phrase.text)
            token_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)

            # Pad and Structure the phrase (Strategy: Fixed Window 10)
            padded_block = self._pad_phrase(token_ids)

            # Append to main list
            final_input_ids.extend(padded_block)

            # Generate Positional IDs
            # Intra-Phrase: 0, 1, 2... 9 (reset for every phrase)
            intra_ids = list(range(self.max_phrase_len))
            final_intra_ids.extend(intra_ids)

            # Inter-Phrase: 0, 0, 0... (same ID for all tokens in this phrase)
            inter_ids = [phrase_idx_counter] * self.max_phrase_len
            final_inter_ids.extend(inter_ids)

            phrase_tracker.append({
                'phrase_obj': phrase,
                # Start index in the final list
                'start_idx': len(final_input_ids) - self.max_phrase_len,
                'head': phrase.root  # The syntactic head of this phrase
            })

            phrase_idx_counter += 1

        # Truncate or Pad the WHOLE sequence to max_seq_len (e.g. 128 or 512)
        if len(final_input_ids) > self.max_seq_len:
            final_input_ids = final_input_ids[:self.max_seq_len]
            final_intra_ids = final_intra_ids[:self.max_seq_len]
            final_inter_ids = final_inter_ids[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(final_input_ids)
            final_input_ids += [self.pad_id] * pad_len
            # Padding position is irrelevant (masked usually)
            final_intra_ids += [0] * pad_len
            final_inter_ids += [0] * pad_len

        return {
            "input_ids": torch.tensor([final_input_ids]),
            "intra_phrase_ids": torch.tensor([final_intra_ids]),
            "inter_phrase_ids": torch.tensor([final_inter_ids]),
            "phrase_tracker": phrase_tracker  # Critical for the masking step
        }

    def apply_dual_masking(self, processed_data, masking_prob=0.15):
        """
        Implementation of:
        1. Syntactic Double Masking (Gov + Dep)
        2. Phrase Masking (Left - [MASK] - Right)
        """
        input_ids = processed_data['input_ids'].clone()
        # Labels are the original IDs
        labels = processed_data['input_ids'].clone()

        phrase_tracker = processed_data['phrase_tracker']

        # Create a boolean mask tensor
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        # --- STRATEGY A: DEPENDENCY PAIR MASKING ---
        for entry in phrase_tracker:
            phrase = entry['phrase_obj']
            head_token = entry['head']

            # Check if this head has a syntactic dependency on another phrase's head
            # (e.g., 'Zuwächsen' (pobj) -> depends on 'nach' (prep))
            governor = head_token.head

            # Find the phrase that contains the governor
            gov_entry = next(
                (p for p in phrase_tracker if p['head'] == governor), None)

            if gov_entry and governor != head_token:  # Avoid self-loops
                if torch.rand(1).item() < masking_prob:
                    # MASK DEPENDENT (Current Phrase Head)
                    # We need to find the specific token index of the head within the phrase block
                    # Simplified: We mask the center of the phrase or the whole content

                    # Masking the content of the Dependent Phrase
                    start_dep = entry['start_idx'] + \
                        1  # +1 to skip [PHRASE_START]
                    end_dep = start_dep + \
                        len(self.tokenizer.tokenize(phrase.text))
                    masked_indices[0, start_dep:end_dep] = True

                    # Masking the content of the Governor Phrase
                    start_gov = gov_entry['start_idx'] + 1
                    end_gov = start_gov + \
                        len(self.tokenizer.tokenize(
                            gov_entry['phrase_obj'].text))
                    masked_indices[0, start_gov:end_gov] = True

        # --- STRATEGY B: WHOLE PHRASE MASKING (Context Prediction) ---
        # "Mask the middle phrase" logic
        # Iterate through phrases in triplets
        for i in range(1, len(phrase_tracker) - 1):
            if torch.rand(1).item() < (masking_prob / 2):  # Lower prob to mix strategies
                target_phrase = phrase_tracker[i]

                # Mask the entire block (including start/end tokens or just content?)
                # Usually just content to preserve structure
                s_idx = target_phrase['start_idx'] + 1
                e_idx = s_idx + self.max_phrase_len - 2  # Mask content area

                masked_indices[0, s_idx:e_idx] = True

        # Apply the [MASK] token to the inputs
        input_ids[masked_indices] = self.mask_id

        # Ignore loss calculation for unmasked tokens (set label to -100)
        labels[~masked_indices] = -100

        return input_ids, labels


# --- USAGE EXAMPLE ---
text = "Nach diesen starken Zuwächsen haben sich die Aktienkurse eingependelt."

processor = MWEProcessor()
data = processor.process_sentence(text)
masked_inputs, masked_labels = processor.apply_dual_masking(data)

print("Original Input Shape:", data['input_ids'].shape)
print("Masked Inputs (sample):", masked_inputs[0][:30])  # Show first 30 tokens