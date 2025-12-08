import torch
import spacy
from transformers import BertTokenizer


class MWEProcessor:
    def __init__(self, model_name='bert-base-german-cased', max_phrase_len=10, max_seq_len=128):
        # Handle case where model_name is a directory (loading saved model)
        if '/' in model_name or '\\' in model_name:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            # Only add tokens if loading from base name (not saved directory)
            special_tokens = {'additional_special_tokens': [
                '[PHRASE_START]', '[PHRASE_END]']}
            self.tokenizer.add_special_tokens(special_tokens)

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
        """
        limit = self.max_phrase_len - 2
        tokens = token_ids[:limit]
        phrase_block = [self.phrase_start_id] + tokens + [self.phrase_end_id]

        padding_needed = self.max_phrase_len - len(phrase_block)
        if padding_needed > 0:
            phrase_block += [self.pad_id] * padding_needed

        return phrase_block

    def process_sentence(self, text):
        doc = self.nlp(text)

        final_input_ids = []
        final_intra_ids = []
        final_inter_ids = []
        phrase_tracker = []

        # --- FIXED CHUNKING LOGIC ---
        # 1. Identify tokens covered by Noun Chunks
        phrases = []
        covered_indices = set()

        for chunk in doc.noun_chunks:
            phrases.append(chunk)
            for token in chunk:
                covered_indices.add(token.i)

        # 2. Catch ALL remaining tokens (Verbs, Prepositions, Punctuation)
        for token in doc:
            if token.i not in covered_indices:
                # Create a span for the single token
                span = doc[token.i: token.i+1]
                phrases.append(span)

        # 3. Sort phrases by their position in the sentence
        # This restores the correct word order: "Nach" -> "diesen..." -> "haben"
        phrases.sort(key=lambda x: x.start)
        # ----------------------------

        phrase_idx_counter = 0

        for phrase in phrases:
            # Tokenize phrase text
            sub_tokens = self.tokenizer.tokenize(phrase.text)
            token_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)

            padded_block = self._pad_phrase(token_ids)

            final_input_ids.extend(padded_block)

            intra_ids = list(range(self.max_phrase_len))
            final_intra_ids.extend(intra_ids)

            inter_ids = [phrase_idx_counter] * self.max_phrase_len
            final_inter_ids.extend(inter_ids)

            phrase_tracker.append({
                'phrase_obj': phrase,
                'start_idx': len(final_input_ids) - self.max_phrase_len,
                'head': phrase.root
            })

            phrase_idx_counter += 1

        # Truncate or Pad Sequence
        if len(final_input_ids) > self.max_seq_len:
            final_input_ids = final_input_ids[:self.max_seq_len]
            final_intra_ids = final_intra_ids[:self.max_seq_len]
            final_inter_ids = final_inter_ids[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(final_input_ids)
            final_input_ids += [self.pad_id] * pad_len
            final_intra_ids += [0] * pad_len
            final_inter_ids += [0] * pad_len

# --- NEW CODE: GENERATE ATTENTION MASK ---
        # 1 = Real Token, 0 = Padding
        # We must mask BOTH the standard pads AND the pads inside phrases
        attention_mask = [1 if token_id !=
                          self.pad_id else 0 for token_id in final_input_ids]

        return {
            "input_ids": torch.tensor([final_input_ids]),
            "attention_mask": torch.tensor([attention_mask]),
            "intra_phrase_ids": torch.tensor([final_intra_ids]),
            "inter_phrase_ids": torch.tensor([final_inter_ids]),
            "phrase_tracker": phrase_tracker
        }

    def apply_dual_masking(self, processed_data, masking_prob=0.15):
        input_ids = processed_data['input_ids'].clone()
        labels = processed_data['input_ids'].clone()
        phrase_tracker = processed_data['phrase_tracker']

        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        # STRATEGY A: GOVERNOR-DEPENDENT MASKING
        for entry in phrase_tracker:
            head_token = entry['head']
            governor = head_token.head

            gov_entry = next(
                (p for p in phrase_tracker if p['head'] == governor), None)

            if gov_entry and governor != head_token:
                if torch.rand(1).item() < masking_prob:
                    # Mask Dependent Content
                    start_dep = entry['start_idx'] + 1
                    end_dep = start_dep + \
                        len(self.tokenizer.tokenize(entry['phrase_obj'].text))
                    masked_indices[0, start_dep:end_dep] = True

                    # Mask Governor Content
                    start_gov = gov_entry['start_idx'] + 1
                    end_gov = start_gov + \
                        len(self.tokenizer.tokenize(
                            gov_entry['phrase_obj'].text))
                    masked_indices[0, start_gov:end_gov] = True

        # Apply [MASK]
        input_ids[masked_indices] = self.mask_id
        labels[~masked_indices] = -100

        return input_ids, labels
