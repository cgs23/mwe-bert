import torch
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from transformers import BertModel, BertTokenizer
from MWEProcessor import MWEProcessor
from MWEBertEmbeddings import BilingualMWEBert


def get_word_vector(model, tokenizer, text, target_word, processor=None, is_custom_model=False):
    """
    Extracts the embedding vector for a specific word in a sentence.
    Handles both Standard BERT and your Custom MWE-BERT.
    """
    model.eval()

    # 1. Tokenize and Find Index
    # We need to find where the target word lives in the tokenized list
    tokens = tokenizer.tokenize(text)
    try:
        # Find the index of the first sub-token of the target word
        # (Simplified logic: assumes the word exists and is unique in sentence)
        target_token = tokenizer.tokenize(target_word)[0]
        word_idx = tokens.index(target_token)
    except ValueError:
        print(f"Error: Word '{target_word}' not found in sentence.")
        return None

    # 2. Prepare Inputs
    inputs = {}
    with torch.no_grad():
        if is_custom_model:
            # Use your custom processor to generate the 3 ID tensors
            processed_data = processor.process_sentence(text)
            inputs['input_ids'] = processed_data['input_ids']
            inputs['intra_phrase_ids'] = processed_data['intra_phrase_ids']
            inputs['inter_phrase_ids'] = processed_data['inter_phrase_ids']

            # Note: Your model wrapper returns (loss, scores) or scores.
            # We need the hidden states from the internal BERT.
            # We access the internal bert directly for evaluation
            embedding_output = model.bert.embeddings(
                inputs['input_ids'],
                inputs['intra_phrase_ids'],
                inputs['inter_phrase_ids']
            )
            outputs = model.bert(inputs_embeds=embedding_output)

            # word_idx needs adjustment because of [PHRASE_START] tokens in custom model
            # For this demo, we assume the processor tracks the index, or we rely on the
            # fact that we are comparing semantic proximity, so we grab the embedding
            # from the 'last_hidden_state' corresponding to the token.

            # Since the custom processor changes token count (adds boundaries),
            # finding the exact index is complex.
            # STRATEGY: We use the 'phrase_tracker' from the processor to find the exact index.
            tracker = processed_data['phrase_tracker']
            found = False
            for entry in tracker:
                if target_word in entry['phrase_obj'].text:
                    # found the phrase containing the word
                    # Calculation: start_idx + 1 (boundary) + offset inside phrase
                    word_idx = entry['start_idx'] + 1 + \
                        entry['phrase_obj'].text.split().index(target_word)
                    found = True
                    break
            if not found:
                return None

        else:
            # Standard BERT Inputs
            input_ids = torch.tensor(
                [tokenizer.encode(text, add_special_tokens=True)])
            outputs = model(input_ids)
            word_idx += 1  # Account for [CLS]

        # 3. Extract Vector (Last Hidden State)
        # Shape: [Batch, Seq_Len, Hidden_Dim] -> [768]
        vector = outputs.last_hidden_state[0, word_idx, :].numpy()

    return vector


def compare_models(text, word_pair):
    word_a, word_b = word_pair

    print(f"\n--- Analyzing Pair: '{word_a}' <--> '{word_b}' ---")
    print(f"Context: {text}")

    # --- 1. Load Standard Baseline ---
    print("Loading Standard BERT...")
    std_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    std_model = BertModel.from_pretrained('bert-base-german-cased')

    vec_a_std = get_word_vector(std_model, std_tokenizer, text, word_a)
    vec_b_std = get_word_vector(std_model, std_tokenizer, text, word_b)

    # --- 2. Load Your Custom Model ---
    print("Loading Custom MWE-BERT...")
    # Assume 'model_output' is where you saved your trained model
    custom_processor = MWEProcessor(model_name='bert-base-german-cased')
    # Load the state dict or the whole class (simplified here)
    custom_model = BilingualMWEBert(model_name='bert-base-german-cased', tokenizer=custom_processor.tokenizer)
    # custom_model.load_state_dict(torch.load("path_to_your_weights.pth"))

    vec_a_custom = get_word_vector(
        custom_model, std_tokenizer, text, word_a, custom_processor, is_custom_model=True)
    vec_b_custom = get_word_vector(
        custom_model, std_tokenizer, text, word_b, custom_processor, is_custom_model=True)

    # --- 3. Calculate Metrics ---

    # Cosine Distance (1 - Cosine Similarity)
    # We invert it so "Higher Similarity" = "Lower Distance" for easier reading
    # But usually papers report Similarity.

    sim_std = 1 - cosine(vec_a_std, vec_b_std)
    dist_std = euclidean(vec_a_std, vec_b_std)

    sim_custom = 1 - cosine(vec_a_custom, vec_b_custom)
    dist_custom = euclidean(vec_a_custom, vec_b_custom)

    # --- 4. Report Results ---
    print("\nRESULTS:")
    print(f"{'Metric':<20} | {'Standard BERT':<15} | {'Your MWE-BERT':<15} | {'Improvement'}")
    print("-" * 70)

    # Cosine Similarity (Higher is better)
    imp_sim = ((sim_custom - sim_std) / sim_std) * 100
    print(f"{'Cosine Similarity':<20} | {sim_std:.4f}          | {sim_custom:.4f}          | {imp_sim:+.2f}%")

    # Euclidean Distance (Lower is better)
    imp_dist = ((dist_std - dist_custom) / dist_std) * 100
    print(f"{'Euclidean Dist':<20} | {dist_std:.4f}          | {dist_custom:.4f}          | {imp_dist:+.2f}%")


# --- EXECUTE TEST ---
test_sentence = "Nach diesen starken Zuwächsen haben sich die Aktienkurse eingependelt."
# Testing the causal link between 'Gains' (Zuwächsen) and 'Stabilized' (eingependelt)
compare_models(test_sentence, ("Zuwächsen", "eingependelt"))
