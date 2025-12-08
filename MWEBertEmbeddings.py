import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertForMaskedLM


class MWEBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. Standard Token Embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # 2. Intra-Phrase Positional Embeddings (Word position inside phrase)
        # We assume a max phrase length of 50
        self.intra_phrase_embeddings = nn.Embedding(50, config.hidden_size)

        # 3. Inter-Phrase Positional Embeddings (Phrase position inside sentence)
        # We assume max 128 phrases per sentence
        self.inter_phrase_embeddings = nn.Embedding(128, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Standard BERT init is std=0.02. We use 0.01 to make them subtle at start.
        nn.init.normal_(self.intra_phrase_embeddings.weight,
                        mean=0.0, std=0.01)
        nn.init.normal_(self.inter_phrase_embeddings.weight,
                        mean=0.0, std=0.01)

    def forward(self,
                input_ids=None,
                intra_phrase_ids=None,
                inter_phrase_ids=None,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                inputs_embeds=None,
                past_key_values_length=0):

        # --- FIX FOR COMPATIBILITY ---
        # If inputs_embeds is already provided (which happens when BertModel
        # calls this internally after we manually calculated embeddings),
        # we just pass it through without recalculating.
        if inputs_embeds is not None:
            return inputs_embeds
        # -----------------------------

        # Checks to ensure we have the data we need for the custom calculation
        if input_ids is None or intra_phrase_ids is None or inter_phrase_ids is None:
            raise ValueError(
                "MWEBertEmbeddings requires input_ids, intra_phrase_ids, and inter_phrase_ids")

        # Calculate individual components
        input_embeds = self.word_embeddings(input_ids)
        intra_pos_embeds = self.intra_phrase_embeddings(intra_phrase_ids)
        inter_pos_embeds = self.inter_phrase_embeddings(inter_phrase_ids)

        # THE CORE FORMULA: Sum the three components
        embeddings = input_embeds + intra_pos_embeds + inter_pos_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BilingualMWEBert(nn.Module):
    """
    Wrapper class to inject custom embeddings into the standard BERT model
    """

    def __init__(self, model_name='bert-base-german-cased', tokenizer=None):
        super().__init__()
        self.config = BertConfig.from_pretrained(model_name)

        # Load standard BERT
        self.bert = BertModel.from_pretrained(model_name)

        # Resize embeddings if tokenizer is provided (handles [PHRASE_START] etc.)
        if tokenizer is not None:
            self.bert.resize_token_embeddings(len(tokenizer))
            self.config.vocab_size = len(tokenizer)

        # SWAP the embeddings layer with your Custom MWE Layer
        self.bert.embeddings = MWEBertEmbeddings(self.config)

        # We must also load the CLS head for masking
        self.cls = BertForMaskedLM.from_pretrained(model_name).cls

        # Resize the output head
        if tokenizer is not None:
            self.cls.predictions.decoder = nn.Linear(
                self.config.hidden_size, len(tokenizer))
            self.cls.predictions.bias = nn.Parameter(
                torch.zeros(len(tokenizer)))

    def forward(self, input_ids, intra_phrase_ids, inter_phrase_ids, labels=None):

        # 1. Manually calculate embeddings using your custom logic
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            intra_phrase_ids=intra_phrase_ids,
            inter_phrase_ids=inter_phrase_ids
        )

        # 2. Pass these pre-calculated embeddings to BERT
        # Note: BertModel will internally call self.bert.embeddings AGAIN,
        # but because we pass 'inputs_embeds', our fixed 'forward' method
        # (lines 35-37) will just return them immediately.
        outputs = self.bert(inputs_embeds=embedding_output)

        sequence_output = outputs.last_hidden_state

        # Prediction Head
        prediction_scores = self.cls(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1,
                            self.config.vocab_size), labels.view(-1))

        return (loss, prediction_scores) if loss is not None else prediction_scores
