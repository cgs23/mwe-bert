import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from transformers import BertModel, BertTokenizer

# Import Custom Modules
from MWEBertEmbeddings import BilingualMWEBert
from MWEProcessor import MWEProcessor

# ==========================================
# 1. TEST DATA
# ==========================================
# TEST_SET = [
#     ("Nach diesen starken Zuwächsen haben sich die Aktienkurse eingependelt.",
#      ("Zuwächsen", "eingependelt")),
#     ("Der Vorstand wird auf der Versammlung eine Dividende ausschütten.",
#      ("Dividende", "ausschütten")),
#     ("Die Zentralbank muss dringend Maßnahmen ergreifen.", ("Maßnahmen", "ergreifen")),
#     ("Wir müssen diesen wichtigen Faktor in Betracht ziehen.", ("Betracht", "ziehen")),
#     ("Das Unternehmen wird den Betrag in Rechnung stellen.", ("Rechnung", "stellen"))
# ]

# ==========================================
# EXTENDED TEST DATA FOR EVALUATION
# ==========================================

TEST_SET = [
    # --- CATEGORY 1: FINANCIAL CAUSALITY & EVENTS ---
    (
        "Aufgrund der hohen Inflation muss die Bank die Zinsen erhöhen.",
        ("Inflation", "erhöhen")  # Inflation -> Raise (rates)
    ),
    (
        "Nach dem plötzlichen Kurssturz haben viele Anleger Geld verloren.",
        ("Kurssturz", "verloren")  # Crash -> Lost (money)
    ),
    (
        "Das Unternehmen musste Insolvenz anmelden.",
        ("Insolvenz", "anmelden")  # Bankruptcy -> File (Fixed Expression)
    ),

    # --- CATEGORY 2: FUNCTIONAL VERB CONSTRUCTIONS (Funktionsverbgefüge) ---
    (
        "Die Kommission wird ein Verfahren gegen das Land einleiten.",
        ("Verfahren", "einleiten")  # To initiate a procedure
    ),
    (
        "Wir müssen dieses Risiko wohl oder übel in Kauf nehmen.",
        ("Kauf", "nehmen")  # "in Kauf nehmen" = to accept/risk
    ),
    (
        "Der Präsident wollte zum Ausdruck bringen, dass er besorgt ist.",
        ("Ausdruck", "bringen")  # "zum Ausdruck bringen" = to express
    ),
    (
        "Die Regierung stellte finanzielle Mittel zur Verfügung.",
        ("Verfügung", "stellte")  # "zur Verfügung stellen" = to make available
    ),
    (
        "Er wollte damit Kritik an der neuen Politik üben.",
        ("Kritik", "üben")  # "Kritik üben" = to criticize
    ),

    # --- CATEGORY 3: SEPARABLE VERBS (Long Distance) ---
    (
        "Der Aktienkurs brach nach den schlechten Nachrichten völlig ein.",
        ("brach", "ein")  # "einbrechen" = to collapse
    ),
    (
        "Der Vorstand gab gestern die neuen Zahlen bekannt.",
        ("gab", "bekannt")  # "bekanntgeben" = to announce
    ),
    (
        "Wir fangen heute mit der wichtigen Sitzung an.",
        ("fangen", "an")  # "anfangen" = to begin
    ),
    (
        "Das Parlament stimmt morgen über den Haushalt ab.",
        ("stimmt", "ab")  # "abstimmen" = to vote
    ),

    # --- CATEGORY 4: JOURNALISTIC PHRASING ---
    (
        "Diese Entscheidung steht in direktem Zusammenhang mit der Krise.",
        ("Zusammenhang", "steht")  # "in Zusammenhang stehen" = to be related to
    ),
    (
        "Wir müssen die Verhandlungen leider zum Abschluss bringen.",
        ("Abschluss", "bringen")  # "zum Abschluss bringen" = to conclude/finish
    )
]


def get_vector(model, tokenizer, text, target_word, processor=None, is_custom=False):
    """Extracts 1D vector for a word."""
    model.eval()
    with torch.no_grad():
        if is_custom:
            processed = processor.process_sentence(text)
            device = next(model.parameters()).device
            input_ids = processed['input_ids'].to(device)
            att_mask = processed['attention_mask'].to(device)
            intra = processed['intra_phrase_ids'].to(device)
            inter = processed['inter_phrase_ids'].to(device)
            emb_out = model.bert.embeddings(input_ids, intra, inter)
            outputs = model.bert(inputs_embeds=emb_out,
                                 attention_mask=att_mask)

            ids = input_ids[0].tolist()
            tokens = processor.tokenizer.convert_ids_to_tokens(ids)
            subtokens = processor.tokenizer.tokenize(target_word)
            if not subtokens:
                return None
            try:
                idx = tokens.index(subtokens[0])
            except ValueError:
                return None
        else:
            device = next(model.parameters()).device
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs)
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            subtokens = tokenizer.tokenize(target_word)
            if not subtokens:
                return None
            try:
                idx = tokens.index(subtokens[0])
            except ValueError:
                return None
        return outputs.last_hidden_state[0, idx, :].detach().cpu().numpy().flatten()


def visualize_comparison(results):
    """
    Generates a publication-ready side-by-side comparison plot.
    """
    # 1. Prepare Data for PCA (Combine all vectors to share the same space)
    all_vectors = []

    # Structure: [Std_Head, Std_Dep, MWE_Head, MWE_Dep] per pair
    for res in results:
        all_vectors.extend([
            res['vec_a_std'], res['vec_b_std'],
            res['vec_a_mwe'], res['vec_b_mwe']
        ])

    # 2. Run PCA to reduce to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(np.array(all_vectors))

    # 3. Setup Plot (Side by Side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
        14, 6), dpi=300)  # High DPI for paper

    # Define Colors for the pairs (Professional Palette)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd']  # Blue, Orange, Green, Red, Purple

    # Iterate through pairs (each pair takes up 4 slots in the coords array)
    for i in range(len(results)):
        base_idx = i * 4
        pair_label = f"{results[i]['pair'][0]}-{results[i]['pair'][1]}"
        color = colors[i % len(colors)]

        # --- LEFT PANEL: STANDARD BERT ---
        # Head Word (Triangle)
        ax1.scatter(coords[base_idx, 0], coords[base_idx, 1],
                    marker='^', color=color, s=100, label=results[i]['pair'][0] if i == 0 else "")
        # Dependent Word (Circle)
        ax1.scatter(coords[base_idx+1, 0], coords[base_idx+1, 1],
                    marker='o', color=color, s=100)
        # Connector Line
        ax1.plot([coords[base_idx, 0], coords[base_idx+1, 0]],
                 [coords[base_idx, 1], coords[base_idx+1, 1]],
                 color=color, linestyle='--', alpha=0.6)
        # Annotate
        ax1.text(coords[base_idx, 0], coords[base_idx, 1]+0.1,
                 results[i]['pair'][0], fontsize=8, color=color)

        # --- RIGHT PANEL: MWE BERT ---
        # Head Word
        ax2.scatter(coords[base_idx+2, 0], coords[base_idx+2, 1],
                    marker='^', color=color, s=100)
        # Dependent Word
        ax2.scatter(coords[base_idx+3, 0], coords[base_idx+3, 1],
                    marker='o', color=color, s=100)
        # Connector Line (Solid for emphasis)
        ax2.plot([coords[base_idx+2, 0], coords[base_idx+3, 0]],
                 [coords[base_idx+2, 1], coords[base_idx+3, 1]],
                 color=color, linestyle='-', linewidth=2, alpha=0.8)
        # Annotate
        ax2.text(coords[base_idx+2, 0], coords[base_idx+2, 1] +
                 0.1, results[i]['pair'][0], fontsize=8, color=color)

    # 4. Styling
    ax1.set_title("Standard BERT (Baseline)", fontsize=14, fontweight='bold')
    ax2.set_title("MWE-BERT (Proposed)", fontsize=14, fontweight='bold')

    for ax in [ax1, ax2]:
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")

        # Add dummy legend for markers
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='^', color='w', label='Head Word', markerfacecolor='black', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='Dependent', markerfacecolor='black', markersize=10)]
        ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    output_file = "paper_visualization.png"
    plt.savefig(output_file)
    print(f"\n[Visual] Publication-ready plot saved to {output_file}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}...")

    # Load Models
    print("Loading Standard BERT...")
    std_tok = BertTokenizer.from_pretrained('bert-base-german-cased')
    std_model = BertModel.from_pretrained('bert-base-german-cased').to(device)

    print("Loading Custom MWE-BERT...")
    path = "./financial_mwe_bert_output"
    mwe_proc = MWEProcessor(model_name=path)
    mwe_model = BilingualMWEBert(
        model_name='bert-base-german-cased', tokenizer=mwe_proc.tokenizer)
    mwe_model.load_state_dict(torch.load(os.path.join(
        path, "pytorch_model.bin"), map_location=device))
    mwe_model.to(device)

    results = []

    print("\n--- CALCULATING METRICS ---")
    print(f"{'MWE Pair':<30} | {'Cos Sim (Std)':<15} | {'Cos Sim (MWE)':<15} | {'Improvement'}")
    print("-" * 80)

    total_improvement = 0

    for text, (w1, w2) in TEST_SET:
        v_std_1 = get_vector(std_model, std_tok, text, w1)
        v_std_2 = get_vector(std_model, std_tok, text, w2)
        v_mwe_1 = get_vector(mwe_model, std_tok, text,
                             w1, mwe_proc, is_custom=True)
        v_mwe_2 = get_vector(mwe_model, std_tok, text,
                             w2, mwe_proc, is_custom=True)

        if any(v is None for v in [v_std_1, v_std_2, v_mwe_1, v_mwe_2]):
            continue

        sim_std = 1 - cosine(v_std_1, v_std_2)
        sim_mwe = 1 - cosine(v_mwe_1, v_mwe_2)
        imp = ((sim_mwe - sim_std) / abs(sim_std)) * 100
        total_improvement += imp

        print(
            f"{w1:<15} - {w2:<12} | {sim_std:.4f}          | {sim_mwe:.4f}          | {imp:+.2f}%")

        results.append({
            'pair': (w1, w2),
            'vec_a_std': v_std_1,
            'vec_b_std': v_std_2,
            'vec_a_mwe': v_mwe_1,
            'vec_b_mwe': v_mwe_2
        })

    print("-" * 80)
    print(f"AVERAGE IMPROVEMENT: {total_improvement / len(results):+.2f}%")

    if len(results) > 0:
        visualize_comparison(results)


if __name__ == "__main__":
    main()
