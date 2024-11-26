from post_processing_multimodel import wiki_anchor, get_models, device_to_embed_map
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
import itertools
from tqdm import tqdm
import json
from collections import defaultdict

items = [
    ("hg", "arkohut/jina-embeddings-v3"),
    ("st", "all-mpnet-base-v2"),
    ("st", "Alibaba-NLP/gte-multilingual-base"),
    ("st", "dunzhang/stella_en_1.5B_v5"),
    ("st", "intfloat/multilingual-e5-large-instruct"),
    ("st", "paraphrase-multilingual-mpnet-base-v2")
]

# Function to generate all combinations of parameters for a specific argument dictionary
def get_param_combinations(params):
    keys, values = zip(*params.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations

def wiki_anchor_eval():
    data_path = Path("/home/yazici/playground/new-prompts/output/gpt-4o-2024-08-06_event2_newest_report_20241107-035313.json")
    anchor_file_path = Path("/mnt/datasets/dop-position-mining/wiki-anchor/anchor_target_counts.csv")
    eval_output_path = Path("/home/yazici/playground/new-prompts/output/eval_multi_model_output.json")
    # read eval_output.json (if it exists, otherwise create it)
    if eval_output_path.exists():
        with open(eval_output_path, "r") as f:
            eval_output = json.load(f)
    else:
        eval_output = []
    print(f"Loaded {len(eval_output)} previous results.")
    df = pd.read_json(data_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df_report = df.explode("positions").reset_index(drop=True)
    # Normalize the 'positions' field into a separate dataframe
    df_positions = pd.json_normalize(df_report["positions"])
    # drop rows that have the targets as empty lists
    df_positions = df_positions[df_positions["targets"].apply(len) > 0]
    # --- Stakeholder Clustering ---
    print(
        f"{len(df_positions['stakeholder'].unique())} stakeholders"
        " before clustering..."
    )
    stakeholders = df_positions["stakeholder"].tolist()
    stakeholders = [stakeholder.lower().strip() for stakeholder in stakeholders]
    # does df_max_views.parquet exist?
    anchor_file_path_dir = Path(anchor_file_path).parent
    if (anchor_file_path_dir / "df_max_views.parquet").exists():
        print("Anchor file found.")
        df_max_views = pd.read_parquet(anchor_file_path_dir / "df_max_views.parquet")
    else:
        raise FileNotFoundError("Anchor file not found.")
    
    df_embeddings_map = {}
    for item in items:
        source, model_name = item
        print(f"Loading embeds for model: {model_name}")
        df_embeddings_map[model_name]=torch.load(device_to_embed_map[model_name])

    all_combinations = []
    for r in range(3, len(items) + 1):
        combinations = itertools.combinations(items, r)
        all_combinations.extend(combinations)

    search_grid = {
        "threshold": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "model_names": all_combinations,
        "voting": ["all", "majority", "any"]
    }

    combinations = get_param_combinations(search_grid)
    print(f"Total number of combinations: {len(combinations)}")
    print(f"Combanation head: {combinations[:5]}")
    # read the stakeholder_eval_set_answers.txt file
    stakeholder_eval_set_answers = ""
    with open("stakeholder_eval_set_answers.txt", "r") as f:
        for line in f:
            stakeholder_eval_set_answers += line.strip()

    stakeholder_eval_set_answers = json.loads(stakeholder_eval_set_answers)

    # Positive samples (pairs of items within the same list)
    positive_samples = []
    for sublist in stakeholder_eval_set_answers:
        for i in range(len(sublist)):
            for j in range(i + 1, len(sublist)):
                positive_samples.append((sublist[i], sublist[j]))

    # Negative samples (pairs of items from different sublists)
    negative_samples = []
    for i in range(len(stakeholder_eval_set_answers)):
        for j in range(i + 1, len(stakeholder_eval_set_answers)):
            # Create all possible pairs between sublist[i] and sublist[j]
            for element1 in stakeholder_eval_set_answers[i]:
                for element2 in stakeholder_eval_set_answers[j]:
                    negative_samples.append((element1, element2))

    print(f"Number of positive samples: {len(positive_samples)}")
    print(f"Number of negative samples: {len(negative_samples)}")

    for comb_index, combo in tqdm(enumerate(combinations), desc="Evaluating combinations", total=len(combinations)):
        threshold = combo["threshold"]
        model_names = combo["model_names"]
        voting_method = combo["voting"]
        # if the results already exist, skip the evaluation
        if any(
            result["threshold"] == threshold and result["model_names"] == model_names and result["voting_method"] == voting_method
            for result in eval_output
        ):
            print(f"Skipping combination {comb_index + 1}/{len(combinations)}")
            continue

        df_embeddings = []
        for model_to_use in model_names:
            source, model_name = model_to_use
            df_embeddings.append(df_embeddings_map[model_name])
        
        models, device_list = get_models(model_names)
        print(f"Got device list: {device_list} | Model length: {len(models)}")

        new_stakeholders, stakeholder_index_to_wiki_id, stakeholder_replacement = wiki_anchor(
            stakeholders=stakeholders,
            df_embeddings=df_embeddings,
            device=device,
            device_list=device_list,
            df_max_views=df_max_views,
            models=models,
            voting=voting_method,
            output_dir=None,
            threshold=threshold,
            clustering_method="fast"
        )
        torch.cuda.empty_cache()

        stakeholder_replacement_grouped = defaultdict(list)

        for k,v in stakeholder_replacement.items():
            stakeholder_replacement_grouped[v].append(k)
        
        stakeholder_clusters_final = [
            [stakeholder for stakeholder in cluster] for cluster in stakeholder_replacement_grouped.values()
        ]

        positive_results = []
        for sublist in stakeholder_clusters_final:
            for i in range(len(sublist)):
                for j in range(i + 1, len(sublist)):
                    positive_results.append((sublist[i], sublist[j]))

        positive_samples_set = set(positive_samples)
        negative_samples_set = set(negative_samples)

        # Convert the samples to sorted tuples (to handle unordered pairs)
        positive_samples_set = {tuple(sorted(sample)) for sample in positive_samples}
        negative_samples_set = {tuple(sorted(sample)) for sample in negative_samples}

        # Calculate true positives (TP): positive samples that exist in positive_results
        true_positives_results = [sample for sample in positive_samples_set if tuple(sorted(sample)) in positive_results]

        # Calculate false negatives (FN): positive samples that do not exist in positive_results
        false_negatives_results = [sample for sample in positive_samples_set if tuple(sorted(sample)) not in positive_results]

        # Calculate false positives (FP): negative samples that exist in positive_results
        false_positives_results = [sample for sample in negative_samples_set if tuple(sorted(sample)) in positive_results]

        # Calculate true negatives (TN): negative samples that do not exist in positive_results
        true_negatives_results = [sample for sample in negative_samples_set if tuple(sorted(sample)) not in positive_results]


        true_positives = len(true_positives_results)
        false_negatives = len(false_negatives_results)
        false_positives = len(false_positives_results)
        true_negatives = len(true_negatives_results)

        # Output the results
        print("True Positives (TP):", true_positives)
        print("False Negatives (FN):", false_negatives)
        print("False Positives (FP):", false_positives)
        print("True Negatives (TN):", true_negatives)

        # Calculate the metrics based on TP, FP, TN, FN
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) != 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) != 0 else 0

        # Output the results
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"False Positive Rate (FPR): {fpr:.4f}")
        print(f"Specificity: {specificity:.4f}")

        eval_output.append({
            "threshold": threshold,
            "model_names": model_names,
            "voting_method": voting_method,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "fpr": fpr,
            "specificity": specificity
        })

        with open(eval_output_path, "w") as f:
            json.dump(eval_output, f)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(wiki_anchor_eval)
