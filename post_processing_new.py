import pandas as pd
from scipy import cluster
from sentence_transformers import SentenceTransformer, util
from data_parallel import get_embeddings_distributed_async
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from pathlib import Path
import pickle
import torch
import time
from typing import Literal, Callable
import pycountry
import requests
from tqdm import tqdm
import heapq
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

def get_embedding_openai(text_array, model="text-embedding-3-small", client=None, batch_size=1500, output_dir=None):
    if client is None:
        client = OpenAI(api_key=os.environ["LSIR_OPENAI_PROJ_KEY"])
    embeds = []
    for i in tqdm(range(0, len(text_array), batch_size), desc="Embedding using OpenAI"):
        # if output_dir != None and (output_dir / f"openai_embeddings_part_{i}.pt").exists():
        #     temp_tensor = torch.load(output_dir / f"openai_embeddings_part_{i}.pt")
        #     embeds += temp_tensor.tolist()
        #     del temp_tensor
        #     continue
        batch = text_array[i:i+batch_size]
        # replace empty strings with '<PAD>'
        batch = [text if text != "" else "<PAD>" for text in batch]
        response = client.embeddings.create(input=batch, model=model)
        response_array = response.data
        temp_array = []
        for element in response_array:
            temp_array.append(element.embedding)
        embeds += temp_array
        if output_dir != None:
            temp_tensor = torch.tensor(embeds)
            torch.save(temp_tensor, output_dir / f"openai_embeddings_part_{i}.pt")
            del temp_tensor
        del response
        del response_array
        del temp_array
        del batch
    return torch.tensor(embeds) 

def wiki_anchor(
    stakeholders,
    df_embeddings,
    device,
    df_max_views,
    model=None,
    threshold=0.15,
    clustering_method="fast",
    output_dir=None,
    event_name="",
):
    batch_size = 64 * 4 if device == "cuda" else 64
    unique_stakeholders = list(set(stakeholders))
    if model is None:
        model = AutoModel.from_pretrained(
            "arkohut/jina-embeddings-v3", trust_remote_code=True
        ).to(device)
    # Initial wiki matches
    unique_stakeholder_embeddings = model.encode(
        unique_stakeholders,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
        batch_size=batch_size,
    )
    search_results = semantic_search(
        unique_stakeholder_embeddings,  # query embeddings
        df_embeddings,  # database embeddings (can remain on CPU)
        top_k=1,  # Retrieve the top match only, adjust as needed,
        device=device,
    )
    # Process search results to get the best matches above the threshold
    results = {}
    for idx, result in enumerate(search_results):
        best_match = result[0]  # top-1 match
        similarity_score = best_match["score"]

        if similarity_score >= (1 - float(threshold)):
            best_match_idx = best_match["corpus_id"]
            best_match_row = df_max_views.iloc[best_match_idx]
            stakeholder = unique_stakeholders[idx]

            # Store the best match for the stakeholder
            results[stakeholder] = (
                best_match_row["normalized_anchor_text"],
                best_match_row["target_page_id"],
                best_match_row["target_item_id"],
            )
    # Optional: Print hit percentage
    hit_count = sum([1 for stakeholder in stakeholders if stakeholder in results])
    print(
        f"Hit count: {hit_count} | Hit percentage before wiki: {hit_count / len(stakeholders) * 100:.2f}%"
    )

    # Extend with wikidata info
    all_wiki_info = []
    wiki_info_index_to_stakeholder = []
    stakeholder_to_wiki_info = {}
    for key, value in tqdm(results.items(), desc="Fetching wiki info"):
        _, __, target_item_id = value
        wikidata_info = fetch_wikidata_info(target_item_id)
        if wikidata_info:
            stakeholder_to_wiki_info[key] = wikidata_info
            labels = wikidata_info["labels"]  # list of labels
            descriptions = (
                wikidata_info["main_label"].lower() + " " + wikidata_info["description"]
            )  # string
            aliases = wikidata_info["aliases"]  # list of aliases
            all_wiki_info += labels + [descriptions] + aliases
            wiki_info_index_to_stakeholder += [key] * (len(labels) + len(aliases) + 1)
    print(f"Length of wiki corpus: {len(all_wiki_info)}")

    missing_stakeholders = list(set(unique_stakeholders) - set(results.keys()))
    missing_stakeholder_embeddings = model.encode(
        missing_stakeholders,
        device=device,
        show_progress_bar=True,
        convert_to_tensor=True,
        batch_size=batch_size,
    )
    all_wiki_info_embeddings = model.encode(
        all_wiki_info,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
        batch_size=batch_size,
    )
    # semantic search
    search_results = semantic_search(
        missing_stakeholder_embeddings,  # query embeddings
        all_wiki_info_embeddings,  # database embeddings (can remain on CPU)
        top_k=1,  # Retrieve the top match only, adjust as needed,
        device=device,
    )

    missing_stakeholder_matches = {}
    missing_stakeholder_replacements = {}
    for idx, result in enumerate(search_results):
        best_match = result[0]
        similarity_score = best_match["score"]
        best_match_idx = best_match["corpus_id"]
        missing_stakeholder_matches[missing_stakeholders[idx]] = (
            all_wiki_info[best_match_idx],
            similarity_score,
            wiki_info_index_to_stakeholder[best_match_idx],
        )
        if similarity_score >= (1 - float(threshold)):
            stakeholder = missing_stakeholders[idx]
            best_match_stakeholder = wiki_info_index_to_stakeholder[best_match_idx]
            missing_stakeholder_replacements[stakeholder] = best_match_stakeholder

    if output_dir != None:
        with open(output_dir / f"stakeholders_{event_name}.pkl", "wb") as f:
            pickle.dump(stakeholders, f)

    stakeholders = [
        (
            stakeholder
            if stakeholder not in missing_stakeholder_replacements
            else missing_stakeholder_replacements[stakeholder]
        )
        for stakeholder in stakeholders
    ]
    hit_count = sum([1 for stakeholder in stakeholders if stakeholder in results])
    print(
        f"Hit count: {hit_count} | Hit percentage after wiki: {hit_count / len(stakeholders) * 100:.2f}%"
    )

    augmented_stakeholders = []
    for i, stakeholder in tqdm(
        enumerate(stakeholders), total=len(stakeholders), desc="Augmenting stakeholders"
    ):
        wikidata_info = stakeholder_to_wiki_info.get(stakeholder)
        if wikidata_info is not None:
            # Append label, description, and aliases to stakeholder name
            augmented_text = wikidata_info["main_label"]
            augmented_text += f" ({wikidata_info['description']})"
            if wikidata_info["en_aliases"]:
                augmented_text += (
                    f" | Aliases: {', '.join(wikidata_info['en_aliases'])}"
                )
            augmented_stakeholders.append(augmented_text)
        else:
            augmented_stakeholders.append(stakeholder)  # If no match, keep original

    if output_dir != None:
        with open(output_dir / f"stakeholder_to_wiki_info_{event_name}.pkl", "wb") as f:
            pickle.dump(stakeholder_to_wiki_info, f)
        with open(
            output_dir / f"missing_stakeholder_matches_{event_name}.pkl", "wb"
        ) as f:
            pickle.dump(missing_stakeholder_matches, f)
        with open(output_dir / f"results_{event_name}.pkl", "wb") as f:
            pickle.dump(results, f)

    del missing_stakeholders
    del missing_stakeholder_replacements
    del missing_stakeholder_matches
    del all_wiki_info_embeddings
    del all_wiki_info
    del wiki_info_index_to_stakeholder
    del results
    # clear cuda cache
    if device == "cuda":
        torch.cuda.empty_cache()
    stakeholders = [
        (
            stakeholder
            if stakeholder not in stakeholder_to_wiki_info
            else stakeholder_to_wiki_info[stakeholder]["main_label"]
        )
        for stakeholder in stakeholders
    ]

    stakeholder_embeddings = model.encode(
        augmented_stakeholders,
        device="cuda:2",
        show_progress_bar=True,
        convert_to_tensor=True,
        batch_size=batch_size,
    ).cpu()

    print("Clustering stakeholders/targets threshold" f" {threshold}...")
    # Use Agglomerative Clustering for stakeholders
    if clustering_method == "agglomerative":
        stakeholder_replacement = agglomerative_clustering(
            stakeholders, stakeholder_embeddings, float(threshold)
        )
    elif clustering_method == "fast":
        stakeholder_replacement = fast_clustering(
            stakeholder_embeddings,
            stakeholders,
            threshold=1 - float(threshold),
        )
    else:
        raise ValueError("Invalid clustering method. Choose 'agglomerative' or 'fast'")
    stakeholders = [
        stakeholder_replacement.get(stakeholder, stakeholder)
        for stakeholder in stakeholders
    ]
    if output_dir != None:
        with open(output_dir / f"replacement_{event_name}.pkl", "wb") as f:
            pickle.dump(stakeholder_replacement, f)
    del stakeholder_embeddings
    del stakeholder_replacement
    del augmented_stakeholders
    del stakeholder_to_wiki_info
    if device == "cuda":
        torch.cuda.empty_cache()
    return stakeholders


def semantic_search(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    query_chunk_size: int = 100,
    corpus_chunk_size: int = 500000,
    top_k: int = 10,
    score_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = util.cos_sim,
    device: str = "cuda",
):
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    Args:
        query_embeddings (Tensor): A 2 dimensional tensor with the query embeddings.
        corpus_embeddings (Tensor): A 2 dimensional tensor with the corpus embeddings.
        query_chunk_size (int, optional): Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory. Defaults to 100.
        corpus_chunk_size (int, optional): Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory. Defaults to 500000.
        top_k (int, optional): Retrieve top k matching entries. Defaults to 10.
        score_function (Callable[[Tensor, Tensor], Tensor], optional): Function for computing scores. By default, cosine similarity.

    Returns:
        List[List[Dict[str, Union[int, float]]]]: A list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in tqdm(
        range(0, len(query_embeddings), query_chunk_size), desc="Query chunks"
    ):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities
            query_embed_batch = query_embeddings[
                query_start_idx : query_start_idx + query_chunk_size
            ]
            corpus_embed_batch = corpus_embeddings[
                corpus_start_idx : corpus_start_idx + corpus_chunk_size
            ]

            # move both to device
            query_embed_batch = query_embed_batch.to(device)
            corpus_embed_batch = corpus_embed_batch.to(device)

            cos_scores = score_function(query_embed_batch, corpus_embed_batch)

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k, len(cos_scores[0])),
                dim=1,
                largest=True,
                sorted=False,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                ):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    if len(queries_result_list[query_id]) < top_k:
                        heapq.heappush(
                            queries_result_list[query_id], (score, corpus_id)
                        )  # heaqp tracks the quantity of the first element in the tuple
                    else:
                        heapq.heappushpop(
                            queries_result_list[query_id], (score, corpus_id)
                        )

    # change the data format and sort
    for query_id in range(len(queries_result_list)):
        for doc_itr in range(len(queries_result_list[query_id])):
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {
                "corpus_id": corpus_id,
                "score": score,
            }
        queries_result_list[query_id] = sorted(
            queries_result_list[query_id], key=lambda x: x["score"], reverse=True
        )

    return queries_result_list


def fetch_wikidata_info(
    item_id, languages=["en", "de", "fr", "uk", "es", "ru", "it", "pl", "tr"]
):
    url = f"https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "Q" + str(item_id),
        "format": "json",
        "props": "labels|descriptions|aliases",
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        entity = data.get("entities", {}).get("Q" + str(item_id), {})

        # Extracting relevant fields
        labels = [
            label["value"].lower()
            for label in entity.get("labels", {}).values()
            if label["language"] in languages
        ]
        description = entity.get("descriptions", {}).get("en", {}).get("value", "").lower()
        all_aliases = []
        for alias_array in entity.get("aliases", {}).values():
            for alias in alias_array:
                if alias["language"] in languages:
                    all_aliases.append(alias["value"].lower())
        en_aliases = []
        for alias in entity.get("aliases", {}).get("en", []):
            en_aliases.append(alias["value"].lower())

        return {
            "main_label": entity.get("labels", {}).get("en", {}).get("value", ""),
            "labels": labels,
            "description": description,
            "aliases": all_aliases,
            "en_aliases": en_aliases,
        }
    return None


def fast_clustering(embeddings: torch.Tensor, items, threshold=0.8):
    """
    Use sentence transformer community detection to cluster similar items.
    The device will be the same as the one used to encode the embeddings.

    Args:
        embeddings: Embeddings for the items.
        items: List of items to cluster (e.g., stakeholders or positions).
        threshold: Distance threshold for clustering.

    Returns:
        A dictionary mapping items to their canonical representative.
    """
    print("Fast clustering start")
    start_time = time.time()
    clusters = util.community_detection(
        embeddings,
        threshold=float(threshold),
        show_progress_bar=True,
        min_community_size=1,
    )
    print(f"Clustering done after {time.time() - start_time:.2f} sec")
    # clusters is a list of list of ints, where a list is a community,
    # where each community is represented as a list of indices.
    item_replacement = {}

    for _, cluster in enumerate(clusters):
        items_in_cluster = [items[idx] for idx in cluster]
        canonical_name = items_in_cluster[0]
        for item in items_in_cluster:
            item_replacement[item] = canonical_name

    return item_replacement, clusters


def agglomerative_clustering(items, embeddings, threshold=0.2):
    """
    Clusters items using Agglomerative Clustering based on cosine similarity.

    Args:
        items: List of items to cluster (e.g., stakeholders or positions).
        embeddings: Embeddings for the items.
        threshold: Distance threshold for clustering.

    Returns:
        A dictionary mapping items to their canonical representative.
    """

    # Apply Agglomerative Clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="complete",
    )
    cluster_labels = clustering.fit_predict(embeddings)

    # Map each item to its canonical representative based on cluster labels
    item_replacement = {}
    clusters = []
    for label in np.unique(cluster_labels):
        cluster_items = np.array(items)[cluster_labels == label]
        # add the indexes of the items in the cluster
        clusters.append([
            idx for idx, item in enumerate(items) if cluster_labels[idx] == label
        ])
        # Choose the first item in the cluster as the canonical representative
        # fiind the guy closest to the center

        canonical_name = cluster_items[0]
        for item in cluster_items:
            item_replacement[item] = canonical_name

    return item_replacement, clusters


def search_country(x: str):
    """
    Search for the country code based on the country name.

    Args:
        x: The country name.

    Returns:
        The ISO-3 code for the country.
    """
    try:
        return pycountry.countries.search_fuzzy(x)[0].alpha_3
    except LookupError:
        return None


def post_process_df(
    df: pd.DataFrame = None,
    data_path: str = "/mnt/datasets/dop-position-mining/generated-reports/"
    "gpt-4o-mini_report_20241009-045126_event1.json",
    anchor_file_path: str = "/mnt/datasets/dop-position-mining/"
    "anchor_target_counts.csv",
    positions_threshold: str = 0.2,
    stakeholders_threshold: str = 0.3,
    target_threshold: str = 0.15,
    predicate_threshold: str = 0.05,
    device: str = "cuda",
    output_dir: str = "/mnt/datasets/dop-position-mining/generated-reports/cleaned",
    output_name: str = "cleaned_data",
    clustering_method: Literal["agglomerative", "fast"] = "fast",
):
    """
    Cluster similar positions and stakeholders in the DataFrame based on
    the cosine similarity of their embeddings using clustering.

    Args:
        - data_path: The path to the DataFrame containing the event data(saved as JSON)
        - positions_threshold: The threshold value for clustering similar positions.
        - stakeholders_threshold: The threshold value for clustering similar
        stakeholders.
        - st_model: The SentenceTransformer model to use for embeddings.
        - device: The device to use for embeddings.

    Returns:
        - df_clean (pd.DataFrame): The cleaned DataFrame with clustered positions
                                    and stakeholders.
    """
    print("\nPostprocessing...")
    if df is None:
        df = pd.read_json(data_path)
    print(f"Using device: {device}")
    model = AutoModel.from_pretrained(
        "arkohut/jina-embeddings-v3", trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        "arkohut/jina-embeddings-v3", trust_remote_code=True
    ) 
    batch_size = 64 * 4 if device == "cuda" else 64
    # Explode the 'positions' field if it contains a list
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
        print("Anchor file not found. Generating anchor file...")
        df_anchor = pd.read_csv(anchor_file_path)
        # remove rows with empty string
        df_anchor = df_anchor[df_anchor["normalized_anchor_text"] != ""]
        df_max_views = df_anchor.sort_values(
            by=["normalized_anchor_text", "target_page_views"], ascending=[True, False]
        )
        df_max_views = df_max_views.drop_duplicates(
            subset="normalized_anchor_text", keep="first"
        )

        # Step 1: Embed all `normalized_anchor_text` values in batch for efficiency
        normalized_texts = (
            df_max_views["normalized_anchor_text"].str.lower().str.strip().tolist()
        )
        normalized_texts = [str(text) for text in normalized_texts]
        chunk_size = max(1, len(normalized_texts) // 10)
        if (anchor_file_path_dir / "anchor_text_embeddings2.pt").exists():
            jina_embeddings = torch.load(anchor_file_path_dir / "anchor_text_embeddings2.pt")
        else:
            jina_embeddings = []
            # Process each chunk in a loop
            for i in range(0, len(normalized_texts), chunk_size):
                current_texts = normalized_texts[i : i + chunk_size]
                current_embeds = model.encode(
                    current_texts,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                    device=device,
                )
                jina_embeddings += current_embeds.cpu().tolist()
                del current_embeds
                if device == "cuda":
                    torch.cuda.empty_cache()

            jina_embeddings = torch.tensor(jina_embeddings)

            # save the embeddins to a .pt file
            torch.save(jina_embeddings, anchor_file_path_dir / "anchor_text_embeddings2.pt")
        df_max_views["jina_embeddings"] = [
            emb.to(torch.float32).cpu().numpy() for emb in jina_embeddings
        ]  # Convert to NumPy array
        # clear cuda memory
        del jina_embeddings
        if device == "cuda":
            torch.cuda.empty_cache()

        # if (anchor_file_path_dir / "open_ai_embeddings.pt").exists():
        #     openai_embeddings = torch.load(anchor_file_path_dir / "open_ai_embeddings.pt")
        # else:
        #     open_ai_dir = (anchor_file_path_dir / "open-ai-temp-files")
        #     # mkdir if it doesn't exist
        #     open_ai_dir.mkdir(parents=True, exist_ok=True)
        #     openai_embeddings = get_embedding_openai(
        #         normalized_texts, model="text-embedding-3-large", output_dir=open_ai_dir
        #     )
        #     torch.save(openai_embeddings, anchor_file_path_dir / "open_ai_embeddings.pt")
        # df_max_views["openai_embeddings"] = [
        #     emb.to(torch.float32).cpu().numpy() for emb in openai_embeddings
        # ]
        # del openai_embeddings
        # if device == "cuda":
        #     torch.cuda.empty_cache()
        
        if (anchor_file_path_dir / "sentence_transformer_embeddings.pt").exists():
            sentence_transformer_embeddings = torch.load(
                anchor_file_path_dir / "sentence_transformer_embeddings.pt"
            )
        else:
            sentence_transformer_embeddings = []
            st_model = SentenceTransformer("all-mpnet-base-v2", device="cuda:1")
            for i in range(0, len(normalized_texts), batch_size):
                sentence_transformer_embeddings += st_model.encode(
                    normalized_texts[i : i + batch_size],
                    show_progress_bar=True,
                    convert_to_tensor=True,
                    device="cuda:1",
                ).cpu().tolist()
            sentence_transformer_embeddings = torch.tensor(sentence_transformer_embeddings)
            torch.save(
                sentence_transformer_embeddings,
                anchor_file_path_dir / "sentence_transformer_embeddings.pt",
            )
            del st_model
        df_max_views["sentence_transformer_embeddings"] = [
            emb.to(torch.float32).cpu().numpy() for emb in sentence_transformer_embeddings
        ]
        del sentence_transformer_embeddings
        if device == "cuda":
            torch.cuda.empty_cache()
        # save the dataframe to a parquet file
        df_max_views.to_parquet(anchor_file_path_dir / "df_max_views.parquet")

    df_embeddings = torch.stack(
        [torch.tensor(emb) for emb in df_max_views["jina_embeddings"].values]
    )
    stakeholders = wiki_anchor(
        stakeholders,
        df_embeddings,
        device,
        df_max_views,
        model=model,
        threshold=stakeholders_threshold,
        clustering_method=clustering_method,
        output_dir=(Path(output_dir).parent.parent / "temp-files"),
        event_name=output_name + "_stakeholders",
    )
    df_positions["stakeholder"] = stakeholders
    print(
        f"{len(df_positions['stakeholder'].unique())} unique"
        " stakeholders after clustering..."
    )

    # --- Position Clustering ---
    positions = df_positions["position"].tolist()

    print(f"{len(positions)} positions before clustering...")

    # Embed the positions
    position_embeddings = model.encode(
        positions, device=device, show_progress_bar=True, convert_to_tensor=True, 
        batch_size=batch_size
    ).cpu()
    print("Clustering positions with threshold" f" {positions_threshold}...")
    if clustering_method == "agglomerative":
        # Use Agglomerative Clustering for positions
        pos_replacement = agglomerative_clustering(
            positions, position_embeddings, float(positions_threshold)
        )
    elif clustering_method == "fast":
        pos_replacement = fast_clustering(
            position_embeddings, positions, threshold=1 - float(positions_threshold)
        )
    else:
        raise ValueError("Invalid clustering method. Choose 'agglomerative' or 'fast'")
    print("Position clustering done.")
    # Replace positions in the dataframe with their most similar (canonical) match
    df_positions["position"] = df_positions["position"].apply(
        lambda x: pos_replacement.get(x, x)
    )

    print(
        f"{len(df_positions['position'].unique())} unique positions"
        " after clustering..."
    )

    # --- Targets and predicate clustering ---
    df_positions_target = df_positions.explode("targets").reset_index(drop=False)
    df_positions_target.rename(columns={"index": "position_index"}, inplace=True)
    df_target = pd.json_normalize(df_positions_target["targets"])
    # add df_target columns to df_positions_target
    df_positions_target = pd.concat([df_positions_target, df_target], axis=1)
    # drop targets column
    df_positions_target.drop(columns=["targets"], inplace=True)

    all_targets = df_positions_target["target"].tolist()
    print(
        f"{df_positions_target['target'].nunique()} unique targets before clustering..."
    )
    all_targets = wiki_anchor(
        all_targets,
        df_embeddings,
        device,
        df_max_views,
        model=model,
        threshold=target_threshold,
        clustering_method=clustering_method,
        output_dir=(Path(output_dir).parent.parent / "temp-files"),
        event_name=output_name + "_targets",
    )
    df_positions_target["target"] = all_targets
    print(
        f"{df_positions_target['target'].nunique()} unique targets after clustering..."
    )
    # delete df_embeddings to free up memory
    del df_embeddings
    del df_max_views
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- predicate clustering ---
    all_predicates = df_positions_target["predicate"].tolist()
    print(
        f"{df_positions_target['predicate'].nunique()} \
        unique predicates before clustering..."
    )
    predicate_embeddings = model.encode(
        all_predicates, device=device, show_progress_bar=True, convert_to_tensor=True,
        batch_size=batch_size
    ).cpu()
    print(f"Clustering predicates with threshold {predicate_threshold}...")

    if clustering_method == "agglomerative":
        predicate_replacement = agglomerative_clustering(
            all_predicates, predicate_embeddings, float(predicate_threshold)
        )
    elif clustering_method == "fast":
        predicate_replacement = fast_clustering(
            predicate_embeddings,
            all_predicates,
            threshold=1 - float(predicate_threshold),
        )
    else:
        raise ValueError("Invalid clustering method. Choose 'agglomerative' or 'fast'")
    print("Predicate clustering done.")
    # Replace predicates in the dataframe with their most similar (canonical) match
    df_positions_target["predicate"] = df_positions_target["predicate"].apply(
        lambda x: predicate_replacement.get(x, x)
    )
    print(
        f"{df_positions_target['predicate'].nunique()} \
        unique predicates after clustering..."
    )
    # group the positions by the position index
    # and accumulate the targets and predicates into a list of objects with
    # shape: {'target': target, 'predicate': predicate, 'stance_type': stance_type}
    df_positions_target_grouped = (
        df_positions_target.groupby("position_index")
        .apply(lambda x: x[["predicate", "target", "stance_type"]].to_dict("records"))
        .reset_index()
    )
    df_positions_target_grouped.rename(columns={0: "targets"}, inplace=True)
    assert df_positions_target_grouped.shape[0] == df_positions.shape[0]
    # join the targets column back to df_positions and drop the targets column
    df_positions_joined = pd.merge(
        df_positions,
        df_positions_target_grouped,
        left_index=True,
        right_on="position_index",
    )
    # rename targets_x to old_targets and targets_y to targets
    df_positions_joined.rename(
        columns={"targets_x": "old_targets", "targets_y": "targets"}, inplace=True
    )
    # find rows where position_index and normal index differ
    # set position_index as index
    df_positions_joined.set_index("position_index", inplace=True)
    # --- Concatenation ---
    df_clean = pd.merge(
        df_report, df_positions_joined, left_index=True, right_index=True
    )
    df_clean.drop(columns=["positions"], inplace=True)

    # --- Country Fixing ---
    # print("Starting country searching...")
    # # go through all the country column, do search_fuzzy
    # # and create a new column called ISO-3
    # df_clean["ISO-3"] = df_clean["country"].apply(search_country)
    # print("Country searching done.")
    # If country.code is in columns upper case it
    if "country.code" in df_clean.columns:
        df_clean["country.code"] = df_clean["country.code"].str.upper()

    # Drop duplicates based on 'position' and 'stakeholder' after clustering
    print(f"{df_clean.shape[0]} rows before duplicate drop...")
    df_clean.drop_duplicates(subset=["position", "stakeholder"], inplace=True)
    print(f"{df_clean.shape[0]} rows after duplicate drop...")

    # Save the cleaned DataFrame
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / (output_name + ".json")
    df_clean.to_json(output_file, orient="records")

    with open(output_path / f"position_replacement_{output_name}.pkl", "wb") as f:
        pickle.dump(pos_replacement, f)
    with open(output_path / f"predicate_replacement_{output_name}.pkl", "wb") as f:
        pickle.dump(predicate_replacement, f)

    return df_clean.reset_index(drop=True)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(post_process_df)
