from pyexpat import model
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

def community_detection(
    cos_scores_full: torch.Tensor,
    threshold: float = 0.75,
    min_community_size: int = 10,
    batch_size: int = 1024,
    show_progress_bar: bool = False,
    device: str = "cpu",
) -> list[list[int]]:
    """
    Function for Fast Community Detection.

    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.

    Args:
        embeddings (torch.Tensor or numpy.ndarray): The input embeddings.
        threshold (float): The threshold for determining if two embeddings are close. Defaults to 0.75.
        min_community_size (int): The minimum size of a community to be considered. Defaults to 10.
        batch_size (int): The batch size for computing cosine similarity scores. Defaults to 1024.
        show_progress_bar (bool): Whether to show a progress bar during computation. Defaults to False.

    Returns:
        List[List[int]]: A list of communities, where each community is represented as a list of indices.
    """
    if not isinstance(cos_scores_full, torch.Tensor):
        cos_scores_full = torch.tensor(cos_scores_full)
    
    cos_scores_full = cos_scores_full.to(device)
    extracted_communities = []

    # Maximum size for community
    min_community_size = min(min_community_size, len(cos_scores_full))
    sort_max_size = min(max(2 * min_community_size, 50), len(cos_scores_full))

    for start_idx in tqdm(
        range(0, len(cos_scores_full), batch_size), desc="Finding clusters", disable=not show_progress_bar
    ):
        # Compute cosine similarity scores
        cos_scores = cos_scores_full[start_idx : start_idx + batch_size] # size: batch_size x len(cos_scores_full)

        # Use a torch-heavy approach if the embeddings are on CUDA, otherwise a loop-heavy one
        if cos_scores_full.device.type in ["cuda", "npu"]:
            # Threshold the cos scores and determine how many close embeddings exist per embedding
            threshold_mask = cos_scores >= threshold
            row_wise_count = threshold_mask.sum(1)

            # Only consider embeddings with enough close other embeddings
            large_enough_mask = row_wise_count >= min_community_size
            if not large_enough_mask.any():
                continue

            row_wise_count = row_wise_count[large_enough_mask]
            cos_scores = cos_scores[large_enough_mask]

            # The max is the largest potential community, so we use that in topk
            k = row_wise_count.max()
            _, top_k_indices = cos_scores.topk(k=k, largest=True)

            # Use the row-wise count to slice the indices
            for count, indices in zip(row_wise_count, top_k_indices):
                extracted_communities.append(indices[:count].tolist())
        else:
            # Minimum size for a community
            top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

            # Filter for rows >= min_threshold
            for i in range(len(top_k_values)):
                if top_k_values[i][-1] >= threshold:
                    # Only check top k most similar entries
                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    # Check if we need to increase sort_max_size
                    while top_val_large[-1] > threshold and sort_max_size < len(cos_scores_full):
                        sort_max_size = min(2 * sort_max_size, len(cos_scores_full))
                        top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    extracted_communities.append(top_idx_large[top_val_large >= threshold].tolist())

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in enumerate(extracted_communities):
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    return unique_communities


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
    models=None,
    models_to_use="both",
    threshold=0.15,
    clustering_method="fast",
    output_dir=None,
    event_name="",
):
    assert len(df_embeddings) == 2, "Two models are required for wiki_anchor"
    assert df_embeddings[0].shape[1] == 1024, "Model 1 should output 1024 dim embeddings"
    assert df_embeddings[1].shape[1] == 768, "Model 2 should output 768 dim embeddings"
    assert models_to_use in ["both", "jina", "st"], "Invalid models_to_use"
    batch_size = 64 * 4 if device == "cuda" else 64
    unique_stakeholders = list(set(stakeholders))
    if models is None:
        models = (
            AutoModel.from_pretrained(
                "arkohut/jina-embeddings-v3", trust_remote_code=True
            ).to(device),
            SentenceTransformer("all-mpnet-base-v2", device=device),
        )
    unique_stakeholder_embeddings = (
        models[0].encode(
            unique_stakeholders,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device,
            batch_size=batch_size,
        ),
        models[1].encode(
            unique_stakeholders,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device,
            batch_size=batch_size,
        )
    )
    search_results = (
        semantic_search(
            unique_stakeholder_embeddings[0],  # query embeddings
            df_embeddings[0],  # database embeddings (can remain on CPU)
            top_k=1,  # Retrieve the top match only, adjust as needed
            device=device,
        ),
        semantic_search(
            unique_stakeholder_embeddings[1],  # query embeddings
            df_embeddings[1],  # database embeddings (can remain on CPU)
            top_k=1,  # Retrieve the top match only, adjust as needed
            device=device,
        )
    )

    del unique_stakeholder_embeddings
    # Process search results to get the best matches above the threshold
    all_results = {}
    stakeholder_index_to_target_item_id = [None] * len(stakeholders)

    for idx in range(len(unique_stakeholders)):
        best_match_model1 = search_results[0][idx][0]
        best_match_model2 = search_results[1][idx][0]
        similarity_score1 = best_match_model1["score"]
        similarity_score2 = best_match_model2["score"]
        best_match_idx = best_match_model1["corpus_id"]
        best_match_row = df_max_views.iloc[best_match_idx]
        stakeholder = unique_stakeholders[idx]
        if models_to_use == "both":
            if best_match_model1["corpus_id"] != best_match_model2["corpus_id"]:
                continue
            if similarity_score1 >= (1 - float(threshold)) and similarity_score2 >= (1 - float(threshold)):
                # Store the best match for the stakeholder
                for idx_orig_stakeholder, orig_stakeholder in enumerate(stakeholders):
                    if orig_stakeholder == stakeholder:
                        assert stakeholder_index_to_target_item_id[idx_orig_stakeholder] is None
                        stakeholder_index_to_target_item_id[idx_orig_stakeholder] = (best_match_row["target_item_id"], best_match_row["target_page_id"])
        elif models_to_use == "jina":
            if similarity_score1 >= (1 - float(threshold)):
                for idx_orig_stakeholder, orig_stakeholder in enumerate(stakeholders):
                    if orig_stakeholder == stakeholder:
                        assert stakeholder_index_to_target_item_id[idx_orig_stakeholder] is None
                        stakeholder_index_to_target_item_id[idx_orig_stakeholder] = (best_match_row["target_item_id"], best_match_row["target_page_id"])
        elif models_to_use == "st":
            if similarity_score2 >= (1 - float(threshold)):
                for idx_orig_stakeholder, orig_stakeholder in enumerate(stakeholders):
                    if orig_stakeholder == stakeholder:
                        assert stakeholder_index_to_target_item_id[idx_orig_stakeholder] is None
                        stakeholder_index_to_target_item_id[idx_orig_stakeholder] = (best_match_row["target_item_id"], best_match_row["target_page_id"])

        all_results[unique_stakeholders[idx]] = (
            best_match_row["normalized_anchor_text"],
            best_match_row["target_page_id"],
            best_match_row["target_item_id"],
            similarity_score1,
            similarity_score2,
        )

    hit_count = sum([1 for item_id in stakeholder_index_to_target_item_id if item_id is not None])
    print(
        f"Hit count: {hit_count} | Hit percentage before wiki: {hit_count / len(stakeholders) * 100:.2f}%"
    )
    unique_wiki_items = set([
        val for val in stakeholder_index_to_target_item_id if val is not None
    ])

    if output_dir is not None:
        with open(output_dir / f"all_results_{event_name}.pkl", "wb") as f:
            pickle.dump(all_results, f)

    # Extend with wikidata info
    all_wiki_info = []
    wiki_info_index_to_itemid = []
    wiki_info_index_to_page_id = []
    itemid_to_wiki_info = {}
    for val in tqdm(unique_wiki_items, desc="Fetching wiki info"):
        target_item_id, target_page_id = val
        wikidata_info = fetch_wikidata_info(target_item_id)
        if wikidata_info:
            itemid_to_wiki_info[target_item_id] = wikidata_info
            labels = wikidata_info["labels"]  # list of labels
            descriptions = (
                wikidata_info["main_label"].lower() + " " + wikidata_info["description"]
            )  # string
            aliases = wikidata_info["aliases"]  # list of aliases
            all_wiki_info += labels + [descriptions] + aliases
            wiki_info_index_to_itemid += [target_item_id] * (len(labels) + len(aliases) + 1)
            wiki_info_index_to_page_id += [target_page_id] * (len(labels) + len(aliases) + 1)
    print(f"Length of wiki corpus: {len(all_wiki_info)}")
    # del unique_wiki_items
    missing_stakeholders = list(set(
        [stakeholder for idx, stakeholder in enumerate(stakeholders) if stakeholder_index_to_target_item_id[idx] is None]
    ))
    print(f"Length of missing stakeholders: {len(missing_stakeholders)}")
    missing_stakeholder_embeddings = (
        models[0].encode(
            missing_stakeholders,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device,
            batch_size=batch_size,
        ),
        models[1].encode(
            missing_stakeholders,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device,
            batch_size=batch_size,
        ),
    )
    all_wiki_info_embeddings = (
        models[0].encode(
            all_wiki_info,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device,
            batch_size=batch_size,
        ),
        models[1].encode(
            all_wiki_info,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=device,
            batch_size=batch_size,
        ),
    )
    # semantic search
    search_results = (
        semantic_search(
            missing_stakeholder_embeddings[0],  # query embeddings
            all_wiki_info_embeddings[0],  # database embeddings (can remain on CPU)
            top_k=1,  # Retrieve the top match only, adjust as needed
            device=device,
        ),
        semantic_search(
            missing_stakeholder_embeddings[1],  # query embeddings
            all_wiki_info_embeddings[1],  # database embeddings (can remain on CPU)
            top_k=1,  # Retrieve the top match only, adjust as needed
            device=device,
        ),
    )

    del missing_stakeholder_embeddings
    del all_wiki_info_embeddings

    missing_stakeholder_matches = {}
    for idx in range(len(missing_stakeholders)):
        best_match_model1 = search_results[0][idx][0]
        best_match_model2 = search_results[1][idx][0]
        similarity_score1 = best_match_model1["score"]
        similarity_score2 = best_match_model2["score"]
        best_match_idx1 = best_match_model1["corpus_id"]
        best_match_idx2 = best_match_model2["corpus_id"]
        missing_stakeholder = missing_stakeholders[idx]
        item_id1 = wiki_info_index_to_itemid[best_match_idx1]
        item_id2 = wiki_info_index_to_itemid[best_match_idx2]
        missing_stakeholder_matches[stakeholder] = (
            similarity_score1,
            all_wiki_info[best_match_idx1],
            itemid_to_wiki_info[item_id1]["main_label"],
            similarity_score2,
            all_wiki_info[best_match_idx2],
            itemid_to_wiki_info[item_id2]["main_label"],
        )
        if models_to_use == "both":
            if item_id1 != item_id2:
                continue
            if similarity_score1 >= (1 - float(threshold)) and similarity_score2 >= (1 - float(threshold)):
                for stakeholder_index, stakeholder in enumerate(stakeholders):
                    if stakeholder == missing_stakeholder:
                        assert stakeholder_index_to_target_item_id[stakeholder_index] is None, f"Stakeholder {stakeholder} already has a match"
                        stakeholder_index_to_target_item_id[stakeholder_index] = (item_id1, wiki_info_index_to_page_id[best_match_idx1])
        elif models_to_use == "jina":
            if similarity_score1 >= (1 - float(threshold)):
                for stakeholder_index, stakeholder in enumerate(stakeholders):
                    if stakeholder == missing_stakeholder:
                        assert stakeholder_index_to_target_item_id[stakeholder_index] is None, f"Stakeholder {stakeholder} already has a match"
                        stakeholder_index_to_target_item_id[stakeholder_index] = (item_id1, wiki_info_index_to_page_id[best_match_idx1])
        elif models_to_use == "st":
            if similarity_score2 >= (1 - float(threshold)):
                for stakeholder_index, stakeholder in enumerate(stakeholders):
                    if stakeholder == missing_stakeholder:
                        assert stakeholder_index_to_target_item_id[stakeholder_index] is None, f"Stakeholder {stakeholder} already has a match"
                        stakeholder_index_to_target_item_id[stakeholder_index] = (item_id2, wiki_info_index_to_page_id[best_match_idx2])

    if output_dir != None:
        with open(output_dir / f"stakeholders_{event_name}.pkl", "wb") as f:
            pickle.dump(stakeholders, f)
        with open(output_dir / f"stakeholder_index_to_target_item_id_{event_name}.pkl", "wb") as f:
            pickle.dump(stakeholder_index_to_target_item_id, f)
        with open(output_dir / f"missing_stakeholder_matches_{event_name}.pkl", "wb") as f:
            pickle.dump(missing_stakeholder_matches, f)


    hit_count = sum([1 for item_id in stakeholder_index_to_target_item_id if item_id is not None])
    print(
        f"Hit count: {hit_count} | Hit percentage after wiki: {hit_count / len(stakeholders) * 100:.2f}%"
    )

    del missing_stakeholders
    del search_results
    del missing_stakeholder_matches

    augmented_stakeholders = []
    unique_stakeholders = list(set(stakeholders))
    for stakeholder in tqdm(
        unique_stakeholders, total=len(unique_stakeholders), desc="Augmenting stakeholders"
    ):
        i = stakeholders.index(stakeholder)
        if stakeholder_index_to_target_item_id[i] is not None:
            stakeholder_itemid = stakeholder_index_to_target_item_id[i][0]
            wikidata_info = itemid_to_wiki_info[stakeholder_itemid]
            # Append label, description, and aliases to stakeholder name
            augmented_text = wikidata_info["main_label"]
            augmented_text += f" ({wikidata_info['description']})"
            if wikidata_info["en_aliases"]:
                augmented_text += (
                    f" | Aliases: {', '.join(wikidata_info['en_aliases'])}"
                )
            augmented_stakeholders.append(augmented_text.lower())
        else:
            augmented_stakeholders.append(stakeholder.lower())  # If no match, keep original

    if output_dir != None:
        with open(output_dir / f"stakeholder_to_wiki_info_{event_name}.pkl", "wb") as f:
            pickle.dump(itemid_to_wiki_info, f)

    del all_wiki_info
    # clear cuda cache
    if device == "cuda":
        torch.cuda.empty_cache()

    stakeholder_embeddings = (
        (models[0].encode(
        augmented_stakeholders,
        device=device,
        show_progress_bar=True,
        convert_to_tensor=True,
        batch_size=batch_size,
        ).cpu()), 
        (models[1].encode(
        augmented_stakeholders,
        device=device,
        show_progress_bar=True,
        convert_to_tensor=True,
        batch_size=batch_size,
        ).cpu())
    )

    if models_to_use == "both":
        stakeholder_embeddings = (
            util.normalize_embeddings(stakeholder_embeddings[0]),
            util.normalize_embeddings(stakeholder_embeddings[1]),
        )

        stakeholder_cos_scores = (
            stakeholder_embeddings[0] @ stakeholder_embeddings[0].T,
            stakeholder_embeddings[1] @ stakeholder_embeddings[1].T,
        )

        del stakeholder_embeddings

        stakeholder_cos_scores = (stakeholder_cos_scores[0] + stakeholder_cos_scores[1]) / 2
    elif models_to_use == "jina":
        stakeholder_embeddings = stakeholder_embeddings[0]
        stakeholder_cos_scores = stakeholder_embeddings @ stakeholder_embeddings.T
        del stakeholder_embeddings
    elif models_to_use == "st":
        stakeholder_embeddings = stakeholder_embeddings[1]
        stakeholder_cos_scores = stakeholder_embeddings @ stakeholder_embeddings.T
        del stakeholder_embeddings

    print("Clustering stakeholders/targets threshold" f" {threshold}...")
    # Use Agglomerative Clustering for stakeholders
    if clustering_method == "agglomerative":
        stakeholder_replacement, clusters = agglomerative_clustering(
            unique_stakeholders, (1-stakeholder_cos_scores), float(threshold)
        )
    elif clustering_method == "fast":
        stakeholder_replacement, clusters = fast_clustering(
            stakeholder_cos_scores,
            unique_stakeholders,
            threshold=1 - float(threshold),
        )
    else:
        raise ValueError("Invalid clustering method. Choose 'agglomerative' or 'fast'")

    del stakeholder_cos_scores
    itemids_per_cluster = []
    clusters_with_multiple_itemids = []
    for stakeholder_ids in clusters:
        stakeholders_in_cluster = [unique_stakeholders[idx] for idx in stakeholder_ids]
        # check if any of the stakeholders in the cluster exist in stakeholder_index_to_target_item_id
        itemids_in_cluster = []
        for stakeholder_in_cluster in stakeholders_in_cluster:
            i = stakeholders.index(stakeholder_in_cluster)
            if stakeholder_index_to_target_item_id[i] is not None:
                itemids_in_cluster.append(stakeholder_index_to_target_item_id[i][0])
        itemids_per_cluster.append(set(itemids_in_cluster))
        # print(f"Cluster: {set(stakeholders_in_cluster)} | Itemids: {set(itemids_in_cluster)}")
        if len(set(itemids_in_cluster)) > 1:
            clusters_with_multiple_itemids.append(
                (set(stakeholders_in_cluster), set(itemids_in_cluster))
            )
        for stakeholder_in_cluster in stakeholders_in_cluster:
            i = stakeholders.index(stakeholder_in_cluster)
            if stakeholder_index_to_target_item_id[i] is not None:
                for orig_stakeholder_id, orig_stakeholder in enumerate(stakeholders):
                    if orig_stakeholder in stakeholders_in_cluster:
                        if stakeholder_index_to_target_item_id[orig_stakeholder_id] is not None and stakeholder_index_to_target_item_id[i] != stakeholder_index_to_target_item_id[orig_stakeholder_id]:
                            # skip the replacement if the stakeholder is already replaced
                            print(f"Skipping replacement for {stakeholder_in_cluster}")
                            continue
                        stakeholder_index_to_target_item_id[orig_stakeholder_id] = stakeholder_index_to_target_item_id[i]
                break
    
    hit_count = sum([1 for item_id in stakeholder_index_to_target_item_id if item_id is not None])
    print(
        f"Hit count: {hit_count} | Hit percentage after clustering: {hit_count / len(stakeholders) * 100:.2f}%"
    )
    for i in range(len(stakeholder_index_to_target_item_id)):
        if stakeholder_index_to_target_item_id[i] is not None:
            wiki_info = itemid_to_wiki_info[stakeholder_index_to_target_item_id[i][0]]
            main_label = wiki_info["main_label"]
            orig_stakeholder = stakeholders[i]
            # copy the original stakeholder to the replacement
            old_replacement = stakeholder_replacement[orig_stakeholder]
            stakeholder_replacement[orig_stakeholder] = main_label

    
    stakeholders = [
        stakeholder_replacement[stakeholder] for stakeholder in stakeholders
    ]

    if output_dir != None:
        with open(output_dir / f"replacement_{event_name}.pkl", "wb") as f:
            pickle.dump(stakeholder_replacement, f)
    if device == "cuda":
        torch.cuda.empty_cache()
    del augmented_stakeholders
    if device == "cuda":
        torch.cuda.empty_cache()
        
    return stakeholders, stakeholder_index_to_target_item_id, stakeholder_replacement


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


def fast_clustering(cos_scores: torch.Tensor, items, threshold=0.8):
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
    clusters = community_detection(
        cos_scores,
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



def agglomerative_clustering(items, cos_distances, threshold=0.2):
    """
    Clusters items using Agglomerative Clustering based on cosine similarity.

    Args:
        items: List of items to cluster (e.g., stakeholders or positions).
        cos_distances: Cosine distances between the items.
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
    cluster_labels = clustering.fit_predict(cos_distances)

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
    models = (
        AutoModel.from_pretrained(
            "arkohut/jina-embeddings-v3", trust_remote_code=True
        ).to(device),
        SentenceTransformer("all-mpnet-base-v2", device=device),
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
        chunk_size = max(1, len(normalized_texts) // 25)
        if (anchor_file_path_dir / "anchor_text_embeddings2.pt").exists():
            jina_embeddings = torch.load(anchor_file_path_dir / "anchor_text_embeddings2.pt")
        else:
            jina_embeddings = []
            # Process each chunk in a loop
            for i in range(0, len(normalized_texts), chunk_size):
                current_texts = normalized_texts[i : i + chunk_size]
                current_embeds = models[0].encode(
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
            for i in range(0, len(normalized_texts), batch_size):
                sentence_transformer_embeddings += models[1].encode(
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

    df_embeddings = [
        torch.stack([torch.tensor(emb) for emb in df_max_views["jina_embeddings"].values]),
        torch.stack([torch.tensor(emb) for emb in df_max_views["sentence_transformer_embeddings"].values]),
    ]

    stakeholders, stakeholder_index_to_wiki_id = wiki_anchor(
        stakeholders,
        df_embeddings,
        device,
        df_max_views,
        models=models,
        threshold=stakeholders_threshold,
        clustering_method=clustering_method,
        output_dir=(Path(output_dir).parent.parent / "temp-files"),
        event_name=output_name + "_stakeholders",
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    df_positions["stakeholder"] = stakeholders
    df_positions["stakeholder_index_to_wiki_id"] = stakeholder_index_to_wiki_id
    print(
        f"{len(df_positions['stakeholder'].unique())} unique"
        " stakeholders after clustering..."
    )

    # --- Position Clustering ---
    positions = df_positions["position"].tolist()

    print(f"{len(positions)} positions before clustering...")

    # Embed the positions
    position_embeddings = models[0].encode(
        positions, device=device, show_progress_bar=True, convert_to_tensor=True, 
        batch_size=batch_size
    ).cpu()
    print("Clustering positions with threshold" f" {positions_threshold}...")
    if clustering_method == "agglomerative":
        # Use Agglomerative Clustering for positions
        pos_replacement, pos_clusters = agglomerative_clustering(
            positions, position_embeddings, float(positions_threshold)
        )
    elif clustering_method == "fast":
        pos_replacement, pos_clusters = fast_clustering(
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
    all_targets, target_index_to_wiki_id = wiki_anchor(
        all_targets,
        df_embeddings,
        device,
        df_max_views,
        models=models,
        threshold=target_threshold,
        clustering_method=clustering_method,
        output_dir=(Path(output_dir).parent.parent / "temp-files"),
        event_name=output_name + "_targets",
    )
    if device == "cuda":
        torch.cuda.empty_cache()
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
    predicate_embeddings = models[0].encode(
        all_predicates, device=device, show_progress_bar=True, convert_to_tensor=True,
        batch_size=batch_size
    ).cpu()
    print(f"Clustering predicates with threshold {predicate_threshold}...")

    if clustering_method == "agglomerative":
        predicate_replacement, predicate_clusters = agglomerative_clustering(
            all_predicates, predicate_embeddings, float(predicate_threshold)
        )
    elif clustering_method == "fast":
        predicate_replacement, predicate_clusters = fast_clustering(
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
