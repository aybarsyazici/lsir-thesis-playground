from datetime import datetime
from pyexpat import model
import pandas as pd
from param import output
from scipy import cluster
from sentence_transformers import SentenceTransformer, util
from zmq import device
from data_parallel import get_embeddings_distributed_async
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from pathlib import Path
import pickle
import torch
import time
from typing import Literal, Callable, Optional, Tuple, Literal, List, Union
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

def get_multimodel_embeds(models, text_array, device_list):
    """
    Generate embeddings for a text array using multiple models, handling device allocation.
    
    Args:
        models: List of models (SentenceTransformer or HuggingFace models).
        text_array: List of text inputs to encode.
        device_list: List of devices (e.g., "cuda:0", "cuda:1", "cpu") for each model.

    Returns:
        List of embeddings, normalized and moved back to the CPU.
    """
    assert len(models) == len(device_list), "Each model must have a corresponding device in device_list."
    embeds = []
    try:
        for i in range(len(models)):
            # Move model to its designated device
            device = device_list[i]
            models[i] = models[i].to(device)
            
            # Get model name
            model_name = (
                models[i].model_card_data.base_model
                if isinstance(models[i], SentenceTransformer)
                else models[i].name_or_path
            )
            
            # Encode on device
            if model_name in ["intfloat/multilingual-e5-large-instruct", "dunzhang/stella_en_1.5B_v5"]:
                print(f"In Elif | Encoding stakeholders using {model_name}")
                text_array = [
                    f"Instruct: Given the name of the stakeholder as query, find the anchor text that is the most relevant/similar to the stakeholder.\nQuery: {stakeholder}"
                    for stakeholder in text_array
                ]
            else:
                print(f"In else | Encoding stakeholders using {model_name}")
            batch_size = 128 if model_name != "dunzhang/stella_en_1.5B_v5" else 16
            embed_to_append = []
            print(f"Length of text_array: {len(text_array)} | Batch count: {len(text_array) // batch_size}")
            for j in tqdm(range(0, len(text_array), batch_size), desc=f"Encoding using {model_name}"):
                embed_to_append.append(
                    models[i].encode(
                        text_array[j:j+batch_size],
                        show_progress_bar=True,
                        convert_to_tensor=True,
                        device=device  # Explicitly set the device for encoding
                    ).cpu()
                )
            # Go from a list of size batch_count x batch_size x embedding_size to
            # a single tensor of size (batch_count * batch_size) x embedding_size
            embed_to_append = torch.cat(embed_to_append, dim=0)
            # Move embeddings back to CPU and normalize
            embeds.append(util.normalize_embeddings(embed_to_append))
            
            # Move model back to CPU to free VRAM
            models[i] = models[i].to("cpu")
    except torch.OutOfMemoryError as e:
        print("Out of memory error at time: ", datetime.now())
        raise e
    return embeds

def wiki_anchor(
    stakeholders,
    df_embeddings,
    device,
    device_list,
    df_max_views,
    models,
    voting="all",
    threshold=0.15,
    clustering_method="fast",
    top_k = 5,
    output_dir=None,
    event_name="",
):

    batch_size = 64 * 4 if device == "cuda" else 64
    assert voting in ["all", "majority", "any"], f"Invalid voting method: {voting}"
    if output_dir is not None:
        with open(output_dir / f"stakeholders_{event_name}.pkl", "wb") as f:
            pickle.dump(stakeholders, f)
    unique_stakeholders = list(set(stakeholders))
    for i in range(len(models)):
        # check whether model is from HuggingFace or SentenceTransformer
        if isinstance(models[i], AutoModel):
            assert df_embeddings[i].shape[1] == models[i].config.hidden_size, f"Model {i} hidden size={models[i].config.hidden_size} does not match the embedding size={df_embeddings[i].shape[1]}"
        elif isinstance(models[i], SentenceTransformer):
            assert df_embeddings[i].shape[1] == models[i].get_sentence_embedding_dimension(), f"Model {i} hidden size={models[i].get_sentence_embedding_dimension()} does not match the embedding size={df_embeddings[i].shape[1]}"
    
    unique_stakeholder_embeddings = get_multimodel_embeds(models, unique_stakeholders, device_list=device_list)
    # Perform semantic search for each model
    search_results = []
    for i in range(len(models)):
        try:
            search_results.append(
                semantic_search(
                    unique_stakeholder_embeddings[i],  # query embeddings
                    df_embeddings[i],  # database embeddings (can remain on CPU)
                    top_k=top_k,
                    device=device,
                )
            )
        except torch.torch.OutOfMemoryError as e:
            search_results.append(
                semantic_search(
                    unique_stakeholder_embeddings[i],  # query embeddings
                    df_embeddings[i],  # database embeddings (can remain on CPU)
                    top_k=top_k,
                    device=device,
                    corpus_chunk_size=1000
                )
            )
    del unique_stakeholder_embeddings
    if device == "cuda":
        torch.cuda.empty_cache()
    # Process search results to get the best matches above the threshold
    all_results = {}
    stakeholder_index_to_target_item_id = [None] * len(stakeholders)

    for idx, stakeholder in enumerate(unique_stakeholders):
        # Gather best matches and scores for all selected models
        matches = []
        for model_idx in range(len(models)):
            for match in search_results[model_idx][idx]:
                matches.append({
                    "score": match["score"],
                    "corpus_id": match["corpus_id"],
                    "anchor_text": df_max_views.iloc[match["corpus_id"]]["normalized_anchor_text"],
                    "model_idx": model_idx,
                })
        # Group matches by corpus_id
        corpus_id_votes = {}
        for match in matches:
            corpus_id = match["corpus_id"]
            if corpus_id not in corpus_id_votes:
                corpus_id_votes[corpus_id] = []
            corpus_id_votes[corpus_id].append(match["score"])

        # Sort corpus_id_votes by the highest score for each corpus_id (descending order)
        sorted_corpus_ids = sorted(
            corpus_id_votes.items(),
            key=lambda item: max(item[1]),  # Use the highest score for sorting
            reverse=True
        )

        # Determine the winning corpus_id based on voting
        winning_corpus_id = None

        if voting == "all":
            # All models must agree and pass the threshold
            for corpus_id, scores in sorted_corpus_ids:
                if len(scores) == len(models) and all(score >= (1 - threshold) for score in scores):
                    winning_corpus_id = corpus_id
                    break

        elif voting == "majority":
            # Majority of models must agree and pass the threshold
            for corpus_id, scores in sorted_corpus_ids:
                if len(scores) > len(models) // 2 and all(score >= (1 - threshold) for score in scores):
                    winning_corpus_id = corpus_id
                    break

        elif voting == "any":
            # Use the corpus_id with the highest score that passes the threshold
            for corpus_id, scores in sorted_corpus_ids:
                highest_score_for_corpus = max(scores)
                if highest_score_for_corpus >= (1 - threshold):
                    winning_corpus_id = corpus_id
                    break

        # Save the result if a winning corpus_id is found
        if winning_corpus_id is not None:
            best_match_row = df_max_views.iloc[winning_corpus_id]
            for idx_orig_stakeholder, orig_stakeholder in enumerate(stakeholders):
                if orig_stakeholder == stakeholder:
                    assert stakeholder_index_to_target_item_id[idx_orig_stakeholder] is None
                    stakeholder_index_to_target_item_id[idx_orig_stakeholder] = (
                        best_match_row["target_item_id"],
                        best_match_row["target_page_id"],
                    )
        all_results[stakeholder] = matches

    del search_results
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
    # Materialize embeddings for missing stakeholders
    if len(all_wiki_info) > 0:
        missing_stakeholder_embeddings = get_multimodel_embeds(models, missing_stakeholders, device_list=device_list)

        # Materialize embeddings for all wiki info
        all_wiki_info_embeddings = get_multimodel_embeds(models, all_wiki_info, device_list=device_list)

        # Perform semantic search for each model
        search_results = []
        for i in range(len(models)):
            try:
                search_results.append(
                    semantic_search(
                        missing_stakeholder_embeddings[i],  # query embeddings
                        all_wiki_info_embeddings[i],  # database embeddings (can remain on CPU)
                        top_k=top_k,
                        device=device,
                    )
                )
            except torch.torch.OutOfMemoryError as e:
                search_results.append(
                    semantic_search(
                        missing_stakeholder_embeddings[i],  # query embeddings
                        all_wiki_info_embeddings[i],  # database embeddings (can remain on CPU)
                        top_k=top_k,
                        device=device,
                        corpus_chunk_size=1000
                    )
                )
        del missing_stakeholder_embeddings
        del all_wiki_info_embeddings

        missing_stakeholder_matches = {}
        for idx, missing_stakeholder in enumerate(missing_stakeholders):
            # Gather matches from all models
            matches = []
            for model_idx in range(len(models)):
                for match in search_results[model_idx][idx]:
                    matches.append({
                        "score": match["score"],
                        "corpus_id": match["corpus_id"],
                        "model_idx": model_idx,
                    })

            # Group matches by corpus_id
            corpus_id_votes = {}
            for match in matches:
                corpus_id = match["corpus_id"]
                if corpus_id not in corpus_id_votes:
                    corpus_id_votes[corpus_id] = []
                corpus_id_votes[corpus_id].append(match["score"])

            # Sort corpus_id_votes by the highest score for each corpus_id (descending order)
            sorted_corpus_ids = sorted(
                corpus_id_votes.items(),
                key=lambda item: max(item[1]),  # Use the highest score for sorting
                reverse=True
            )

            # Save detailed matches for analysis
            missing_stakeholder_matches[missing_stakeholder] = {
                "all_matches": [
                    (all_wiki_info[corpus_id], score) for corpus_id, score in sorted_corpus_ids
                ],
            }

            # Determine the winning corpus_id based on voting
            winning_corpus_id = None

            if voting == "all":
                # All models must agree and pass the threshold
                for corpus_id, scores in sorted_corpus_ids:
                    if len(scores) == len(models) and all(score >= (1 - threshold) for score in scores):
                        winning_corpus_id = corpus_id
                        break

            elif voting == "majority":
                # Majority of models must agree and pass the threshold
                for corpus_id, scores in sorted_corpus_ids:
                    if len(scores) > len(models) // 2 and all(score >= (1 - threshold) for score in scores):
                        winning_corpus_id = corpus_id
                        break

            elif voting == "any":
                # Use the corpus_id with the highest score that passes the threshold
                for corpus_id, scores in sorted_corpus_ids:
                    highest_score_for_corpus = max(scores)
                    if highest_score_for_corpus >= (1 - threshold):
                        winning_corpus_id = corpus_id
                        break

            # Save the result if a winning corpus_id is found
            if winning_corpus_id is not None:
                best_match_row = all_wiki_info[winning_corpus_id]
                best_item_id = wiki_info_index_to_itemid[winning_corpus_id]
                for stakeholder_index, stakeholder in enumerate(stakeholders):
                    if stakeholder == missing_stakeholder:
                        assert stakeholder_index_to_target_item_id[stakeholder_index] is None, f"Stakeholder {stakeholder} already has a match"
                        stakeholder_index_to_target_item_id[stakeholder_index] = (
                            best_item_id,
                            wiki_info_index_to_page_id[winning_corpus_id],
                        )
        
        del search_results
        del missing_stakeholder_matches


    if output_dir != None:
        with open(output_dir / f"stakeholder_index_to_target_item_id_{event_name}.pkl", "wb") as f:
            pickle.dump(stakeholder_index_to_target_item_id, f)
        with open(output_dir / f"missing_stakeholder_matches_{event_name}.pkl", "wb") as f:
            pickle.dump(missing_stakeholder_matches, f)


    hit_count = sum([1 for item_id in stakeholder_index_to_target_item_id if item_id is not None])
    print(
        f"Hit count: {hit_count} | Hit percentage after wiki: {hit_count / len(stakeholders) * 100:.2f}%"
    )
    del missing_stakeholders

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

    stakeholder_embeddings = get_multimodel_embeds(models, augmented_stakeholders, device_list=device_list)

    stakeholder_cos_scores = [
        embeddings @ embeddings.T for embeddings in stakeholder_embeddings
    ]

    # move all cos_scores to cpu
    stakeholder_cos_scores = [cos_scores.cpu() for cos_scores in stakeholder_cos_scores]

    stakeholder_cos_scores_avg = sum(stakeholder_cos_scores) / len(stakeholder_cos_scores)

    del stakeholder_cos_scores
    del stakeholder_embeddings
    print("Clustering stakeholders/targets threshold" f" {threshold}...")
    # Use Agglomerative Clustering for stakeholders
    if clustering_method == "agglomerative":
        stakeholder_replacement, clusters = agglomerative_clustering(
            items=unique_stakeholders, cos_distances=(1-stakeholder_cos_scores_avg), threshold=float(threshold)
        )
    elif clustering_method == "fast":
        stakeholder_replacement, clusters = fast_clustering(
            cos_scores=stakeholder_cos_scores_avg,
            items=unique_stakeholders,
            threshold=1 - float(threshold),
        )
    else:
        raise ValueError("Invalid clustering method. Choose 'agglomerative' or 'fast'")

    del stakeholder_cos_scores_avg
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

def fast_clustering(items, threshold=0.8, cos_scores: torch.Tensor = None, embeddings: torch.Tensor = None):
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
    if cos_scores is None and embeddings is None:
        raise ValueError("Either cos_scores or embeddings must be provided.")
    print("Fast clustering start")
    start_time = time.time()
    if cos_scores is not None:
        clusters = community_detection(
            cos_scores,
            threshold=float(threshold),
            show_progress_bar=True,
            min_community_size=1,
        )
    else:
        clusters = util.community_detection(
            embeddings,
            threshold=float(threshold),
            min_community_size=1,
            show_progress_bar=True
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

def agglomerative_clustering(items, threshold=0.2, cos_distances: torch.Tensor = None, embeddings: torch.Tensor = None):
    """
    Clusters items using Agglomerative Clustering based on cosine similarity.

    Args:
        items: List of items to cluster (e.g., stakeholders or positions).
        cos_distances: Cosine distances between the items.
        threshold: Distance threshold for clustering.

    Returns:
        A dictionary mapping items to their canonical representative.
    """
    if cos_distances is None and embeddings is None:
        raise ValueError("Either cos_distances or embeddings must be provided.")
    if cos_distances is not None:
        # Apply Agglomerative Clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="precomputed",
            linkage="complete",
        )
        cluster_labels = clustering.fit_predict(cos_distances)
    else:
        # Apply Agglomerative Clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            linkage="complete",
            affinity="cosine",
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

def get_models(model_names: Tuple[Tuple[Literal['hg', 'st'], str, Optional[str]]]) -> List[Union[AutoModel, SentenceTransformer]]:
    """
    Return models from HuggingFace and SentenceTransformer.
    
    Args:
        model_names: List of model names to load, where each model name is a tuple of the library(hg or st), the model name and device.
    
    Returns:
        models: List of models loaded from HuggingFace and SentenceTransformer.
    """
    assert all([model_name[0] in ["hg", "st"] for model_name in model_names]), "Invalid model name"
    assert all(len(model_name) in [2, 3] for model_name in model_names), "Invalid model name"
    
    # Check for the dedicated model 'dunzhang/stella_en_1.5B_v5'
    stella_model_name = "dunzhang/stella_en_1.5B_v5"
    dedicated_model = stella_model_name in [model_name[1] for model_name in model_names]
    
    if len(model_names[0]) == 3:
        device_list = [model_name[2] for model_name in model_names]
    else:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            model_count = len(model_names)
            
            if dedicated_model:
                # Ensure 'dunzhang/stella_en_1.5B_v5' is on its own GPU (e.g., 'cuda:0')
                device_list = [f"cuda:0" if model_name[1] == stella_model_name else None for model_name in model_names]
                # Distribute remaining models across the other GPUs
                remaining_devices = [f"cuda:{i}" for i in range(1, device_count)] * ((model_count - 1) // (device_count - 1)) 
                remaining_devices += [f"cuda:{i}" for i in range(1, (model_count - 1) % (device_count - 1) + 1)]
                
                # Fill in the remaining devices list for models that aren't the dedicated model
                for i, model_name in enumerate(model_names):
                    if model_name[1] != stella_model_name and device_list[i] is None:
                        device_list[i] = remaining_devices.pop(0)
            else:
                # Distribute models across the available GPUs
                device_list = [f"cuda:{i}" for i in range(device_count)] * (model_count // device_count) + [f"cuda:{i}" for i in range(model_count % device_count)]
        else:
            device_list = ["cpu"] * len(model_names)

    models = []
    for model_name, device in zip(model_names, device_list):
        if model_name[0] == "hg":
            models.append(AutoModel.from_pretrained(model_name[1], trust_remote_code=True).to(device))
        elif model_name[0] == "st":
            models.append(SentenceTransformer(model_name[1], trust_remote_code=True, device=device))
    
    return models, device_list


device_to_embed_map = {
    "arkohut/jina-embeddings-v3": Path("/mnt/datasets/dop-position-mining/wiki-anchor/jina_embeddings.pt"),
    "all-mpnet-base-v2": Path("/mnt/datasets/dop-position-mining/wiki-anchor/sentence_transformer_embeddings.pt"),
    "Alibaba-NLP/gte-multilingual-base": Path("/mnt/datasets/dop-position-mining/wiki-anchor/gte_embeddings.pt"),
    "paraphrase-multilingual-mpnet-base-v2": Path("/mnt/datasets/dop-position-mining/wiki-anchor/paraphrase_mpnet.pt"),
    "dunzhang/stella_en_1.5B_v5": Path("/mnt/datasets/dop-position-mining/wiki-anchor/stella_en.pt"),
    "intfloat/multilingual-e5-large-instruct": Path("/mnt/datasets/dop-position-mining/wiki-anchor/e5_embeddings.pt"),
}

def post_process_df(
    df: pd.DataFrame = None,
    data_path: str = "/mnt/datasets/dop-position-mining/generated-reports/"
    "gpt-4o-mini_report_20241009-045126_event1.json",
    anchor_file_path: str = "/mnt/datasets/dop-position-mining/"
    "anchor_target_counts.csv",
    model_names: Tuple[Tuple[Literal['hg', 'st'], str, Optional[str]]] = [("hg", "arkohut/jina-embeddings-v3", "cuda:0"), ("st", "all-mpnet-base-v2", "cuda:1")],
    anchor_voting: str = "all",
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
    models = get_models(model_names)
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
        df_max_views.to_parquet(anchor_file_path_dir / "df_max_views.parquet")

    # Step 1: Embed all `normalized_anchor_text` values in batch for efficiency
    normalized_texts = (
        df_max_views["normalized_anchor_text"].str.lower().str.strip().tolist()
    )
    normalized_texts = [str(text) for text in normalized_texts]

    df_embeddings = []
    for model_name in model_names:
        df_embeddings.append(torch.load(device_to_embed_map[model_name[1]]))

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
            items=positions, embeddings=position_embeddings, threshold=float(positions_threshold)
        )
    elif clustering_method == "fast":
        pos_replacement, pos_clusters = fast_clustering(
            embeddings=position_embeddings, items=positions, threshold=1 - float(positions_threshold)
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
            items=all_predicates, embeddings=predicate_embeddings, thresohld=float(predicate_threshold)
        )
    elif clustering_method == "fast":
        predicate_replacement, predicate_clusters = fast_clustering(
            embeddings=predicate_embeddings,
            items=all_predicates,
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
