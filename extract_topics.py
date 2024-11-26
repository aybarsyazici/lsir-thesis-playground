from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    OpenAI,
    PartOfSpeech,
)

try:
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP
except ImportError:
    from hdbscan import HDBSCAN
    from umap import UMAP
import tiktoken
import openai
from bertopic import BERTopic
import numpy as np
from sklearn.metrics.pairwise import (
    cosine_similarity,
)

# from IPython.display import display
import pickle

class SimpleLogger:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def log(self, message: str, *args):
        if self.verbose:
            print(message, *args)


def extract_event_topics(
    df: pd.DataFrame = None,
    data_path: Path = Path(
        "/mnt/datasets/dop-position-mining/generated-reports/\
    \cleaned/report_event1_postprocessed.json"
    ),
    st_model: str = "all-mpnet-base-v2",
    openai_key: str = None,
    output_dir: str = (
        "/mnt/datasets/dop-position-mining/generated-reports/frontend-data"
    ),
    output_name: str = "",
    umap_components: int = 10,
    umap_neighbors: int = 5,
    hdbscan_min_cluster_size: int = 50,
    hdbscan_min_samples: int = 5,
    device: str = "cuda",
    save_output: bool = True,
    verbose: bool = True,
):
    """
    Extract topics from the event data and save the result in the output directory.

    Args:
        -   data_path: Path to the postprocessed event data,
        pd.DataFrame saved in JSON format.
        -   st_model: Sentence Transformer model to use for topic extraction.
        -   topic_representation: Representation to use for topic extraction.
        -   openai_key: OpenAI API key.
        -   output_dir: Output directory to save the extracted topics.
    """
    logger = SimpleLogger(verbose)
    logger.log("Loading event data")
    if df is None:
        df_event = pd.read_json(data_path)
    else:
        df_event = df
    # Filter to those that are only about politics
    # (Has is_about_politics column to True)
    # ruff: noqa: E712
    # check if is_about_event column exists
    if "is_about_politics" in df_event.columns:
        df_event = df_event[df_event["is_about_politics"] == True]
    elif "is_about_event" in df_event.columns:
        df_event = df_event[df_event["is_about_event"] == True]
    else:
        logger.log("No column to filter by politics")

    embedder = SentenceTransformer(st_model, device=device)
    all_positions = df_event["position"].tolist()
    logger.log("Embedding positions")
    # use cuda if available while embedding
    corpus_embeddings = embedder.encode(
        all_positions,
        show_progress_bar=True if verbose else False,
        device=device,
    )

    # KeyBERT
    keybert_model = KeyBERTInspired()

    # Part-of-Speech
    pos_model = PartOfSpeech("en_core_web_sm")

    # MMR
    mmr_model = MaximalMarginalRelevance(diversity=0.3)

    if openai_key is not None and openai_key != "":
        tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        client = openai.OpenAI(api_key=openai_key)

        openai_model = OpenAI(
            client,
            model="gpt-4o-mini",
            delay_in_seconds=2,
            chat=True,
            nr_docs=10,
            doc_length=300,
            tokenizer=tokenizer,
        )
        # All representation models
        representation_model = {
            "OpenAI": openai_model,
            "MMR": mmr_model,
            "POS": pos_model,
            "keybert": keybert_model,
        }

        hierarchical_representation = {
            "Main": openai_model,
        }
    else:
        representation_model = {
            "MMR": mmr_model,
            "POS": pos_model,
            "keybert": keybert_model,
        }

        hierarchical_representation = {
            "Main": keybert_model,
        }

    umap_model = UMAP(
        n_components=umap_components,
        n_neighbors=umap_neighbors,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric="euclidean",
        gen_min_span_tree=True,
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedder,
        calculate_probabilities=True,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
    )
    logger.log("Fitting BERTopic")
    topics, probs = topic_model.fit_transform(
        embeddings=corpus_embeddings,
        documents=all_positions,
    )
    # logger.log the topic_model.get_topic_info() column names
    logger.log("Topic info columns:", topic_model.get_topic_info().columns)
    # outlier row is the row with Topic column == -1
    outlier_row = topic_model.get_topic_info()[
        topic_model.get_topic_info()["Topic"] == -1
    ]
    # We need the 'Count' column of this row, if it doesn't exist, we set it to 0
    outlier_count = outlier_row["Count"].values[0] if not outlier_row.empty else 0
    outlier_percentage = outlier_count / len(all_positions) * 100
    logger.log(
        f"Number of outliers: {outlier_count} | Percentage of"
        f" outliers: {outlier_count / len(all_positions) * 100:.2f}%"
    )
    # display(topic_model.get_topic_info().head())
    logger.log("Reducing outliers")
    # Reduce outliers with pre-calculate embeddings instead
    try:
        new_topics = topic_model.reduce_outliers(
            all_positions,
            topics,
            strategy="embeddings",
            embeddings=corpus_embeddings,
        )

        topic_model.update_topics(
            all_positions,
            topics=new_topics,
            representation_model=representation_model,
        )
    except ValueError:
        # If there are no outliers, just use the original topics
        new_topics = topics

    # create a dictionary of topics to sentences that belong to that topic
    topic_sentences = {}
    for i, topic in enumerate(new_topics):
        if topic not in topic_sentences:
            topic_sentences[topic] = []
        topic_sentences[topic].append(i)

    topic_model.representation_model = hierarchical_representation

    def distance_function(x):
        return 1 - np.clip(cosine_similarity(x), -1, 1)

    logger.log("Extracting hierarchical topics")
    hierarchical_topics = topic_model.hierarchical_topics(
        all_positions,
        distance_function=distance_function,
    )
    hierarchical_topics["sentences_in_topic"] = hierarchical_topics["Topics"].apply(
        lambda x: [topic_sentences[i] for i in x]
    )
    # flatten the list of sentences
    hierarchical_topics["sentences_in_topic"] = hierarchical_topics[
        "sentences_in_topic"
    ].apply(lambda x: [item for sublist in x for item in sublist])
    # create a column with the number of sentences in each topic
    hierarchical_topics["sentence_count"] = hierarchical_topics[
        "sentences_in_topic"
    ].apply(lambda x: len(x))

    logger.log("Calculating 2D UMAP embeddings, for visualizations...")
    reduced_embeddings = UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=umap_neighbors,
        random_state=42,
        min_dist=0.0,
    ).fit_transform(corpus_embeddings)
    if save_output:
        logger.log("Saving all results...")
        # create directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_event.to_json(output_dir / (output_name + ".json"), orient="records")
        hierarchical_topics.to_json(
            output_dir / f"hierarchical_topics_{output_name}.json",
            orient="records",
        )
        topic_model.get_topic_info().to_json(
            output_dir / f"orig_topics_{output_name}.json",
            orient="records",
        )
        # save the topics to a pkl file
        with open(
            output_dir / f"topics_{output_name}.pkl",
            "wb",
        ) as f:
            pickle.dump(new_topics, f)
        # save reduced embeddings to a file
        with open(output_dir / f"reduced_embeddings_{output_name}.pkl", "wb") as f:
            pickle.dump(reduced_embeddings, f)
    logger.log("Done")
    return (
        outlier_count,
        outlier_percentage,
        df_event,
        hierarchical_topics,
        topic_model.get_topic_info(),
        new_topics,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(extract_event_topics)
