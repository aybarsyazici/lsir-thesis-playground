from extract_topics import extract_event_topics
import itertools
from tqdm import tqdm
import torch
from pathlib import Path
import json

search_grid = {
    "umap_args": {
        "n_neighbors": [5, 10, 15, 30, 50, 75, 150],
        "n_components": [10, 20, 30, 60, 90, 200],
    },
    "dbscan_args": {
        "min_cluster_size": [5, 10, 20, 30, 50, 100],
        "min_samples": [5, 10, 20, 30, 50, 100],
    },
}

# Function to generate all combinations of parameters for a specific argument dictionary
def get_param_combinations(params):
    keys, values = zip(*params.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations

def bertopic_gridsearch(
        data_path: str = ("/home/yazici/playground/new-prompts/"
        "output/cleaned/event2multianchor.json"),
        output_dir: str = "/home/yazici/playground/new-prompts/output",
        save_bertopic_output: bool = False,
        save_gridsearch_output: bool = True,
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
    # Output path
    output_dir = Path(output_dir)
    # Create output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # check if output dir/ results.json exist
    if (output_dir / "results_event2multianchor.json").exists():
        with open(output_dir / "results_event2multianchor.json") as f:
            results = json.load(f) # json array
    else:
        results = []

    # Get combinations for umap_args
    umap_combinations = get_param_combinations(search_grid['umap_args'])

    # Get combinations for dbscan_args
    dbscan_combinations = get_param_combinations(search_grid['dbscan_args'])

    # Create final combination of both umap_args and dbscan_args
    final_combinations = [
        {'umap_args': umap, 'dbscan_args': dbscan}
        for umap, dbscan in itertools.product(umap_combinations, dbscan_combinations)
    ]

    # Print number of combinations and a few examples
    print(f"Total combinations: {len(final_combinations)}")
    for combo in final_combinations[:3]:  # Show first 3 combinations as examples
        print(combo)
    
    counter = 0
    for combo in tqdm(final_combinations):
        # Check if this combination results already exist
        if any(
            combo['umap_args'] == res['umap_args'] and combo['dbscan_args'] == res['dbscan_args']
            for res in results
        ):
            print("Skipping existing result")
            continue
        # clean cuda cache
        torch.cuda.empty_cache()
        try:
            outlier_count, outlier_percentage, _, hierarchical_topics, topic_df, __ = extract_event_topics(
                data_path=data_path,
                output_dir=output_dir,
                save_output=save_bertopic_output,
                device="cuda",
                umap_components=combo['umap_args']['n_components'],
                umap_neighbors=combo['umap_args']['n_neighbors'],
                hdbscan_min_cluster_size=combo['dbscan_args']['min_cluster_size'],
                hdbscan_min_samples=combo['dbscan_args']['min_samples'],
                verbose=False,
            )
            results = results + [
                {
                    'umap_args': combo['umap_args'],
                    'dbscan_args': combo['dbscan_args'],
                    'outlier_count': int(outlier_count),
                    'outlier_percentage': float(outlier_percentage),
                    'hierarchical_topic_count': len(hierarchical_topics),
                    'original_topic_count': len(topic_df),
                }
            ]
        except Exception as e:
            print(f"Error with combination {combo}: {e}")
            results = results + [
                {
                    'umap_args': combo['umap_args'],
                    'dbscan_args': combo['dbscan_args'],
                    'outlier_count': -1,
                    'outlier_percentage': -1,
                    'hierarchical_topic_count': -1,
                    'original_topic_count': -1,
                }
            ]
        if not save_gridsearch_output:
            continue
        counter += 1
        if counter % 10 == 0:
            with open(output_dir / "results_event2anchornewest.json", "w") as f:
                json.dump(results, f, indent=4)

    with open(output_dir / "results_event2anchornewest.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(bertopic_gridsearch)
