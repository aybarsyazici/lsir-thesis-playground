import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import yaml
import os

def show_report_summary(news_limit, model, elapsed_time):
    """
    Display the report generation summary.

    Args:
        - news_limit (number): The number of news used to generate the report.
        - model (str): The model used to generate text.
        - elapsed_time (float): Text generation duration.
    """
    print("----------------" * 7)
    print("Summary:")
    print(f"- {news_limit} news were used to generate the report")
    print(f"- {model} inference took {elapsed_time:.2f} seconds")
    print("----------------" * 7)

def get_author(row):
    """
    Determine the value for the 'author' based on
    the presence of 'author' and the values in the 'source'
    column.

    Args:
        - row (pd.Series): A row of the DataFrame.

    Returns:
        - str: The updated value for the 'actor' column.
    """
    if pd.notnull(row["author"]):
        return f"{row['author']} of {row['source']}"
    else:
        return f"Journalist of {row['source']}"

def save_json(output_dir, model, data, extra_info=""):
    """
    Save a report to a specified directory in JSON format.

    Args:
        - output_dir (str): The file path where the report will be saved.
        - model (str): The model name used to generate the report.
        - data (dict): The data to be saved in the report.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filepath = f"{output_dir}/{model}_{extra_info}_report_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Report saved at: {filepath}")
    return filepath


def generate_gpt_report(
    df: pd.DataFrame = None,
    data_path: str = (
        "/mnt/nlp4sd/aorellan/summer-position-mining/data/preprocessed/event3.json"
    ),
    output_dir: str = ("/home/yazici/playground/new-prompts/output"),
    model: str = "gpt-4o-mini",
    config_path: str = "config.yaml",
    event_name: str = "event_3",
    json_schema_path: str = "report_schema.json",
    openai_key: str = "",
    temperature: float = 0.1,
    prompt: str = "",
    save_report: bool = True,
    extra_info="",
):
    """
    Generate a report for a given news dataset using OpenAI API.

    Args:
        - df (pd.DataFrame): The news dataset to generate the report. If not provided,
                            it will read the dataset from the data_path.
        - data_path (str): The path to the news dataset.
        - output_dir (str): The directory to save the generated reports.
        - model (str): The model to generate the report.
        - config_path (str): The path to the configuration file.
        - json_schema_path (str): The path to the JSON schema file.
        - openai_key (str): OpenAI API key to use GPT models.
        - temperature (float): A value between 0 and 1 that determines the creativity
                                of the chosen model.
        - prompt (str): The prompt to define the model's behavior.
        - save_report (bool): Flag indicating whether the generated report should be
                                saved.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    delimiter = config["prompts"]["delimiter"]

    if len(prompt) > 0:
        system_prompt = prompt
    else:
        system_prompt = config["prompts"]["zero_shot"]

    event_title = config["datasets"]["link_along_events"][event_name]["name"]
    event_description = config["datasets"]["link_along_events"][event_name]["description"]

    if df is None or df.empty:
        df = pd.read_json(data_path)

    outputs = []

    with open(json_schema_path, "r") as f:
        json_schema = json.load(f)

    print(f"Starting to generate report for {len(df)} news...")
    t0 = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        output = {
            "id": row["id"],
            "title": row["title"],
            "author": get_author(row),
            "summary": row.get("summary", ""),
            "body": row["body"],
            "timestamp": row["@timestamp"],
            "positions": [],
        }
        user_prompt = (
            f"Event Title: {event_title}\nEvent Description: {event_description}\n"
            f"{delimiter}\nTitle: {row['title']}\nContent:"
            f" {row['body']}\n{delimiter}"
        )

        try:
            generated_text = generate_text(
                openai_key=openai_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                json_schema=json_schema,
                temperature=temperature,
            )

            if len(generated_text["positions"]) > 0:
                output["political_issue"] = generated_text["political_issue"]
                output["positions"] = generated_text["positions"]
                output["is_about_event"] = generated_text["is_about_event"]
            else:
                continue

        except Exception as e:
            print(f"Error: {e}")
            continue

        outputs.append(output)

    t1 = time.time()
    elapsed_time = t1 - t0
    show_report_summary(len(df), model, elapsed_time)

    if save_report and len(outputs) > 0:
        save_json(
            output_dir=output_dir,
            model=model,
            data=outputs,
            extra_info=extra_info,
        )

    return pd.DataFrame(outputs)


def generate_text(
    openai_key,
    system_prompt,
    user_prompt,
    model,
    json_schema,
    temperature=0.1,
):
    """
    Generate text using OpenAI API,

    Args:
        - system_prompt (str): Prompt that sets the model's behavior.
        - user_prompt (str): The prompt to interact with the model.
        - model (str): The model's name for generating the response.
        - temperature (float): The creativity temperature of the model output.

    Returns:
        - content (str): The generated response from the model.
    """
    client = OpenAI(api_key=openai_key)

    response = client.chat.completions.create(
        model=model,
        response_format=json_schema,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(generate_gpt_report)
