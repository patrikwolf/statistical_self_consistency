import ast
import json
from pathlib import Path

import pandas as pd

from datasets import load_dataset


def load_goqa_data(row_indices: list | None = None, countries: list | None = None) -> list[dict]:
    survey_df = load_dataset("Anthropic/llm_global_opinions")["train"].to_pandas()

    if row_indices is None:
        row_indices = survey_df.index.tolist()

    data = []
    for row_idx in row_indices:
        row_info = parse_row(row=survey_df.iloc[row_idx], question_idx=row_idx)

        if countries is not None:
            filter_selections(row_info=row_info, countries=countries)

        # Add to list
        data.append(row_info)

    return data


def parse_row(row: pd.DataFrame, question_idx: int) -> dict:
    # Pre-processing
    selections_raw = row["selections"].removeprefix("defaultdict(<class 'list'>, ").removesuffix(")")

    # Extract information
    row_info = {
        "question": row["question"],
        "question_identifier": f"Q{question_idx}",
        "selections": ast.literal_eval(selections_raw),
        "options": ast.literal_eval(row["options"]),
        "source": row["source"],
    }

    return row_info


def filter_selections(row_info: dict, countries: list) -> None:
    # Filter selection by country list
    filtered_selections = {
        country: values
        for country, values in row_info["selections"].items()
        if country in countries
    }

    # Replace unfiltered with filtered version
    row_info["selections"] = filtered_selections


def load_questions_identifiers() -> dict:
    # Determine base directory
    base_dir = Path(__file__).parent

    # Read file
    questions_path = base_dir / "questions_identifiers.json"
    with questions_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def load_questions_by_country() -> dict:
    # Determine base directory
    base_dir = Path(__file__).parent

    # Read file
    questions_path = base_dir / "questions_by_country.json"
    with questions_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def get_question_indices_for_country(countries: list[str]) -> list[int]:
    questions_by_country = load_questions_by_country()

    question_sets = [
        set(questions_by_country[country]["questions"])
        for country in countries
    ]

    # Compute intersection
    shared_question_indices = set.intersection(*question_sets)

    return sorted(shared_question_indices)


if __name__ == "__main__":
    # Filtering
    row_indices = [0, 1, 2]
    countries = ["Belgium", "Italy"]

    # Load GlobalOpinionQA dataset
    goqa_data = load_goqa_data(row_indices=row_indices, countries=countries)
    print(goqa_data)
    print("***")
    print("First Question")
    print(json.dumps(goqa_data[0], indent=2))
    print("***")

    questions_identifiers = load_questions_identifiers()
    questions_by_country = load_questions_by_country()

    print(json.dumps(questions_identifiers["Q0"], indent=2))
    print("***")
    print(json.dumps(questions_identifiers["Q1"], indent=2))
    print("***")
    print(json.dumps(questions_by_country["Switzerland"], indent=2))
    print("***")

    question_indices = get_question_indices_for_country(countries=countries)

    print(f"Found {len(question_indices)} questions for countries {countries}")
    print(question_indices)
