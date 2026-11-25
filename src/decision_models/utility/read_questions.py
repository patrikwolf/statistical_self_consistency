import json
import copy

from pathlib import Path


def read_filtered_questions() -> dict:
    # Construct directory path
    base_dir = Path(__file__).resolve().parent
    parent_dir = base_dir.parent

    # Read file
    with open(parent_dir / "data_source" / "filtered_augmented.json", "r") as f:
        data = json.load(f)

    return data["questions"]


def filter_questions():
    # Construct directory path
    base_dir = Path(__file__).resolve().parent
    parent_dir = base_dir.parent

    with open(parent_dir / "archive" / "fq.json", "r") as f:
        data = json.load(f)

    # Extract questions
    questions = data["og_questions"]

    # GPT filter
    retain = [0, 8, 10, 13, 17, 20, 21, 23, 26, 32, 33, 38, 42, 44, 46, 48, 52, 69, 75, 81, 83, 85, 87, 89, 91, 93, 95, 97]
    filtered_questions = [question for idx, question in enumerate(questions) if idx in retain]

    # Filtered questions
    """
    for i in range(len(questions)):
        if i not in retain:
            questions[i]["retain"] = False
        else:
            questions[i]["retain"] = True
            print("*" * 80)
            print(questions[i]["title"])
    """

    # Copy metadata from original questions to filtered questions
    filtered_json = copy.deepcopy(data)
    filtered_json.pop("og_questions", None)
    filtered_json["questions"] = filtered_questions

    # Save to JSON
    with open("filtered_questions.json", "w") as f:
        json.dump(filtered_json, f, indent=4)


if __name__ == "__main__":
    # filter_questions()

    # Read questions
    questions = read_filtered_questions()

    print(json.dumps(questions[0], indent=4))
