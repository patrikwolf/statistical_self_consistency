import json

from hugging_face_dataset.create_dataset.create_dataset_helper import load_json_dict, get_core_question_handles
from file_logging.read_and_write_json import save_as_json
from utility.directories import get_wvs_data_dir
from utility.time_helper import get_experiment_timestamp


def _get_answer_options(question_identifier: str, question_dict: dict, verbose: bool = False) -> list[str]:
    # Find answer options (without negative values)
    answer_options = []
    for option in question_dict[question_identifier]["options"]:
        if int(option["code"]) > 0:
            answer_options.append(option["label"])
        elif verbose:
            print(f"Skipping option: {option['label']}")

    return answer_options


def _load_question_metadata() -> tuple[dict, dict]:
    # Data directory
    wvs_data_dir = get_wvs_data_dir()

    # Load question dict (map question handle to full text questions)
    question_dir = wvs_data_dir / "question_dict_v9_manual.json"
    question_dict = load_json_dict(question_dir)

    # Load question matching dict (match questions from GlobalOpinionQA to question handle in WVS)
    matching_dir = wvs_data_dir / "wvs_questions_matching_complete.json"
    matching_dict = load_json_dict(matching_dir)

    return question_dict, matching_dict


def get_question_answer_list(
        experiment_folder: str,
        question_identifiers: list[str] = None,
        truncate: int = None,
) -> list[dict]:
    # Load metadata
    question_dict, matching_dict = _load_question_metadata()

    # Extract core question handles
    core_question_identifiers = get_core_question_handles(matching_dict=matching_dict)

    # Check if we get a list of desired question identifiers
    if question_identifiers:
        print(f"  ----> WARNING: Returning on the questions specified in {question_identifiers}")
        core_question_identifiers = list(set(core_question_identifiers) & set(question_identifiers))

    # Extend list with question and answer options
    question_answer_list = []
    for question_identifier in core_question_identifiers:
        question_answer_list.append(
            {
                "question_identifier": question_identifier,
                "question": question_dict[question_identifier]["question"],
                "answer_options": _get_answer_options(question_identifier=question_identifier, question_dict=question_dict)
            }
        )

    # Truncate
    if truncate:
        question_answer_list = question_answer_list[:truncate]
        print("*" * 80)
        print(f"  ---> WARNING: Truncated question-answer list to {truncate} items!")
        print("*" * 80)

    # Save to file
    results_path = save_as_json(
        data=question_answer_list,
        experiment=experiment_folder,
        filename="question_answer_list.json",
    )
    print(f"\nQuestion-answer list saved to {results_path}")

    return question_answer_list


if __name__ == "__main__":
    # Experiment folder
    _, _, experiment_folder = get_experiment_timestamp(experiment_name="wvs_llm_estimates")

    # Get question answer list
    question_identifier_list = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    question_answer_list = get_question_answer_list(experiment_folder=experiment_folder, truncate=3)
    print(json.dumps(question_answer_list, indent=2))
