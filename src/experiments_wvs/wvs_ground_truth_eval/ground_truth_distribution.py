import json

from data_loader_wvs.wvs_sav_loader import WvsSavLoader
from file_logging.read_and_write_json import save_as_json
from utility.directories import get_wvs_data_dir
from hugging_face_dataset.create_dataset.create_dataset_helper import compute_answer_distributions, \
    get_country_codes, load_json_dict
from utility.time_helper import get_experiment_timestamp


def get_ground_truth_distributions(
        experiment_folder: str,
        country: str,
        question_identifiers: list[str]
) -> dict:

    # Load data
    print("Loading survey data...")
    wvs_data_dir = get_wvs_data_dir()
    data_loader = WvsSavLoader()
    survey_df, _ = data_loader.load()
    value_label_dir = wvs_data_dir / "wvs_attribute_dict.json"
    value_label_dict = load_json_dict(value_label_dir)
    question_dir = wvs_data_dir / "question_dict_v9_manual.json"
    question_dict = load_json_dict(question_dir)

    # Get country codes
    country_codes = get_country_codes(countries=[country], value_label_dict=value_label_dict)
    assert len(country_codes) == 1, "Expected only one country code"
    country_code = country_codes[0]
    country_handle = "B_COUNTRY"

    # Filter survey by country
    filtered_survey_df = survey_df[survey_df[country_handle] == float(country_code)]
    print(f"Number of samples before country-specific filtering: {len(survey_df)}")
    print(f"Number of samples after country-specific filtering: {len(filtered_survey_df)}")

    # Get ground-truth answer distribution at root-level for all questions
    ground_truth_distributions = {}
    for question_identifier in question_identifiers:
        print(f"Computing ground truth distribution for question: {question_identifier}...")

        # Get answer distribution at root-level, i.e., empty attribute list
        distribution_results_list = compute_answer_distributions(survey_df=filtered_survey_df,
                                                                 question_identifier=question_identifier,
                                                                 question_dict=question_dict,
                                                                 value_label_dict=value_label_dict,
                                                                 demographic_attributes=[])

        # Assert length
        assert len(distribution_results_list) == 1, (f"Expected only one answer distribution for root node, "
                                                     f"but got {len(distribution_results_list)}")
        distribution_results = distribution_results_list[0]

        # Post-processing
        ground_truth_distribution = {
            "encoded_distribution": [],
            "answer_distribution": [],
        }
        answer_encoding = distribution_results["answer_encoding"]
        for key, value in distribution_results["encoded_answer_distribution"].items():
            if key != "nan" and int(key) > 0:
                answer_option = next(item["label"] for item in answer_encoding if item["code"] == key)
                ground_truth_distribution["encoded_distribution"].append(
                    {
                        "key": key,
                        "answer_option": answer_option,
                        "probability": value
                    }
                )
                ground_truth_distribution["answer_distribution"].append(value)

        # Add sum of relevant distribution values
        ground_truth_distribution["sum_of_relevant_probs"] = sum(ground_truth_distribution["answer_distribution"])

        # Add results to dict
        ground_truth_distributions[question_identifier] = ground_truth_distribution

    # Save results to file
    results_path = save_as_json(
        data=ground_truth_distributions,
        experiment=f"{experiment_folder}",
        filename="ground_truth_distributions.json",
    )
    print(f"\nGround-truth results for root saved to {results_path}")

    return ground_truth_distributions


if __name__ == "__main__":
    # Experiment folder
    _, _, experiment_folder = get_experiment_timestamp(experiment_name="wvs_llm_reconstruction")

    # Question identifiers
    question_identifiers = ["Q1", "Q2"]

    # Compute ground-truth distribution
    ground_truth_distributions = get_ground_truth_distributions(
        experiment_folder=experiment_folder,
        country="Canada",
        question_identifiers=question_identifiers,
    )
    print("*" * 80)
    print("Ground-truth distributions for all questions:")
    print(json.dumps(ground_truth_distributions, indent=4))
