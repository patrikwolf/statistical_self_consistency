from config.prior_computation_config import PriorComputationConfig
from data_loader_acs.data_loader import load_original_attribute_dict
from experiments_acs.aggregation_refinement.split_selection.split_helper import load_decomposition_dict
from experiments_acs.aggregation_refinement.standard_values.standard_value_loader import load_standard_values
from experiments_acs.llm_piors.filter_helper import generate_filters
from experiments_acs.llm_piors.joint_elicitation import joint_prior_elicitation
from file_logging.read_and_write_json import save_as_json
from utility.hyperparameters import assert_decomposition_validity
from utility.time_helper import get_experiment_timestamp


def main(cfg: PriorComputationConfig,
         experiment_folder: str,
         timestamp: str,
         levels: list[str],
         list_of_filters_in_tree: list,
         template_filename: str,
         improved_age_desc: bool
         ) -> None:

    # Initialize results
    results_across_levels = []

    # Iterate over all levels
    for level in levels:
        # Extract filters for given level
        list_of_filters_in_level = []
        for f in list_of_filters_in_tree:
            if f["level"] == level:
                list_of_filters_in_level.append(f["filters"])

        print(f"\nExtracted {len(list_of_filters_in_level)} filters for {level}")

        # Change log file name
        cfg.summary_results_file = f"{level}_summary_results.json"
        cfg.llm_prior_results_file = f"{level}_unnormalized_priors.json"

        # Prior elicitation for level
        llm_prior_results = joint_prior_elicitation(
            cfg=cfg,
            model=None,
            tokenizer=None,
            filter_list=list_of_filters_in_level,
            level=level,
            experiment_folder=experiment_folder,
            timestamp=timestamp,
            template_filename=template_filename,
            improved_age_desc=improved_age_desc,
        )

        # Store LLM prior results
        for res in llm_prior_results["results"]:
            results_across_levels.append(
                {
                    "model": llm_prior_results["model"],
                    "num_of_samples": llm_prior_results["num_of_samples"],
                    "filters": res["filters"],
                    "normalized_averaged_prior": res["normalized_averaged_prior"],
                }
            )

    # Save results to JSON file
    results_path = save_as_json(
        data=results_across_levels,
        experiment=experiment_folder,
        filename="final_summary_results.json",
        timestamp=False)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    # todo: change parameters (model, splits, ...)

    # Model
    model = "anthropic/claude-sonnet-4.6"
    reasoning_effort = "none"
    # model = "qwen/qwen3.5-27b"

    # Decomposition
    settings = ["standard_split", "permutations", "splits"]
    setting_idx = 2
    print("*" * 80)
    print(f"Using setting: {settings[setting_idx]}")
    if settings[setting_idx] == "standard_split":
        standard_values = load_standard_values()
        decomposition_attributes = standard_values["decomposition_attributes"]
    elif settings[setting_idx] == "permutations":
        standard_values = load_standard_values()
        standard_decomposition_attributes = standard_values["decomposition_attributes"]

        # Permute decomposition
        # level_1: AGEP
        # level_2: COW
        # level_3: WKHP
        # level_4: OCCP
        decomposition_attributes = {
            "level_1": standard_decomposition_attributes["level_1"],
            "level_2": standard_decomposition_attributes["level_2"],
            "level_3": standard_decomposition_attributes["level_3"],
            "level_4": standard_decomposition_attributes["level_4"],
        }
    elif settings[setting_idx] == "splits":
        split_name = "COW_MAR_ESR_AGEP"
        decomposition_attributes = load_decomposition_dict(split_name=split_name)
    else:
        raise ValueError(f"Unknown setting: {setting_idx}")

    print("*" * 80)
    print([decomposition_attributes[level]["attribute"] for level in decomposition_attributes.keys()])
    print("*" * 80)

    # Configuration
    cfg = PriorComputationConfig(
        experiment_name="joint_prior_elicitation",
        model_name=model,
        reasoning_effort=reasoning_effort,
        # todo: increase to 50
        num_of_samples=50,
        max_number_attempts=10,
        summary_results_file="prior_summary_results.json",
        llm_prior_results_file="unnormalized_priors.json",
    )

    # Experiment folder
    date, time, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name)

    # Load survey dataset
    attribute_dict = load_original_attribute_dict()

    # Assert validity of decomposition attributes
    assert_decomposition_validity(decomposition_attributes=decomposition_attributes, attribute_dict=attribute_dict)

    # Extract levels
    levels = list(decomposition_attributes.keys())

    # Filters
    list_of_filters_in_tree = generate_filters(decomposition_attributes=decomposition_attributes,
                                               attribute_dict=attribute_dict)
    print(f"Generated {len(list_of_filters_in_tree)} filters in total")

    # Prompt template
    template_filename = "ACS_joint_prior_estimation_v2.txt"

    # Toggle improved age description
    improved_age_desc = False

    # Run main analysis
    main(cfg=cfg,
         experiment_folder=experiment_folder,
         timestamp=f"{date}__{time}",
         levels=levels,
         list_of_filters_in_tree=list_of_filters_in_tree,
         template_filename=template_filename,
         improved_age_desc=improved_age_desc,
         )
