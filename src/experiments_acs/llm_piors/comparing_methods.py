import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config.minimal_experiment_config import MinimalExperimentConfig
from config.prior_computation_config import PriorComputationConfig
from data_loader_acs.data_loader import load_survey_data
from experiments_acs.aggregation_refinement.standard_values.standard_value_loader import load_standard_values
from experiments_acs.filtering.filter_survey_data import filter_survey_data
from experiments_acs.llm_piors.individual_elicitation import generate_filters, individual_prior_elicitation
from experiments_acs.llm_piors.joint_elicitation import joint_prior_elicitation
from utility.directories import get_plot_dir
from utility.plot_style import set_plot_style
from utility.time_helper import get_experiment_timestamp


def create_filters(filter_level: str,
                   attribute_dict: dict,
                   ) -> tuple[list, list]:
    # Decomposition attributes
    standard_values = load_standard_values()
    decomposition_attributes = standard_values["decomposition_attributes"]

    # Filters
    filter_tree = generate_filters(decomposition_attributes=decomposition_attributes, attribute_dict=attribute_dict)
    print(f"Generated {len(filter_tree)} filters")

    # Extract filters for given level
    filter_list = []
    for f in filter_tree:
        if f["level"] == filter_level:
            filter_list.append(f["filters"])
    print(f"Extracted {len(filter_list)} filters for {filter_level}")

    # Extract filter attributes
    filter_attributes = [f.attribute for f in filter_list[0]]

    return filter_list, filter_attributes


def compute_ground_truth_priors(
        survey_df: pd.DataFrame,
        attribute_dict: dict,
        filter_list: list,
        income_col: str,
):
    # Filter data
    ground_truth_priors = []
    for idx, filters in enumerate(filter_list):
        print(f"\nComputing ground-truth priors for filter {idx + 1} / {len(filter_list)}")
        filtered_data, sizes = filter_survey_data(survey_df=survey_df,
                                                  attribute_dict=attribute_dict,
                                                  filters=filters,
                                                  target_attribute=income_col)
        ground_truth_priors.append(sizes["weighted_prior"])

    return ground_truth_priors


def plot_results(
        ground_truth_priors: list,
        individual_normalized_priors: list,
        joint_normalized_priors: list,
        timestamp: str,
        filter_level: str,
        filter_attributes: list,
        model_name: str,
        reasoning_effort: str,
        num_of_samples: int,
        joint_template: str,
):
    # Plot directory
    plot_base_dir = get_plot_dir()
    plot_dir = plot_base_dir / "llm_prior_method_comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot style
    set_plot_style(sans_serif=True)

    # Compute TV distances
    tv_individual = 0.5 * sum(abs(np.array(ground_truth_priors) - np.array(individual_normalized_priors)))
    tv_joint = 0.5 * sum(abs(np.array(ground_truth_priors) - np.array(joint_normalized_priors)))

    title = (r"\huge\bfseries Comparing Method to Elicit Priors from LLMs"
             "\n"
             r"\normalsize "
             f"Level: {filter_level.split("_")[1]}"
             f", filter attributes: {filter_attributes}"
             "\n"
             r"\normalsize "
             f"Model: {model_name}"
             f", reasoning effort: {reasoning_effort}"
             f", number of samples: {num_of_samples}"
             "\n"
             r"\normalsize "
             f"Template for joint elicitation: {joint_template.split("_")[-1].split('.')[0]}"
             "\n"
             r"\normalsize "
             f"TV distance between GT and individual: {tv_individual:.4f}, TV distance between GT and joint: {tv_joint:.4f}"
             )

    # Create figure
    width = max(8.5, 2.5 * (2 ** len(filter_attributes)))
    plt.figure(figsize=(width, 6))

    # Extract data
    x = np.arange(len(individual_normalized_priors))

    # Bar plots
    bar_width = 0.2
    plt.bar(x, ground_truth_priors, width=bar_width, label="GT priors")
    plt.bar(x - bar_width, individual_normalized_priors, width=bar_width, label="individual elicitation")
    plt.bar(x + bar_width, joint_normalized_priors, width=bar_width, label="joint elicitation")

    # Formatting
    plt.title(title)
    plt.xticks(range(len(x)), [(integer + 1) for integer in x])
    plt.grid()

    # Formatting
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_subdir = plot_dir / filter_level
    plot_subdir.mkdir(parents=True, exist_ok=True)
    plot_name = "prior_comparison"
    plt.savefig(plot_subdir / f"{timestamp}_{plot_name}_alternative.png", dpi=300)
    plt.savefig(plot_subdir / f"{timestamp}_{plot_name}_alternative.pdf")

    # Show
    plt.show()


def main(survey_name: str,
         survey_year: int,
         set_nan_to_zero: bool,
         income_col: str,
         model_name: str,
         reasoning_effort: str,
         num_of_samples: int,
         filter_level: str,
         joint_template: str,
         improved_age_desc: bool,
         ):
    # Load survey data
    survey_df, attribute_dict = load_survey_data(year=survey_year, set_nan_to_zero=set_nan_to_zero)

    # Create filters
    filter_list, filter_attributes = create_filters(filter_level=filter_level, attribute_dict=attribute_dict)

    # Ground-truth
    ground_truth_priors = compute_ground_truth_priors(survey_df=survey_df,
                                                      attribute_dict=attribute_dict,
                                                      filter_list=filter_list,
                                                      income_col=income_col,
                                                      )

    # Individual prior elicitation
    cfg_individual = MinimalExperimentConfig(
        survey_name=survey_name,
        survey_year=survey_year,
        set_nan_to_zero=set_nan_to_zero,
        experiment_name="individual_prior_elicitation",
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        prompting_scheme="sociodemographic",
        prompt_reasoning=False,
        num_of_samples=num_of_samples,
        max_number_attempts=10,
        weighted=True,
        log_csv="individual_prior_results.csv",
        llm_prior_results_file="unnormalized_priors.json",
    )

    # Experiment folder
    _, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg_individual.experiment_name)

    # Prior elicitation
    individual_normalized_priors = individual_prior_elicitation(
        cfg=cfg_individual,
        model=None,
        tokenizer=None,
        filter_list=filter_list,
        level=filter_level,
        experiment_folder=experiment_folder,
        timestamp=timestamp,
    )
    print(f"\n[Individual] Normalized (averaged) LLM prior estimates: {individual_normalized_priors}")

    # Joint elicitation
    cfg_joint = PriorComputationConfig(
        experiment_name="joint_prior_elicitation",
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        num_of_samples=num_of_samples,
        max_number_attempts=10,
        llm_prior_results_file="unnormalized_priors.json",
    )

    # Experiment folder
    _, timestamp, experiment_folder = get_experiment_timestamp(experiment_name=cfg_individual.experiment_name)

    # Prior elicitation
    llm_prior_results = joint_prior_elicitation(
        cfg=cfg_joint,
        model=None,
        tokenizer=None,
        filter_list=filter_list,
        level=filter_level,
        experiment_folder=experiment_folder,
        timestamp=timestamp,
        template_filename=joint_template,
        improved_age_desc=improved_age_desc,
    )
    joint_normalized_priors = [item["normalized_averaged_prior"] for item in llm_prior_results["results"]]
    print(f"\nNormalized (averaged) LLM prior estimates: {joint_normalized_priors}")

    # Plotting
    plot_results(
        ground_truth_priors=ground_truth_priors,
        individual_normalized_priors=individual_normalized_priors,
        joint_normalized_priors=joint_normalized_priors,
        timestamp=timestamp,
        filter_level=filter_level,
        filter_attributes=filter_attributes,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        num_of_samples=num_of_samples,
        joint_template=joint_template
    )


if __name__ == "__main__":
    # Parameters
    survey_name = "ACS"
    survey_year = 2024
    set_nan_to_zero = True

    income_col = "PINCP"
    weight_col = "PWGTP"

    model_name = "openai/gpt-5.4"
    # todo: change reasoning effort to "NONE"
    reasoning_effort = "none"
    num_of_samples = 50

    filter_level = "level_3"

    # Prompt template (joint prediction is better than individual prediction)
    # joint_template = "ACS_joint_prior_estimation_v1.txt"
    joint_template = "ACS_joint_prior_estimation_v2.txt"  # --> Template 2 seems to work better than template 1

    # Toggle improved age description
    improved_age_desc = False

    # Run comparison
    main(survey_name=survey_name,
         survey_year=survey_year,
         set_nan_to_zero=set_nan_to_zero,
         income_col=income_col,
         model_name=model_name,
         reasoning_effort=reasoning_effort,
         num_of_samples=num_of_samples,
         filter_level=filter_level,
         joint_template=joint_template,
         improved_age_desc=improved_age_desc,
         )
