from config.goqa_config import GlobalOpinionQAConfig
from data_loader_global_opinion_qa.data_loader import load_goqa_data
from experiments_global_opinion_qa.micro_macro_implicit_thresholded.analysis.implicit_micro_macro import main
from utility.argument_parser import parse_arguments

if __name__ == "__main__":
    # Config
    cfg = GlobalOpinionQAConfig(
        experiment_name="goqa_micro_macro_thresholded",
        model_name="openai/gpt-4o-mini",
        reasoning_effort="none",
        prompting_scheme="sociodemographic",
        num_of_samples=20,
        max_number_attempts=10,
        log_csv="micro_macro_goqa_thresholded.csv"
    )

    # Load GlobalOpinionQA dataset
    countries = ["Germany", "Japan", "France", "United States", "Australia", "Italy", "Spain"]
    row_indices = [55, 136, 573, 839, 1108, 1445, 1512, 1531]

    # Parse arguments
    shard_id, shard, datetime = parse_arguments()
    print(f"Shard ID: {shard_id} ({shard_id + 1} out of {len(row_indices)})")

    # Select question
    row_index = row_indices[shard_id]
    print(f"Question: Q{row_index}")
    survey_data = load_goqa_data(row_indices=[row_index], countries=countries)
    assert len(survey_data) == 1

    # Prompt templates
    direct_template = f"GOQA_Q{row_index}_direct.txt"
    micro_to_macro_template = f"GOQA_Q{row_index}_micro_macro.txt"

    # Run main experiment
    main(
        cfg=cfg,
        shard=shard,
        timestamp=datetime,
        survey_item=survey_data[0],
        direct_template=direct_template,
        micro_to_macro_template=micro_to_macro_template
    )
