import sys

from datetime import datetime
from utility.directories import get_results_dir, get_latex_template_dir
from experiments_acs.tree_ground_truth_combined.generate_combined_tree import generate_combined_tree
from experiments_acs.tree_income_distribution.latex.convert_shards_to_json_tree import convert_shards_to_json_tree
from experiments_acs.tree_income_distribution.latex.convert_shards_to_json_tree import (convert_shards_to_json_tree
                                                                                        as binned_json_converter)


if __name__ == "__main__":
    if len(sys.argv) == 10:
        distr_experiment_name = sys.argv[1]
        distr_datetime = sys.argv[2]
        distr_cluster = sys.argv[3].lower() == "true"
        binned_experiment_name = sys.argv[4]
        binned_datetime = sys.argv[5]
        binned_cluster = sys.argv[6].lower() == "true"
        subsampling_factor = int(sys.argv[7])
        lower_limit = int(sys.argv[8])
        upper_limit = int(sys.argv[9])
    else:
        print("---> Using default paths for testing:")
        distr_experiment_name = "ground_truth_income"
        distr_datetime = "2026-04-16__18-20-00"
        distr_cluster = True
        binned_experiment_name = "ground_truth_few_bins"
        binned_datetime = "2026-04-17__15-30-00"
        binned_cluster = True
        subsampling_factor = 5
        lower_limit = -20_000
        upper_limit = 300_000

    # Convert shards to JSON tree
    distr_num_levels = convert_shards_to_json_tree(
        experiment_name=distr_experiment_name,
        datetime=distr_datetime,
        cluster=distr_cluster,
        llm=False,
    )

    # Convert shards to JSON tree
    binned_num_levels = binned_json_converter(
        experiment_name=binned_experiment_name,
        datetime=binned_datetime,
        cluster=binned_cluster,
        llm=False,
    )

    # Construct paths
    distr_base_path = get_results_dir(experiment_name=distr_experiment_name, cluster=distr_cluster, use_timestamp=True,
                                      timestamp=distr_datetime)
    distr_json_path = distr_base_path / "tree"
    binned_base_path = get_results_dir(experiment_name=binned_experiment_name, cluster=binned_cluster, use_timestamp=True,
                                       timestamp=binned_datetime)
    binned_json_path = binned_base_path / "tree"

    # Combined experiment directory
    combined_name = "gt_combined_tree"
    datetime = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    base_path = get_results_dir(
        experiment_name=combined_name,
        cluster=False,
        use_timestamp=True,
        create_dir=True,
    )
    template_dir = get_latex_template_dir()
    if distr_num_levels <= 3:
        template_path = template_dir / "combined_income_tree/template_comb_tree_two.tex"
    elif distr_num_levels == 4:
        template_path = template_dir / "combined_income_tree/template_comb_tree_three.tex"
    elif distr_num_levels == 5:
        template_path = template_dir / "combined_income_tree/template_comb_tree_four.tex"
    elif distr_num_levels == 6:
        template_path = template_dir / "combined_income_tree/template_comb_tree_five.tex"
    else:
        raise NotImplementedError(f"No LaTeX template for {distr_num_levels} levels implemented!")
    out_path = base_path / "main.tex"

    # Generate TikZ tree from JSON and template
    generate_combined_tree(
        distr_json_path=distr_json_path,
        binned_json_path=binned_json_path,
        template_path=template_path,
        out_path=out_path,
        subsampling_factor=subsampling_factor,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )

    print("Done!")
