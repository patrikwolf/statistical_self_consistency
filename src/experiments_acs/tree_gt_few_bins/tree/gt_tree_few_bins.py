import sys

from datetime import datetime
from experiments_acs.tree_gt_few_bins.tree.generate_tex_distribution_tree import generate_binary_distribution_tree
from experiments_acs.tree_income_distribution.latex.convert_shards_to_json_tree import convert_shards_to_json_tree
from utility.directories import get_results_dir, get_latex_template_dir


if __name__ == "__main__":
    if len(sys.argv) == 5:
        gt_experiment_name = sys.argv[1]
        gt_datetime = sys.argv[2]
        gt_cluster = sys.argv[3].lower() == "true"
        equal_bar_width = sys.argv[4].lower() == "true"
    else:
        print("---> Using default paths for testing:")
        gt_experiment_name = "ground_truth_few_bins"
        gt_datetime = "2026-04-14__09-04-05"
        gt_cluster = True
        equal_bar_width = True

    # Convert shards to JSON tree
    gt_num_levels = convert_shards_to_json_tree(
        experiment_name=gt_experiment_name,
        datetime=gt_datetime,
        cluster=gt_cluster,
        llm=False,
    )

    # Construct paths
    gt_base_path = get_results_dir(experiment_name=gt_experiment_name, cluster=gt_cluster, use_timestamp=True,
                                   timestamp=gt_datetime)
    gt_json_path = gt_base_path / "tree"

    combined_name = "ground_truth_few_bins_tree"
    datetime = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    base_path = get_results_dir(
        experiment_name=combined_name,
        cluster=False,
        use_timestamp=True,
        create_dir=True,
    )
    template_dir = get_latex_template_dir()
    if gt_num_levels <= 3:
        template_path = template_dir / "gt_histogram_tree" / "template_hist_tree_three.tex"
    else:
        raise NotImplementedError(f"No LaTeX template for {gt_num_levels} levels implemented!")
    out_path = base_path / "main.tex"

    # Generate TikZ tree from JSON and template
    generate_binary_distribution_tree(
        gt_json_path=gt_json_path,
        template_path=template_path,
        out_path=out_path,
        equal_bar_width=equal_bar_width,
    )

    print("Done!")
