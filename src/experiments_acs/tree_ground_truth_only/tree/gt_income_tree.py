import sys

from datetime import datetime
from experiments_acs.tree_ground_truth_only.tree.generate_tex_distribution_tree import generate_binary_distribution_tree
from experiments_acs.tree_income_distribution.latex.convert_shards_to_json_tree import convert_shards_to_json_tree
from utility.directories import get_results_dir, get_latex_template_dir


if __name__ == "__main__":
    if len(sys.argv) == 7:
        gt_experiment_name = sys.argv[1]
        gt_datetime = sys.argv[2]
        gt_cluster = sys.argv[3].lower() == "true"
        subsampling_factor = int(sys.argv[4])
        lower_limit = int(sys.argv[5])
        upper_limit = int(sys.argv[6])
    else:
        print("---> Using default paths for testing:")
        gt_experiment_name = "ground_truth_income"
        gt_datetime = "2026-04-13__13-59-11"
        gt_cluster = True
        subsampling_factor = 5
        lower_limit = -20_000
        upper_limit = 300_000

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

    combined_name = "gt_income_distribution_tree"
    datetime = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    base_path = get_results_dir(
        experiment_name=combined_name,
        cluster=False,
        use_timestamp=True,
        create_dir=True,
    )
    template_dir = get_latex_template_dir()
    if gt_num_levels <= 3:
        template_path = template_dir / "distribution_tree/template_variance_tree_two.tex"
    elif gt_num_levels == 4:
        template_path = template_dir / "distribution_tree/template_variance_tree_three.tex"
    elif gt_num_levels == 5:
        template_path = template_dir / "distribution_tree/template_variance_tree_four.tex"
    elif gt_num_levels == 6:
        template_path = template_dir / "distribution_tree/template_variance_tree_five.tex"
    else:
        raise NotImplementedError(f"No LaTeX template for {gt_num_levels} levels implemented!")
    out_path = base_path / "main.tex"

    # Generate TikZ tree from JSON and template
    generate_binary_distribution_tree(
        gt_json_path=gt_json_path,
        template_path=template_path,
        out_path=out_path,
        subsampling_factor=subsampling_factor,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )

    print("Done!")
