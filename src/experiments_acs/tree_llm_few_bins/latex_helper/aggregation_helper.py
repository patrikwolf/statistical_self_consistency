import json
import numpy as np

from experiments_acs.tree_income_threshold.latex.aggregation_helper import load_joint_llm_prior
from utility.prior_weighted_avg import weighted_avg
from utility.wasserstein_helper import _evaluate_normalized_wasserstein_distance


def build_aggregated_distribution_tree(level_tree: dict, load_joint_llm_priors: bool) -> dict:
    tree = {}

    node_id = 0
    prev_node = None
    for level, level_nodes in level_tree.items():

        # Aggregate nodes at the same level by computing weighted averages of ground truth and LLM predictions.
        if len(level_nodes) == 1:
            level_node = level_nodes[0]

            gt_counts = level_node["ground_truth"]["counts"]
            gt_distribution = (np.array(gt_counts) / sum(gt_counts)).tolist()

            agg_node = {}
            agg_node["id"] = 0
            agg_node["level"] = level
            agg_node["children"] = []
            agg_node["num_of_samples"] = level_node["llm"]["num_of_samples"],
            agg_node["bin_edges"] = level_node["llm"]["bin_edges"],
            # agg_node["aggregated_ground_truth_prior"] = level_node["ground_truth"]["weighted_prior"]
            agg_node["aggregated_gt_distribution"] = gt_distribution
            agg_node["aggregated_avg_llm_distribution"] = level_node["llm"]["avg_bin_distribution_list"]
            agg_node["error_of_aggregated_distribution"] = _evaluate_normalized_wasserstein_distance(
                direct_estimate=gt_distribution,
                aggregated_estimate=level_node["llm"]["avg_bin_distribution_list"]
            )
            agg_node["aggregated_estimation_error_list"] = compute_error_list(
                aggregated_gt_distribution=gt_distribution,
                raw_llm_distribution_list=level_node["llm"]["distribution_list"],
            ) if "distribution_list" in level_node["llm"] else []
        else:
            agg_node = build_level_node_distribution(
                node_id=node_id,
                level_str=level,
                level_nodes=level_nodes,
                load_joint_llm_priors=load_joint_llm_priors,
            )

        # Link to previous level node
        if prev_node is not None:
            prev_node["children"].append(agg_node["id"])

        # Add level node to tree
        tree[node_id] = agg_node
        prev_node = agg_node
        node_id += 1

    return tree


def convert_list_of_distribution_dict_to_list(distribution_dict_list: list[dict]) -> list:
    bins = sorted(
        list(distribution_dict_list[0].keys()),
        key=lambda qid: int(qid.removeprefix("Bin ")),
    )

    distribution_list = []
    for raw_llm_distribution in distribution_dict_list:
        # Convert distribution
        distribution_list.append([raw_llm_distribution[bin] for bin in bins])

    return distribution_list


def compute_error_list(aggregated_gt_distribution: list, raw_llm_distribution_list: list) -> list:
    # Convert to list of lists
    llm_distribution_list = convert_list_of_distribution_dict_to_list(raw_llm_distribution_list)

    # Compute error list
    error_list = [
        _evaluate_normalized_wasserstein_distance(
            direct_estimate=aggregated_gt_distribution,
            aggregated_estimate=llm_distribution,
        )
        for llm_distribution in llm_distribution_list
    ]

    return error_list


def build_aggregated_error_tree(level_tree: dict) -> dict:
    tree = {}

    node_id = 0
    prev_node = None
    for level, level_nodes in level_tree.items():

        # Aggregate errors from nodes at the same level
        if len(level_nodes) == 1:
            level_node = level_nodes[0]

            # Compute ground-truth distribution
            gt_counts = level_node["ground_truth"]["counts"]
            gt_distribution = (np.array(gt_counts) / sum(gt_counts)).tolist()

            # Compute error
            error = _evaluate_normalized_wasserstein_distance(
                direct_estimate=gt_distribution,
                aggregated_estimate=level_node["llm"]["avg_bin_distribution_list"])

            # Create node
            agg_node = {}
            agg_node["id"] = 0
            agg_node["level"] = level
            agg_node["children"] = []
            agg_node["num_of_samples"] = level_node["llm"]["num_of_samples"],
            agg_node["aggregated_ground_truth_prior"] = level_node["ground_truth"]["weighted_prior"]
            agg_node["aggregated_error"] = error
            agg_node["ground_truth_prior_list"] = [level_node["ground_truth"]["weighted_prior"]]
            agg_node["node_wise_error_list"] = [error]
            agg_node["label_list"] = [level_node["llm"]["short_label"]]
        else:
            agg_node = build_level_node_error(
                node_id=node_id,
                level_str=level,
                level_nodes=level_nodes,
            )

        # Link to previous level node
        if prev_node is not None:
            prev_node["children"].append(agg_node["id"])

        # Add level node to tree
        tree[node_id] = agg_node
        prev_node = agg_node
        node_id += 1

    return tree


def build_level_node_distribution(
        node_id: int,
        level_str: str,
        level_nodes: list[dict],
        load_joint_llm_priors: bool,
) -> dict:
    # Level
    level = int(level_str.split("_")[1])
    num_of_samples = level_nodes[0]["llm"]["num_of_samples"]
    bin_edges = level_nodes[0]["llm"]["bin_edges"]

    # Initialize level nodes
    agg_node = {
        "id": node_id,
        "level": level_str,
        "children": [],
        "num_of_samples": num_of_samples,
        "bin_edges": bin_edges,
        "load_joint_llm_priors": load_joint_llm_priors,
        "ground_truth_prior_list": [],
        "loaded_joint_priors": [],
        "gt_distribution_list": [],
        "avg_llm_distribution_list": [],
        "list_of_llm_distribution_lists": [],
    }

    # Collect values from all nodes at this level
    for level_node in level_nodes:
        gt_counts = level_node["ground_truth"]["counts"]
        gt_distribution = (np.array(gt_counts) / sum(gt_counts)).tolist()

        # Load jointly elicited prior from files
        if load_joint_llm_priors:
            loaded_joint_prior = load_joint_llm_prior(filters=level_node["llm"]["filters"],
                                                      model=level_node["llm"]["model"])["normalized_averaged_prior"]
        else:
            loaded_joint_prior = -1

        # Save results
        agg_node["ground_truth_prior_list"].append(level_node["ground_truth"]["weighted_prior"])
        agg_node["gt_distribution_list"].append(gt_distribution)
        agg_node["avg_llm_distribution_list"].append(level_node["llm"]["avg_bin_distribution_list"])
        agg_node["list_of_llm_distribution_lists"].append(
            convert_list_of_distribution_dict_to_list(distribution_dict_list=level_node["llm"]["distribution_list"])
        ) if "distribution_list" in level_node["llm"] else []
        agg_node["loaded_joint_priors"].append(loaded_joint_prior)

    # Assertion for LLM predictions
    num_level_nodes = 2 ** level
    assert len(agg_node["avg_llm_distribution_list"]) == num_level_nodes, (
        f"Dimension mismatch, expected {num_level_nodes} entries in level {level}, but got "
        f"{len(agg_node['avg_llm_distribution_list'])}")

    # Assertion for ground-truth prior --> make sure we have collected all nodes
    assert abs(sum(agg_node["ground_truth_prior_list"]) - 1.0) < 1e-8, (f"Priors on level {level} "
                                                                        f"do not add up to 1.0, got "
                                                                        f"{sum(agg_node['ground_truth_prior_list'])}")

    # Choose pior
    if load_joint_llm_priors:
        # Use jointly elicited priors
        print("  ---> USING LLM-ESTIMATED PRIORS FOR AGGREGATION OF LLM ESTIMATES")
        agg_node["prior_list"] = agg_node["loaded_joint_priors"]
    else:
        # Resort to ground-truth prior from survey
        print("  ---> USING GT PRIORS FOR AGGREGATION OF LLM ESTIMATES")
        agg_node["prior_list"] = agg_node["ground_truth_prior_list"]

    # Assertion for prior
    assert abs(sum(agg_node["prior_list"]) - 1.0) < 1e-8, (f"Priors on level {agg_node.name} do not add up to 1.0, "
                                                           f"sum is {sum(level_node['priors'])}")

    # Weighted average for the level node
    # agg_node["aggregated_ground_truth_prior"] = sum(agg_node["ground_truth_prior_list"])
    agg_node["aggregated_prior"] = sum(agg_node["prior_list"])
    agg_node["aggregated_gt_distribution"] = weighted_avg(values=np.array(agg_node["gt_distribution_list"]),
                                                          priors=agg_node["ground_truth_prior_list"]).tolist()
    agg_node["aggregated_avg_llm_distribution"] = weighted_avg(values=np.array(agg_node["avg_llm_distribution_list"]),
                                                               priors=agg_node["prior_list"]).tolist()
    agg_node["error_of_aggregated_distribution"] = _evaluate_normalized_wasserstein_distance(
        direct_estimate=agg_node["aggregated_gt_distribution"],
        aggregated_estimate=agg_node["aggregated_avg_llm_distribution"])

    if len(agg_node["list_of_llm_distribution_lists"]) > 0:
        agg_node["aggregated_llm_distribution_list"] = weighted_avg(
            values=np.array(agg_node["list_of_llm_distribution_lists"]),
            priors=agg_node["prior_list"]).tolist()
        agg_node["aggregated_estimation_error_list"] = [
            _evaluate_normalized_wasserstein_distance(
                direct_estimate=agg_node["aggregated_gt_distribution"],
                aggregated_estimate=llm_distribution,
            ) for llm_distribution in agg_node["aggregated_llm_distribution_list"]
        ]

    return agg_node


def build_level_node_error(
        node_id: int,
        level_str: str,
        level_nodes: list[dict],
) -> dict:
    # Level
    level = int(level_str.split("_")[1])
    num_of_samples = level_nodes[0]["llm"]["num_of_samples"]

    # Initialize level nodes
    agg_node = {
        "id": node_id,
        "level": level_str,
        "children": [],
        "num_of_samples": num_of_samples,
        "ground_truth_prior_list": [],
        "node_wise_error_list": [],
        "label_list": [],
    }

    for level_node in level_nodes:
        # Compute ground-truth distribution
        gt_counts = level_node["ground_truth"]["counts"]
        gt_distribution = (np.array(gt_counts) / sum(gt_counts)).tolist()

        # Compute error
        error = _evaluate_normalized_wasserstein_distance(
            direct_estimate=gt_distribution,
            aggregated_estimate=level_node["llm"]["avg_bin_distribution_list"])

        # Save results
        agg_node["node_wise_error_list"].append(error)
        agg_node["ground_truth_prior_list"].append(level_node["ground_truth"]["weighted_prior"])
        agg_node["label_list"].append(level_node["llm"]["short_label"])

    # Assertion for errors
    num_level_nodes = 2 ** level
    assert len(agg_node["node_wise_error_list"]) == num_level_nodes, (
        f"Dimension mismatch, expected {num_level_nodes} entries in level {level}, but got "
        f"{len(agg_node['node_wise_error_list'])}")

    # Assertion for ground-truth prior --> make sure we have collected all nodes
    assert abs(sum(agg_node["ground_truth_prior_list"]) - 1.0) < 1e-8, (f"Priors on level {level} "
                                                                        f"do not add up to 1.0, got "
                                                                        f"{sum(agg_node['ground_truth_prior_list'])}")

    # Weighted average
    agg_node["aggregated_ground_truth_prior"] = sum(agg_node["ground_truth_prior_list"])
    agg_node["aggregated_error"] = weighted_avg(values=np.array(agg_node["node_wise_error_list"]),
                                                priors=agg_node["ground_truth_prior_list"]).tolist()

    return agg_node


if __name__ == "__main__":
    level_tree = {
        "level_0": [
            {
                "ground_truth": {
                    "id": 0,
                    "parent_id": "null",
                    "level": "level_0",
                    "label": "",
                    "short_label": "root",
                    "attributes": [],
                    "decisive_attribute": "none",
                    "decisive_value": "none",
                    "counts": [
                        95713198,
                        79284318,
                        81209282,
                        86405237
                    ],
                    "bin_edges": [
                        -11500,
                        1,
                        25000,
                        60000,
                        1849000
                    ],
                    "probability_mass": 1.0,
                    "target_mean": 44314.587999440824,
                    "target_std": 73198.29630578226,
                    "weight_after_filtering": 342612035.0,
                    "weighted_prior": 1.0
                },
                "children": [
                    1,
                    2
                ],
                "llm": {
                    "id": 0,
                    "parent_id": "null",
                    "level": "level_0",
                    "label": "",
                    "short_label": "root",
                    "attributes": [],
                    "decisive_attribute": "none",
                    "decisive_value": "none",
                    "num_of_samples": 3,
                    "avg_bin_distribution_list": [
                        0.05333333333333334,
                        0.28,
                        0.3633333333333333,
                        0.3033333333333333
                    ],
                    "bin_edges": [
                        -11500,
                        1,
                        25000,
                        60000,
                        1849000
                    ]
                }
            }
        ],
        "level_1": [
            {
                "ground_truth": {
                    "id": 1,
                    "parent_id": 0,
                    "level": "level_1",
                    "label": "AGEP = nan \u2013 96",
                    "short_label": "AGEP = nan \u2013 96",
                    "attributes": [
                        "AGEP"
                    ],
                    "decisive_attribute": "AGEP",
                    "decisive_value": [
                        float("nan"),
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26,
                        27,
                        28,
                        29,
                        30,
                        69,
                        70,
                        71,
                        72,
                        73,
                        74,
                        75,
                        76,
                        77,
                        78,
                        79,
                        80,
                        81,
                        82,
                        83,
                        84,
                        85,
                        86,
                        87,
                        88,
                        89,
                        90,
                        91,
                        92,
                        93,
                        94,
                        95,
                        96
                    ],
                    "counts": [
                        80292465,
                        44349541,
                        33250939,
                        20106733
                    ],
                    "bin_edges": [
                        -11500,
                        1,
                        25000,
                        60000,
                        1849000
                    ],
                    "probability_mass": 1.0,
                    "target_mean": 22885.900234207165,
                    "target_std": 46589.97045500427,
                    "weight_after_filtering": 177999678.0,
                    "weighted_prior": 0.5195371435215345
                },
                "children": [],
                "llm": {
                    "id": 1,
                    "parent_id": 0,
                    "level": "level_1",
                    "label": "AGEP = nan \u2013 96",
                    "short_label": "AGEP = nan \u2013 96",
                    "attributes": [
                        "AGEP"
                    ],
                    "decisive_attribute": "AGEP",
                    "decisive_value": [
                        float("nan"),
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26,
                        27,
                        28,
                        29,
                        30,
                        69,
                        70,
                        71,
                        72,
                        73,
                        74,
                        75,
                        76,
                        77,
                        78,
                        79,
                        80,
                        81,
                        82,
                        83,
                        84,
                        85,
                        86,
                        87,
                        88,
                        89,
                        90,
                        91,
                        92,
                        93,
                        94,
                        95,
                        96
                    ],
                    "num_of_samples": 3,
                    "avg_bin_distribution_list": [
                        0.09333333333333334,
                        0.38999999999999996,
                        0.27,
                        0.24666666666666667
                    ],
                    "bin_edges": [
                        -11500,
                        1,
                        25000,
                        60000,
                        1849000
                    ]
                }
            },
            {
                "ground_truth": {
                    "id": 2,
                    "parent_id": 0,
                    "level": "level_1",
                    "label": "AGEP = 31 \u2013 68",
                    "short_label": "AGEP = 31 \u2013 68",
                    "attributes": [
                        "AGEP"
                    ],
                    "decisive_attribute": "AGEP",
                    "decisive_value": [
                        31,
                        32,
                        33,
                        34,
                        35,
                        36,
                        37,
                        38,
                        39,
                        40,
                        41,
                        42,
                        43,
                        44,
                        45,
                        46,
                        47,
                        48,
                        49,
                        50,
                        51,
                        52,
                        53,
                        54,
                        55,
                        56,
                        57,
                        58,
                        59,
                        60,
                        61,
                        62,
                        63,
                        64,
                        65,
                        66,
                        67,
                        68
                    ],
                    "counts": [
                        15420733,
                        34934777,
                        47958343,
                        66298504
                    ],
                    "bin_edges": [
                        -11500,
                        1,
                        25000,
                        60000,
                        1849000
                    ],
                    "probability_mass": 1.0,
                    "target_mean": 67485.99257494381,
                    "target_std": 88153.98908522527,
                    "weight_after_filtering": 164612357.0,
                    "weighted_prior": 0.4804628564784655
                },
                "children": [],
                "llm": {
                    "id": 2,
                    "parent_id": 0,
                    "level": "level_1",
                    "label": "AGEP = 31 \u2013 68",
                    "short_label": "AGEP = 31 \u2013 68",
                    "attributes": [
                        "AGEP"
                    ],
                    "decisive_attribute": "AGEP",
                    "decisive_value": [
                        31,
                        32,
                        33,
                        34,
                        35,
                        36,
                        37,
                        38,
                        39,
                        40,
                        41,
                        42,
                        43,
                        44,
                        45,
                        46,
                        47,
                        48,
                        49,
                        50,
                        51,
                        52,
                        53,
                        54,
                        55,
                        56,
                        57,
                        58,
                        59,
                        60,
                        61,
                        62,
                        63,
                        64,
                        65,
                        66,
                        67,
                        68
                    ],
                    "num_of_samples": 3,
                    "avg_bin_distribution_list": [
                        0.05333333333333334,
                        0.20000000000000004,
                        0.3433333333333333,
                        0.4033333333333333
                    ],
                    "bin_edges": [
                        -11500,
                        1,
                        25000,
                        60000,
                        1849000
                    ]
                }
            }
        ]
    }

    # Load priors
    load_joint_llm_priors = True

    # Build tree
    tree = build_aggregated_distribution_tree(level_tree=level_tree, load_joint_llm_priors=load_joint_llm_priors)
    print(json.dumps(tree, indent=4))
