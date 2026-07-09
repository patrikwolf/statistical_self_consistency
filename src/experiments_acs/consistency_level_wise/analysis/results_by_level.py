import time
import json

from experiments_acs.consistency_level_wise.helper.compute_aggregation import compute_all_aggregated_estimates
from experiments_acs.consistency_level_wise.helper.evaluation import evaluate_aggregated_results
from experiments_acs.consistency_level_wise.helper.llm_estimates import get_all_llm_estimates
from experiments_acs.consistency_level_wise.helper.build_tree_nodes import get_attribute_combinations
from config.model_config import ModelConfig
from experiments_acs.filtering.filter_definitions import extend_generic_filter
from data_loader_acs.data_loader import load_original_attribute_dict
from data_loader_acs.value_map import get_value_map
from experiments_acs.filtering.filter import Filter
from utility.directories import get_results_dir
from utility.time_helper import get_experiment_timestamp


def main(
        cfg: ModelConfig,
        binary_splits: dict,
        bin_edges: list,
) -> tuple[dict, str]:
    print(f"\nModel: {cfg.model_name}\n")

    # Start timer
    start_time = time.time()

    # Experiment folder
    _, _, experiment_folder = get_experiment_timestamp(experiment_name=cfg.experiment_name)

    # Save config
    experiment_path = get_results_dir(experiment_name=experiment_folder, cluster=False, use_timestamp=False,
                                      create_dir=True)
    config_path = experiment_path / "config.json"
    cfg.save_json(path=config_path)

    # Combinations of attribute splits
    combination_dict = get_attribute_combinations(experiment_folder=experiment_folder, splits=binary_splits)

    # Sample LLM estimate for all combinations
    llm_estimates_list = get_all_llm_estimates(
        cfg=cfg,
        experiment_folder=experiment_folder,
        combination_dict=combination_dict,
        bin_edges=bin_edges,
    )

    # Compute all aggregated estimates and store them
    aggregation_results = compute_all_aggregated_estimates(
        experiment_folder=experiment_folder,
        llm_estimate_dict=llm_estimates_list,
        combination_dict=combination_dict,
    )

    # Evaluate aggregated versus direct estimate (Wasserstein)
    sanity_check_results = evaluate_aggregated_results(
        experiment_folder=experiment_folder,
        llm_estimates_list=llm_estimates_list,
        aggregation_results=aggregation_results,
    )

    # Timer
    duration_sec = time.time() - start_time
    print(f"\nTIMER: Script finished after {duration_sec:.0f} seconds (= {duration_sec / 60:.1f} minutes).")

    return sanity_check_results, experiment_folder


if __name__ == "__main__":
    # Load attribute dict
    attribute_dict = load_original_attribute_dict()

    # Binary split
    binary_splits = {
        "AGEP": [
            Filter(
                attribute="AGEP",
                description="age",
                values=[float("nan"), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        23, 24, 25, 26, 27, 28, 29, 30, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96],
                value_map=get_value_map(attribute="AGEP", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            ),
            Filter(
                attribute="AGEP",
                description="age",
                values=[31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68],
                value_map=get_value_map(attribute="AGEP", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            )
        ],
        "COW": [
            Filter(
                attribute="COW",
                description="class of worker",
                values=[8, 9, float("nan")],
                value_map=get_value_map(attribute="COW", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            ),
            Filter(
                attribute="COW",
                description="class of worker",
                values=[1, 2, 3, 4, 5, 6, 7],
                value_map=get_value_map(attribute="COW", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            )
        ],
        "WKHP": [
            Filter(
                attribute="WKHP",
                description="usual number of hours worked per week",
                values=[float("nan"), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                value_map=get_value_map(attribute="WKHP", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            ),
            Filter(
                attribute="WKHP",
                description="usual number of hours worked per week",
                values=[40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                        88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
                value_map=get_value_map(attribute="WKHP", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            ),
        ],
        "OCCP": [
            Filter(
                attribute="OCCP",
                description="occupation",
                values=[float("nan"), 9920, 3960, 4150, 4055, 2723, 4140, 4130, 4420, 4720, 2350, 4030, 4622, 4160,
                        4110, 4640, 4600, 4435, 4020, 4120, 4230, 3646, 8950, 5320, 3946, 9640, 9350, 5300, 8300, 4940,
                        6040, 9645, 3601, 7610, 9610, 2545, 7800, 2633, 6600, 2440, 5900, 6050, 3430, 4521, 8310, 4350,
                        4540, 4900, 3602, 5400, 8320, 4251, 7840, 7850, 4655, 5160, 9620, 4220, 9365, 9415, 4255, 9630,
                        3648, 8350, 8800, 5020, 5510, 8335, 9110, 4040, 3940, 7830, 3603, 2300, 3647, 3605, 8710, 4510,
                        3649, 3640, 4500, 4010, 9142, 4522, 4621, 8540, 5310, 8365, 3630, 5850, 7260, 3424, 5820, 5810,
                        2740, 7810, 3422, 3421, 8256, 3645, 8850, 4461, 9720, 5860, 2060, 4740, 4525, 3620, 6120, 7510,
                        8530, 5260, 5610, 6410, 8910, 4760, 8465, 4530, 9600, 3401],
                value_map=get_value_map(attribute="OCCP", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            ),
            Filter(
                attribute="OCCP",
                description="occupation",
                values=[10, 20, 40, 51, 52, 60, 101, 102, 110, 120, 135, 136, 137, 140, 150, 160, 205, 220, 230, 300,
                        310, 335, 340, 350, 360, 410, 420, 425, 440, 500, 510, 520, 530, 540, 565, 600, 630, 640, 650,
                        700, 705, 710, 725, 726, 735, 750, 800, 810, 820, 830, 845, 850, 860, 900, 910, 930, 940, 960,
                        1005, 1006, 1007, 1010, 1021, 1022, 1031, 1032, 1050, 1065, 1105, 1106, 1108, 1200, 1220, 1240,
                        1305, 1306, 1310, 1320, 1340, 1350, 1360, 1400, 1410, 1420, 1430, 1440, 1450, 1460, 1520, 1530,
                        1541, 1545, 1551, 1555, 1560, 1600, 1610, 1640, 1650, 1700, 1710, 1720, 1745, 1750, 1760, 1800,
                        1821, 1822, 1825, 1840, 1860, 1900, 1910, 1920, 1935, 1970, 1980, 2001, 2002, 2003, 2004, 2005,
                        2006, 2011, 2012, 2013, 2014, 2015, 2016, 2025, 2040, 2050, 2100, 2105, 2145, 2170, 2180, 2205,
                        2310, 2320, 2330, 2360, 2400, 2435, 2555, 2600, 2631, 2632, 2634, 2635, 2636, 2640, 2700, 2710,
                        2721, 2722, 2751, 2752, 2755, 2770, 2805, 2810, 2825, 2830, 2840, 2850, 2861, 2862, 2865, 2905,
                        2910, 2920, 3000, 3010, 3030, 3040, 3050, 3090, 3100, 3110, 3120, 3140, 3150, 3160, 3200, 3210,
                        3220, 3230, 3245, 3250, 3255, 3256, 3258, 3261, 3270, 3300, 3310, 3321, 3322, 3323, 3324, 3330,
                        3402, 3423, 3500, 3515, 3520, 3545, 3550, 3610, 3655, 3700, 3710, 3720, 3725, 3740, 3750, 3801,
                        3802, 3820, 3840, 3870, 3900, 3910, 3930, 3945, 4000, 4200, 4210, 4240, 4252, 4330, 4340, 4400,
                        4465, 4700, 4710, 4750, 4800, 4810, 4820, 4830, 4840, 4850, 4920, 4930, 4950, 4965, 5000, 5010,
                        5040, 5100, 5110, 5120, 5140, 5150, 5165, 5220, 5230, 5240, 5250, 5330, 5340, 5350, 5360, 5410,
                        5420, 5500, 5521, 5522, 5530, 5540, 5550, 5560, 5600, 5630, 5710, 5720, 5730, 5740, 5840, 5910,
                        5920, 5940, 6005, 6010, 6115, 6130, 6200, 6210, 6220, 6230, 6240, 6250, 6260, 6305, 6330, 6355,
                        6360, 6400, 6441, 6442, 6460, 6515, 6520, 6530, 6540, 6660, 6700, 6710, 6720, 6730, 6740, 6765,
                        6800, 6825, 6835, 6850, 6950, 7000, 7010, 7020, 7030, 7040, 7100, 7120, 7130, 7140, 7150, 7160,
                        7200, 7210, 7220, 7240, 7300, 7315, 7320, 7330, 7340, 7350, 7360, 7410, 7420, 7430, 7540, 7560,
                        7640, 7700, 7720, 7730, 7740, 7750, 7855, 7905, 7925, 7950, 8000, 8025, 8030, 8040, 8100, 8130,
                        8140, 8225, 8250, 8255, 8450, 8500, 8510, 8555, 8600, 8610, 8620, 8630, 8640, 8650, 8720, 8730,
                        8740, 8750, 8760, 8810, 8830, 8920, 8930, 8940, 8990, 9005, 9030, 9040, 9050, 9121, 9122, 9130,
                        9141, 9150, 9210, 9240, 9265, 9300, 9310, 9410, 9430, 9510, 9570, 9650, 9760, 9800, 9810, 9825,
                        9830],
                value_map=get_value_map(attribute="OCCP", attribute_dict=attribute_dict),
                getter=extend_generic_filter
            ),
        ],
    }

    # Bins
    bin_edges = [-11500, 1, 25000, 60000, 1849000]

    # Model
    model = "qwen/qwen3.6-plus"

    # Config
    cfg = ModelConfig(
        experiment_name="self_consistency_per_level",
        model_name=model,
        reasoning_effort="none",
        sampling_temperature=1.0,
        num_of_samples=20,
        max_number_attempts=10,
    )

    # Run analysis
    # --> Run time is approximately 19 minutes (with 20 samples; GPT 5.4) and costs $2
    results, experiment_folder = main(
        cfg=cfg,
        binary_splits=binary_splits,
        bin_edges=bin_edges,
    )
    timestamp = experiment_folder.split("/")[1]

    print("\n" + "*" * 80)
    print(f"Final results for model {cfg.model_name}:")
    print(f"Sanity checks: {timestamp}")
    print("*" * 80 + "\n")
    print(json.dumps(results, indent=4))
