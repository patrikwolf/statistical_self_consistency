def process_results(entries: list) -> dict:
    plot_results = {}

    # Plot all results in the list
    for i, entry in enumerate(entries):
        # Collect data from dict
        level_0_error = entry["refinement_results"][0]["aggregated_error"]
        level_1_error = entry["refinement_results"][1]["aggregated_error"]
        level_2_error = entry["refinement_results"][2]["aggregated_error"]
        level_3_error = entry["refinement_results"][3]["aggregated_error"]

        error_gain_l1 = (level_0_error - level_1_error) / level_0_error
        error_gain_l2 = (level_0_error - level_2_error) / level_0_error
        error_gain_l3 = (level_0_error - level_3_error) / level_0_error

        # Add to dict
        plot_results[entry["model"]] = {
            "model": entry["model"],
            "Level 1": error_gain_l1,
            "Level 2": error_gain_l2,
            "Level 3": error_gain_l3,
        }

    return plot_results
