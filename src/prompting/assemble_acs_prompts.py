from typing import Literal
from experiments_acs.filtering.filter import Filter
from prompting.formatting_helper import format_numeric_value, assemble_prompt_with_optional_filters, \
    format_template, _generate_attribute_list, map_filter_value, make_example_json, example_filters, \
    example_filter_partition


def assemble_income_prompt(
        filters: list[Filter],
        income_threshold: float,
        prompting_scheme: Literal["sociodemographic", "persona", "unspecified"],
        filtered_template_filename: str,
        no_filter_template_filename: str,
        income_greater_than_threshold: bool,
        extract_confidence: bool = False,
        confidence: int | float | None = None,
        improved_age_desc: bool = False,
) -> str:
    if prompting_scheme == "sociodemographic":
        prompt = assemble_sociodemographic_income_prompt(
            filters=filters,
            income_threshold=income_threshold,
            filtered_template_filename=filtered_template_filename,
            no_filter_template_filename=no_filter_template_filename,
            extract_confidence=extract_confidence,
            confidence=confidence,
            improved_age_desc=improved_age_desc,
        )
    elif prompting_scheme == "persona":
        prompt = assemble_persona_prompt_income(
            filters=filters,
            income_threshold=income_threshold,
            filtered_template_filename=filtered_template_filename,
            improved_age_desc=improved_age_desc,
        )
    elif prompting_scheme == "unspecified":
        raise ValueError("Please specify prompting scheme. The option 'unspecified' is invalid.")
    else:
        raise ValueError(f"Unknown prompting scheme: {prompting_scheme}")

    # Replace "exceeds" in prompt with "at most"
    if not income_greater_than_threshold:
        prompt = prompt.replace("annual income exceeds", "annual income is at most")

    return prompt


def assemble_thresholded_commute_time_prompt(
        filters: list[Filter],
        commute_time: int,
        prompting_scheme: Literal["sociodemographic", "persona", "unspecified"],
        filtered_template_filename: str,
        no_filter_template_filename: str,
) -> str:
    if prompting_scheme != "sociodemographic":
        raise ValueError(f"Prompting scheme '{prompting_scheme}' is not supported.")

    template_values = {
        "threshold": commute_time,
    }

    return assemble_prompt_with_optional_filters(
        filters=filters,
        filtered_template_filename=filtered_template_filename,
        no_filter_template_filename=no_filter_template_filename,
        **template_values,
    )


def assemble_commute_time_micro_macro_prompt(
        commute_time: int,
        prompting_scheme: Literal["sociodemographic", "persona", "unspecified"],
        template_filename: str,
) -> str:
    if prompting_scheme != "sociodemographic":
        raise ValueError(f"Prompting scheme '{prompting_scheme}' is not supported.")

    # Format prompt
    prompt = format_template(template_filename=template_filename,
                             threshold=commute_time)

    return prompt


def assemble_thresholded_explicit_micro_macro_prompt(
        base_filters: list[Filter],
        filter_list: list[list[Filter]],
        income_threshold: float,
        prompting_scheme: Literal["sociodemographic", "persona", "unspecified"],
        filtered_template_filename: str,
        improved_age_desc: bool,
) -> tuple[str, list]:
    if prompting_scheme != "sociodemographic":
        raise ValueError(f"Prompting scheme '{prompting_scheme}' is not supported.")

    if improved_age_desc:
        print()

    # Generate all group descriptions
    group_wise_attribute_lists = []
    for idx, filters in enumerate(filter_list):
        combined_filters = base_filters + filters
        group_wise_attribute_lists.append(
            {
                "group_name": f"Subpopulation {idx + 1}",
                "attribute_list_string": _generate_attribute_list(
                    filters=combined_filters,
                    improved_age_desc=improved_age_desc),
                "filters": [f.serialize() for f in combined_filters],
            }
        )

    # Concatenate
    groups = ""
    for group in group_wise_attribute_lists:
        groups += group["group_name"] + "\n" + group["attribute_list_string"] + "\n\n"

    # Truncate last two \n
    groups = groups[:-2]

    # Format prompt
    prompt = format_template(template_filename=filtered_template_filename,
                             subpopulations=groups,
                             income_threshold=format_numeric_value(income_threshold),
                             )

    return prompt, group_wise_attribute_lists


def assemble_thresholded_implicit_micro_macro_prompt(
        income_threshold: float,
        prompting_scheme: Literal["sociodemographic", "persona", "unspecified"],
        filtered_template_filename: str,
) -> str:
    if prompting_scheme != "sociodemographic":
        raise ValueError(f"Prompting scheme '{prompting_scheme}' is not supported.")

    # Format prompt
    prompt = format_template(template_filename=filtered_template_filename,
                             income_threshold=format_numeric_value(income_threshold))

    return prompt


def assemble_few_bin_implicit_micro_macro_prompt(
        bin_edges: list[float],
        prompting_scheme: Literal["sociodemographic", "persona", "unspecified"],
        filtered_template_filename: str,
        uniform_example_distribution: bool = False,
) -> tuple[str, dict]:
    if prompting_scheme != "sociodemographic":
        raise ValueError(f"Prompting scheme '{prompting_scheme}' is not supported.")

    # Bins
    bins = [[left, right] for left, right in zip(bin_edges[:-1], bin_edges[1:])]

    # Generate group names
    bin_details = {
        f"Bin {idx + 1}": {
            "bin_name": f"Bin {idx + 1}",
            "bin": bin,
        } for idx, bin in enumerate(bins)
    }
    bin_name_list = [item["bin_name"] for item in bin_details.values()]
    bin_names = ", ".join(bin_name_list)

    # Concatenate
    bins_str = ""
    for bin, bin_name in zip(bins, bin_name_list):
        bins_str += bin_name + "\n" + f"{bin[0]} to {bin[1]} USD" + "\n\n"

    # Truncate last two \n
    bins_str = bins_str[:-2]

    # Generate example output
    example_output = make_example_json(num_groups=len(bins), group_descriptor="Bin",
                                       uniform_distribution=uniform_example_distribution)

    # Format prompt
    prompt = format_template(template_filename=filtered_template_filename,
                             bins=bins_str,
                             bin_names=bin_names,
                             example_output=example_output,
                             )

    return prompt, bin_details


def assemble_two_bin_income_prompt(
        filters: list[Filter],
        income_threshold: float,
        prompting_scheme: Literal["sociodemographic", "persona", "unspecified"],
        filtered_template_filename: str,
        no_filter_template_filename: str,
):
    if prompting_scheme == "sociodemographic":
        template_values = {
            "income_threshold": format_numeric_value(income_threshold),
        }
        return assemble_prompt_with_optional_filters(
            filters=filters,
            filtered_template_filename=filtered_template_filename,
            no_filter_template_filename=no_filter_template_filename,
            **template_values,
        )
    else:
        raise ValueError(f"Prompting scheme '{prompting_scheme}' is not supported.")


def assemble_few_bin_income_prompt(
        bin_edges: list[float],
        filters: list[Filter],
        prompting_scheme: Literal["sociodemographic", "persona", "unspecified"],
        filtered_template_filename: str,
        no_filter_template_filename: str,
        uniform_example_distribution: bool = False,
        improved_age_desc: bool = False,
) -> tuple[str, dict]:

    if prompting_scheme != "sociodemographic":
        raise ValueError(f"Prompting scheme '{prompting_scheme}' is not supported.")

    # Bins
    bins = [[left, right] for left, right in zip(bin_edges[:-1], bin_edges[1:])]

    # Generate group names
    bin_details = {
        f"Bin {idx + 1}": {
            "bin_name": f"Bin {idx + 1}",
            "bin": bin,
        } for idx, bin in enumerate(bins)
    }
    bin_name_list = [item["bin_name"] for item in bin_details.values()]
    bin_names = ", ".join(bin_name_list)

    # Concatenate
    bins_str = ""
    for bin, bin_name in zip(bins, bin_name_list):
        bins_str += bin_name + "\n" + f"{bin[0]} to {bin[1]} USD" + "\n\n"

    # Truncate last two \n
    bins_str = bins_str[:-2]

    # Generate example output
    example_output = make_example_json(num_groups=len(bins),
                                       group_descriptor="Bin",
                                       uniform_distribution=uniform_example_distribution)

    prompt = assemble_prompt_with_optional_filters(
        filters=filters,
        bins=bins_str,
        bin_names=bin_names,
        example_output=example_output,
        filtered_template_filename=filtered_template_filename,
        no_filter_template_filename=no_filter_template_filename,
        improved_age_desc=improved_age_desc,
    )

    return prompt, bin_details


def assemble_sociodemographic_income_prompt(
        filters: list[Filter],
        income_threshold: float,
        filtered_template_filename: str = "ACS_income_v1.txt",
        no_filter_template_filename: str = "ACS_income_no_filters_v1.txt",
        extract_confidence: bool = False,
        confidence: int | float | None = None,
        improved_age_desc: bool = False,
) -> str:
    template_values = {
        "income_threshold": format_numeric_value(income_threshold),
    }
    if extract_confidence:
        if confidence is None:
            raise ValueError("confidence must be provided when extract_confidence=True.")
        template_values["confidence"] = format_numeric_value(confidence)

    return assemble_prompt_with_optional_filters(
        filters=filters,
        filtered_template_filename=filtered_template_filename,
        no_filter_template_filename=no_filter_template_filename,
        improved_age_desc=improved_age_desc,
        **template_values,
    )


def assemble_persona_prompt_income(
        filters: list[Filter],
        income_threshold: float,
        filtered_template_filename: str,
        improved_age_desc: bool = False,
) -> str:
    # Initialize template values
    template_values = {
        "income_threshold": format_numeric_value(income_threshold),
    }

    # Generate persona
    persona = generate_persona(filters=filters, improved_age_desc=improved_age_desc)
    template_values["persona"] = persona

    return format_template(
        filtered_template_filename,
        **template_values,
    )


def generate_persona(filters: list[Filter], improved_age_desc: bool = False) -> str:
    if improved_age_desc:
        raise ValueError("improved_age_desc is not implemented yet.")

    # Initialize persona
    persona = "You are a person who lives in the United States."

    for filter in filters:
        value_map = filter.value_map
        characteristics = f"Your {filter.description} is "
        if len(filter.values) > 1:
            characteristics += "either "
            for value in filter.values:
                characteristics += map_filter_value(value=value, value_map=value_map) + " or "

            # Remove the trailing " or "
            characteristics = characteristics[:-4] + "."
        else:
            characteristics += map_filter_value(value=filter.values[0], value_map=value_map) + "."

        # Add to persona
        persona += " " + characteristics

    return persona


def assemble_income_prompt_with_ground_truth(
        filters: list[Filter],
        income_threshold: float,
        true_income: float,
        improved_age_desc: bool,
        template_filename: str = "ACS_income_with_ground_truth_v2.txt",
) -> str:
    return format_template(
        template_filename,
        attributes=_generate_attribute_list(filters=filters, improved_age_desc=improved_age_desc),
        true_income=format_numeric_value(true_income),
        income_threshold=format_numeric_value(income_threshold),
    )


def assemble_income_distribution_prompt(
    filters: list[Filter],
    template_filename: str = "ACS_income_distribution_v1.txt",
    no_filter_template_filename: str = "ACS_income_distribution_no_filters_v1.txt",
) -> str:
    return assemble_prompt_with_optional_filters(
        filters=filters,
        filtered_template_filename=template_filename,
        no_filter_template_filename=no_filter_template_filename,
    )


def assemble_commute_distribution_prompt(
    filters: list[Filter],
    template_filename: str = "ACS_commute_distribution_v1.txt",
    no_filter_template_filename: str = "ACS_commute_distribution_no_filters_v1.txt",
) -> str:
    return assemble_prompt_with_optional_filters(
        filters=filters,
        filtered_template_filename=template_filename,
        no_filter_template_filename=no_filter_template_filename,
    )


def assemble_individual_prior_estimation_prompt(
        base_filters: list[Filter],
        decomposition_filters: list[Filter],
        template_filename: str = "ACS_prior_estimation_v1.txt",
        improved_age_desc: bool = False,
) -> str:
    filters = base_filters + decomposition_filters
    return format_template(
        template_filename=template_filename,
        attributes=_generate_attribute_list(filters=filters, improved_age_desc=improved_age_desc)
    )


def assemble_joint_prior_estimation_prompt(
        base_filters: list[Filter],
        filter_list: list[list[Filter]],
        improved_age_desc: bool,
        template_filename: str = "ACS_joint_prior_estimation_v1.txt",
) -> tuple[str, list]:
    # Generate all group descriptions
    group_wise_attribute_lists = []
    for idx, filters in enumerate(filter_list):
        combined_filters = base_filters + filters
        group_wise_attribute_lists.append(
            {
                "group_name": f"Group {idx + 1}",
                "attribute_list_string": _generate_attribute_list(
                    filters=combined_filters,
                    improved_age_desc=improved_age_desc),
                "filters": [f.serialize() for f in combined_filters],
            }
        )

    # Concatenate
    groups = ""
    for group in group_wise_attribute_lists:
        groups += group["group_name"] + "\n" + group["attribute_list_string"] + "\n\n"

    # Truncate last two \n
    groups = groups[:-2]

    # Generate group names
    group_names = ", ".join([f"Group {idx + 1}" for idx in range(len(filter_list))])

    # Generate example output
    example_output = make_example_json(num_groups=len(filter_list))

    prompt = format_template(template_filename=template_filename,
                             groups=groups,
                             group_names=group_names,
                             example_output=example_output)

    return prompt, group_wise_attribute_lists


def assemble_generic_prompt(
        filters: list[Filter],
        template_filename: str,
        improved_age_desc: bool,
) -> str:
    return format_template(
        template_filename,
        attributes=_generate_attribute_list(filters=filters, improved_age_desc=improved_age_desc),
    )


if __name__ == "__main__":
    income_threshold = 40000
    filters = example_filters()

    income_prompt = assemble_sociodemographic_income_prompt(
        filters=filters,
        income_threshold=income_threshold,
        filtered_template_filename="ACS_income_v3.txt",
        improved_age_desc=True,
    )

    income_prompt_conf = assemble_sociodemographic_income_prompt(
        filters=filters,
        income_threshold=income_threshold,
        filtered_template_filename="ACS_income_v4_confidence.txt",
        extract_confidence=True,
        confidence=85,
    )

    income_prompt_gt = assemble_income_prompt_with_ground_truth(
        filters=filters,
        income_threshold=income_threshold,
        true_income=20000,
        improved_age_desc=False,
    )

    persona_prompt = generate_persona(filters=filters)

    filter_list = example_filter_partition()
    prior_prompt, _ = assemble_joint_prior_estimation_prompt(
        base_filters=[],
        filter_list=filter_list,
        improved_age_desc=False,
    )

    few_bin_income_prompt, _ = assemble_few_bin_income_prompt(
        bin_edges=[0, 1000, 20_000],
        filters=filters,
        prompting_scheme="sociodemographic",
        filtered_template_filename="ACS_income_few_bins_v1.txt",
        no_filter_template_filename="ACS_income_few_bins_no_filter_v1.txt"
    )

    micro_macro_explicit_groups, _ = assemble_thresholded_explicit_micro_macro_prompt(
        base_filters=[],
        filter_list=filter_list,
        income_threshold=income_threshold,
        prompting_scheme="sociodemographic",
        filtered_template_filename="ACS_income_thresholded_explicit_micro_macro.txt",
        improved_age_desc=True,
    )

    micro_macro_implicit = assemble_thresholded_implicit_micro_macro_prompt(
        income_threshold=income_threshold,
        prompting_scheme="sociodemographic",
        filtered_template_filename="ACS_income_thresholded_implicit_micro_macro_v1.txt",
    )

    commute_prompt = assemble_thresholded_commute_time_prompt(
        filters=[],
        commute_time=32,
        prompting_scheme="sociodemographic",
        filtered_template_filename="",
        no_filter_template_filename="ACS_commute_time_no_filters_v1.txt",
    )

    commute_micro_macro = assemble_commute_time_micro_macro_prompt(
        commute_time=32,
        prompting_scheme="sociodemographic",
        template_filename="ACS_commute_time_micro_macro_v1.txt",
    )

    print(income_prompt)
    print("\n\n" + "*" * 80 + "\n\n")
    print(income_prompt_conf)
    print("\n\n" + "*" * 80 + "\n\n")
    print(income_prompt_gt)
    print("\n\n" + "*" * 80 + "\n\n")
    print(persona_prompt)
    print("\n\n" + "*" * 80 + "\n\n")
    print(prior_prompt)
    print("\n\n" + "*" * 80 + "\n\n")
    print(few_bin_income_prompt)
    print("\n\n" + "*" * 80 + "\n\n")
    print(micro_macro_explicit_groups)
    print("\n\n" + "*" * 80 + "\n\n")
    print(micro_macro_implicit)
    print("\n\n" + "*" * 80 + "\n\n")
    print(commute_prompt)
    print("\n\n" + "*" * 80 + "\n\n")
    print(commute_micro_macro)
