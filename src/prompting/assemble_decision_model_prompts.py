from experiments_acs.filtering.filter import Filter
from prompting.formatting_helper import format_numeric_value, example_filters, _generate_attribute_list, \
    example_filter_partition


def assemble_income_prompt(
    filters: list[Filter],
    income_threshold: float | int,
    improved_age_desc: bool = False,
) -> tuple[str, str, dict[str, str]]:
    threshold = format_numeric_value(income_threshold)

    question = (f"Is this person's annual income greater than {threshold} USD? "
                f"Estimate the probability under the specified context.")

    if filters:
        attributes = _generate_attribute_list(
            filters=filters,
            improved_age_desc=improved_age_desc,
        )
        context = f"The person is an individual in the United States with the following attributes:\n{attributes}"
    else:
        context = "The person is an individual in the United States."

    options = {
        "A": f"Yes, their annual income is greater than {threshold} USD.",
        "B": f"No, their annual income is at most {threshold} USD.",
    }

    return question, context, options


def assemble_joint_prior_estimation_prompt(
    base_filters: list[Filter],
    filter_list: list[list[Filter]],
    improved_age_desc: bool,
) -> tuple[str, str, dict[str, str], list]:
    question = (
        "Which group does a randomly sampled person belong to? "
        "Estimate the probability of each option under the specified context."
    )

    context = (
        "A person is sampled uniformly at random from the U.S. population. "
        "Use your background knowledge about the demographic composition "
        "of the United States."
    )

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

    options = {}
    for idx, filters in enumerate(filter_list, start=1):
        combined_filters = base_filters + filters

        # Generate the attribute list for the current group
        attributes = _generate_attribute_list(
            filters=combined_filters,
            improved_age_desc=improved_age_desc,
        )

        # Save in list
        group_wise_attribute_lists.append(
            {
                "group_name": f"Group {idx + 1}",
                "attribute_list_string": _generate_attribute_list(
                    filters=combined_filters,
                    improved_age_desc=improved_age_desc),
                "filters": [f.serialize() for f in combined_filters],
            }
        )

        # Generate the option text for the current group
        options[f"Group {idx}"] = (
            f"The person has the following attributes:\n{attributes}"
            if combined_filters
            else "The person has no sociodemographic restrictions."
        )

    return question, context, options, group_wise_attribute_lists


if __name__ == "__main__":
    income_threshold = 40000
    filters = example_filters()

    question, context, options = assemble_income_prompt(
        filters=filters,
        income_threshold=income_threshold,
        improved_age_desc=True,
    )

    question_empty, context_empty, options_empty = assemble_income_prompt(
        filters=[],
        income_threshold=income_threshold,
        improved_age_desc=True,
    )

    filter_list = example_filter_partition()
    prior_question, prior_context, prior_options, _ = assemble_joint_prior_estimation_prompt(
        base_filters=[],
        filter_list=filter_list,
        improved_age_desc=True,
    )

    print("Question")
    print(question)
    print("-" * 30)
    print("Context")
    print(context)
    print("-" * 30)
    print("Options")
    print(options)

    print("\n\n" + "*" * 80 + "\n\n")

    print("Question")
    print(question_empty)
    print("-" * 30)
    print("Context")
    print(context_empty)
    print("-" * 30)
    print("Options")
    print(options_empty)

    print("\n\n" + "*" * 80 + "\n\n")

    print("Question")
    print(prior_question)
    print("Context")
    print(prior_context)
    print("Options")
    print(prior_options)
