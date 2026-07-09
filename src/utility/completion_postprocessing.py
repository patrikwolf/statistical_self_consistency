import re
import json

from typing import Tuple


def extract_choice(text: str) -> Tuple[int, int] | Tuple[None, None]:
    """
    Extracts a final numeric choice (1 or 2) from an LLM output string.
    Handles cases like:
      - "The answer is 1."
      - "Answer: **2**"
      - "I choose 1!"
      - "**1**."
    Returns 1, 2, or None if no valid choice is found.
    """
    # Regex explanation:
    # - \*{0,2} allows 0–2 asterisks before/after (for Markdown bold/italic)
    # - _{0,2} allows underscores (for __1__)
    # - ([12]) captures the number
    # - [\s\.\,\!\?\:\)\]]*$ allows trailing punctuation/whitespace
    pattern = r"(?:\*{0,2}_{0,2}\s*)([12])(?:\s*_{0,2}\*{0,2})[\s\.\,\!\?\:\)\]]*$"
    choice, start_index = extract_number(text=text, pattern=pattern)

    if choice is not None:
        return int(choice), start_index

    # No valid choice found
    return None, None


def extract_estimates(text: str, extract_confidence: bool) -> (
        Tuple[float, float, float, int] | Tuple[None, None, None, None]):
    """
    Extracts final estimates from an LLM output string.

    :return: probability, lower_bound, upper_bound, index
    """
    if text is None:
        print(" ----> Warning: No response text provided...")
        return None, None, None, None

    if extract_confidence:
        return extract_with_confidence(text=text)
    else:
        probability, index = extract_number(text=text)
        return probability, None, None, index


def extract_number(
        text: str,
        pattern: str = r"\[\[\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*\]\]"
) -> Tuple[float, int] | Tuple[None, None]:
    """
    Extract a probability estimate from an LLM output string in the format [[p]]

    Examples of supported outputs:
      - "[[0.5]]"
      - "[[8000]]"
      - "The answer is [[0.95]]."
      - "$$ [[1.0]] $$"
      - "Probability: [[0]]"

    Returns:
        A tuple (value, index), where value is the extracted float and index is
        the start position of the captured number inside the input string.
        Returns (None, None) if no valid bracketed probability is found.
    """
    # Old regex
    # pattern: str = r"(?:\*{0,2}_{0,2}\s*)(\d+(?:\.\d+)?)(?:\s*_{0,2}\*{0,2})[\s\.\,\!\?\:\)\]]*$"
    # pattern: str = r"\[\[\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*\]\]"    # only [0,1]

    # Normalize text
    text = text.strip().lower()

    # Find match
    match = re.search(pattern, text)
    if match:
        label = match.group(1)
        start_index = match.start(1)
        return float(label), start_index

    # No valid choice found
    return None, None


def extract_textual_response(text: str,
                             pattern: str = r"\[\[\s*(ABOVE|BELOW)\s*\]\]"
                             ) -> Tuple[str, int] | Tuple[None, None]:

    # Normalize text
    text = text.strip().upper()

    # Find match
    match = re.search(pattern, text)
    if match:
        label = match.group(1)
        start_index = match.start(1)
        return label, start_index

    # No valid choice found
    return None, None


def extract_json_response(text: str, pattern: str = r"\{.*\}") -> Tuple[dict, int] | Tuple[None, None]:
    # Find match
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        json_str = match.group(0)
        start_index = match.start(0)

        try:
            return json.loads(json_str), start_index
        except json.JSONDecodeError:
            print("   ---> Warning: Invalid JSON response")
            print(f"  ---> Response: {text}")
            print(f"  ---> Matched string: {json_str}")

    # No valid choice found
    return None, None


def extract_with_confidence(text: str,
                            pattern: str = r"\[\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]\]")\
        -> Tuple[float, float, float, int] | Tuple[None, None, None, None]:
    # Normalize text
    text = text.strip().lower()

    # Find match
    match = re.search(pattern, text)
    if match:
        p, l, u = map(float, match.groups())
        start_index = match.start(1)
        return float(p), float(l), float(u), start_index

    # No valid choice found
    return None, None, None, None


if __name__ == "__main__":
    text = """
### Step 2: Use known income distribution patterns
In the U.S. in 2018, the median income for people with an **Associate's degree** was approximately **$43,000*
 (source: U.S. Census Bureau).

This means that **about 50% of people with an Associate's degree had an income above the median**, which is close to
 $40,000.

### Step 3: Refine the estimate
Since the **median income** is just above $40,000, the probability that someone with an Associate's degree earns **more
 than $40,000** is **slightly above 50%**.

### Step 4: Adjust for age
People in their late 50s (like 59 years old) are typically in the **middle of their careers**. This age group tends to
 have **higher earnings** than younger individuals with the same education level, especially if they have **some work
 experience**.

So, we can estimate that the probability is **slightly higher than 50%**.

### Final Estimate:
**~0.55**

### ✅ Final Answer:
**0.55**
"""

    # Extract choice
    choice, index = extract_number(text)
    print(f"Extracted choice: {choice} at index {index}")

    text = """
So, we can estimate that the probability is **slightly higher than 50%**.

### Final Estimate:
[[0.5, 0.35, 0.65]]
    """

    # Extract probability with confidence interval
    p, l, u, index = extract_with_confidence(text=text)
    print(f"Extracted probability: {p} at index {index}")
    print(f"Extracted lower bound: {l} at index {index}")
    print(f"Extracted upper bound: {u} at index {index}")

    text = """
    So, we can estimate that the probability is **slightly higher than 50%**.

    ### Final Estimate:
    [[0.5]]
    """

    p, index = extract_number(text=text)
    print(f"Extracted probability: {p} at index {index}")

    text = r"""
    Given the individual’s age range, type of employment, and likely occupation, the probability that their annual
     income exceeds $40,000 is **very high**.

    $$
    [[0.95]]
    $$
    """

    p, index = extract_number(text=text)
    print(f"Extracted probability: {p} at index {index}")

    text = r"""
    Given the individual’s age range, type of employment, and likely occupation, the probability that their annual
     income exceeds $40,000 is **very high**.

    $$
    [[10000]]
    $$
    """

    p, index = extract_number(text=text)
    print(f"Extracted probability: {p} at index {index}")

    text = "[[ABOVE]]"

    p, index = extract_textual_response(text=text)
    print(f"Extracted textual response: {p} at index {index}")
