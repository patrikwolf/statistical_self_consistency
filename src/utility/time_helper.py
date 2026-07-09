from datetime import datetime


def convert_timestamp_to_readable_format(timestamp: str) -> str:
    """
    Convert a timestamp string in the format "YYYY-MM-DD__HH-MM-SS" to a more human-readable format.
    """

    # Parse the input timestamp
    dt = datetime.strptime(timestamp, "%Y-%m-%d__%H-%M-%S")

    # Format it to a more readable string
    readable_timestamp = dt.strftime("%d.%m.%Y, %H:%M:%S")

    return readable_timestamp


def get_experiment_timestamp(experiment_name: str, timestamp: str | None = None) -> tuple[str, str, str]:
    if timestamp is not None:
        date, timestamp = timestamp.split("_")
    else:
        date = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H-%M-%S")

    # Create experiment folder name
    experiment_folder = f"{experiment_name}/{date}__{timestamp}"
    return date, timestamp, experiment_folder


if __name__ == "__main__":
    # Example usage
    timestamp = "2024-06-30__14-30-00"
    readable = convert_timestamp_to_readable_format(timestamp)
    print(f"Original timestamp: {timestamp}")
    print(f"Readable format: {readable}")
