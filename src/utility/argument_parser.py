import argparse


def parse_arguments() -> tuple[int, str, str | None]:
    """Parse command line arguments.

    Returns:
        shard_id: The shard ID.
        shard: The shard string.
        datetime: The datetime string. Format: YYYY-MM-DD_HH-MM-SS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, required=False, default=-1)
    parser.add_argument("--datetime", type=str, required=False, default=None)
    args = parser.parse_args()

    shard_id = args.shard_id
    if shard_id == -1:
        shard_id = 0
        shard = "shard_0"
    else:
        shard = f"shard_{shard_id}"

    return shard_id, shard, args.datetime
