

def get_markers(num_markers: int, size_scale: float = 1.0):
    markers = ["o", "s", "X", "P", "D", "v", "D", "<", ">", "*"]

    if num_markers > len(markers):
        raise ValueError(f"num_markers should be less than or equal to {len(markers)}")

    marker_sizes = []
    for marker in markers:
        if marker in ["X", "P", "<", ">", "*"]:
            marker_sizes.append(8 * size_scale)
        else:
            marker_sizes.append(6 * size_scale)

    return markers[:num_markers], marker_sizes[:num_markers]
