import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


def get_histogram_colors(llm: bool, num_colors):
    if llm:
        if num_colors > 5:
            return plt.cm.viridis(np.linspace(0.15, 0.9, num_colors))
        else:
            return plt.cm.viridis(np.linspace(0.35, 0.9, num_colors))
    else:
        colors = [
            [0.392157, 0.560784, 1.000000, 1.0],
            [0.470588, 0.368627, 0.941176, 1.0],
            [0.862745, 0.149020, 0.498039, 1.0],
            [0.996078, 0.380392, 0.000000, 1.0],
            [1.000000, 0.690196, 0.000000, 1.0],
        ]
        if num_colors == 5:
            return colors
        elif num_colors > 5:
            return resample_colors(colors, num_colors)
        else:
            return colors[1:num_colors + 1]


def resample_colors(colors, num_colors):
    cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)
    return cmap(np.linspace(0, 1, num_colors))


def html_to_rgba_array(color: str, alpha: float = 1.0) -> str:
    """
    Convert an HTML color string '#RRGGBB' to a normalized RGBA array string.

    Example:
        '#785EF0' -> '[0.470588 0.368627 0.941176 1.]'
    """

    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected color in format '#RRGGBB', got {color!r}")
    r = int(color[0:2], 16) / 255
    g = int(color[2:4], 16) / 255
    b = int(color[4:6], 16) / 255

    return f"[{r:.6f}, {g:.6f}, {b:.6f}, {alpha:.1f}]"


if __name__ == "__main__":
    html_colors = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]
    for color in html_colors:
        rgba = html_to_rgba_array(color, alpha=1.0)
        print(rgba)
    print("\n" + "*" * 80 + "\n")

    # Resample GT colors
    colors = [
        [0.392157, 0.560784, 1.000000, 1.0],
        [0.470588, 0.368627, 0.941176, 1.0],
        [0.862745, 0.149020, 0.498039, 1.0],
        [0.996078, 0.380392, 0.000000, 1.0],
        [1.000000, 0.690196, 0.000000, 1.0],
    ]

    colors_8 = resample_colors(colors, 8)
    print(colors_8)
