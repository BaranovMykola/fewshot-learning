import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def register_colormap():
    n_colors = 256

    color_array = plt.get_cmap('gist_rainbow')(range(n_colors))
    color_array[:, -1] = np.linspace(0.0, 1.0, n_colors)

    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha', colors=color_array)

    plt.register_cmap(cmap=map_object)
