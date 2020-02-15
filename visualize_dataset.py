import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.dataset.fss import FssDataset


def show_sample(ax, sample, name):
    img = sample[:, :, :3]
    mask = sample[:, :, 3]
    ax.imshow(img.numpy()/255)
    ax.imshow(mask.numpy()/255, cmap='rainbow_alpha', alpha=0.4)
    ax.title.set_text(name)


n_colors = 256

color_array = plt.get_cmap('gist_rainbow')(range(n_colors))
color_array[:, -1] = np.linspace(0.0, 1.0, n_colors)

map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha', colors=color_array)

plt.register_cmap(cmap=map_object)

n_cols = 6
n_rows = 6
dataset = FssDataset('./fewshots.csv')
test = dataset.create(dataset.test, 5).shuffle(1024)


fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * 4, n_rows * 4))
fig.suptitle('Sample examples', fontsize=16)
# ax = ax.flatten()

for idx, (query, classname, class_id, support) in enumerate(test.take(n_cols)):
    name = f'{classname.numpy().decode()} ({class_id})'
    show_sample(ax[idx, 0], query, name)
    for i in range(1, n_cols):
        show_sample(ax[idx, i], support[i-1, :, :, :], f'Support {i}')

plt.savefig('./dump/dataset_example.png')
