import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.dataset.fss import FssDataset


n_colors = 256

color_array = plt.get_cmap('gist_rainbow')(range(n_colors))
color_array[:, -1] = np.linspace(0.0, 1.0, n_colors)

map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha', colors=color_array)

plt.register_cmap(cmap=map_object)

n_cols = 6
n_rows = 6
dataset = FssDataset('./fewshots.csv')
dataset.test_support(972)

# tf_dataset = FssDataset.create(dataset.train_df).take(n_cols*n_rows)
#
# fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * 5, n_rows * 5))
# fig.suptitle('Sample examples', fontsize=16)
# ax = ax.flatten()
#
# for idx, (sample, classname, class_id) in enumerate(tf_dataset):
#     img = sample[:, :, :3]
#     mask = sample[:, :, 3]
#     ax[idx].imshow(img.numpy()/255)
#     ax[idx].imshow(mask.numpy()/255, cmap='rainbow_alpha', alpha=0.4)
#     ax[idx].title.set_text(f'{classname.numpy().decode()} ({class_id})')
#
# plt.savefig('./dump/dataset_example.png')
