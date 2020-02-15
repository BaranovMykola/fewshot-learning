import numpy as np
import matplotlib.pyplot as plt
import resource
from matplotlib.colors import LinearSegmentedColormap

from src.dataset.fss import FssDataset


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def show_sample(ax, sample, name):
    img = sample[:, :, :3]
    mask = sample[:, :, 3]
    ax.imshow(img.numpy()/255)
    ax.imshow(mask.numpy()/255, cmap='rainbow_alpha', alpha=0.4)
    ax.title.set_text(name)


def memory_limit(percentage: float):
    """
    只在linux操作系统起作用
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * percentage, hard))


memory_limit(0.5)


n_colors = 256

color_array = plt.get_cmap('gist_rainbow')(range(n_colors))
color_array[:, -1] = np.linspace(0.0, 1.0, n_colors)

map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha', colors=color_array)

plt.register_cmap(cmap=map_object)

n_cols = 6
n_rows = 6
dataset = FssDataset('./fewshots.csv')
test = dataset.create(dataset.test, 5).shuffle(128)


fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * 4, n_rows * 4))
fig.suptitle('Sample examples', fontsize=16)
# ax = ax.flatten()

for idx, (query, classname, class_id, support) in enumerate(test.take(n_cols)):
    name = f'{classname.numpy().decode()} ({class_id})'
    show_sample(ax[idx, 0], query[0, :, :, :], name)
    for i in range(1, n_cols):
        show_sample(ax[idx, i], support[i-1, :, :, :], f'Support {i}')

plt.savefig('./dump/dataset_example.png')
