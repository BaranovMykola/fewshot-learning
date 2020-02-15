import matplotlib.pyplot as plt

import tensorflow as tf

from src.dataset.fss import FssDataset
from src.utils.memory import memory_limit
from src.visualization.maks import register_colormap


def show_sample(subplot, sample, title):
    img = sample[:, :, :3]
    mask = sample[:, :, 3]
    subplot.imshow(img.numpy())
    subplot.imshow(mask.numpy(), cmap='rainbow_alpha', alpha=0.4)
    subplot.title.set_text(title)


def main():
    register_colormap()
    memory_limit(0.5)

    n_cols = 6
    n_rows = 6
    dataset = FssDataset('./fewshots.csv')
    test = dataset.test

    fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * 4, n_rows * 4))
    fig.suptitle('Sample examples', fontsize=16)

    for idx, (query, support, mask, classname, class_id) in enumerate(test.take(n_cols)):
        name = f'{classname.numpy().decode()} ({class_id})'
        sample_masked = tf.concat([query[0, :, :, :3], mask], axis=-1)
        show_sample(ax[idx, 0], sample_masked, name)
        for i in range(1, n_cols):
            show_sample(ax[idx, i], support[i-1, :, :, :], f'Support {i}')

    plt.savefig('./dump/dataset_example.png')


if __name__ == '__main__':
    main()
