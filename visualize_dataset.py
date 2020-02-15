import matplotlib.pyplot as plt

from src.dataset.fss import FssDataset
from src.utils.memory import memory_limit


def show_sample(subplot, sample, title):
    img = sample[:, :, :3]
    mask = sample[:, :, 3]
    subplot.imshow(img.numpy() / 255)
    subplot.imshow(mask.numpy() / 255, cmap='rainbow_alpha', alpha=0.4)
    subplot.title.set_text(title)


def main():
    memory_limit(0.5)

    n_cols = 6
    n_rows = 6
    dataset = FssDataset('./fewshots.csv')
    test = dataset.test

    fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * 4, n_rows * 4))
    fig.suptitle('Sample examples', fontsize=16)

    for idx, (query, classname, class_id, support) in enumerate(test.take(n_cols)):
        name = f'{classname.numpy().decode()} ({class_id})'
        show_sample(ax[idx, 0], query[0, :, :, :], name)
        for i in range(1, n_cols):
            show_sample(ax[idx, i], support[i-1, :, :, :], f'Support {i}')

    plt.savefig('./dump/dataset_example.png')


if __name__ == '__main__':
    main()
