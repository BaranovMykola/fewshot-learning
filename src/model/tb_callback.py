import io
import itertools
import os

import tensorflow as tf
import matplotlib.pyplot as plt

from src.visualization import show_sample
from src.visualization.maks import register_colormap


class TbCallback(tf.keras.callbacks.Callback):

    def __init__(self, model: tf.keras.Model, test: tf.data.Dataset, root: str, steps: int):
        super().__init__()
        assert len(test.element_spec) == 5

        register_colormap()
        self.test = test.batch(1)
        self.model = model
        self.root = os.path.join(root, 'imgs')
        self.file_writer = tf.summary.create_file_writer(self.root)
        self.steps = steps

    def on_epoch_end(self, epoch, logs=None):
        n_cols = 2
        n_rows = 6

        ds_iter = iter(self.test)
        for s in range(self.steps):
            fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * 4, n_rows * 4))
            fig.suptitle('Sample examples', fontsize=16)

            for idx, (query, support, mask, classname, class_id) in enumerate(itertools.islice(ds_iter, n_rows)):
                pred_mask = self.model((query, support), training=False)
                show_sample(ax[idx, 0], query[0, 0, :, :, :3], mask[0], classname[0], class_id[0])
                show_sample(ax[idx, 1], query[0, 0, :, :, :3], pred_mask[0], classname[0], class_id[0])

            plt.savefig(os.path.join(self.root, f'epoch_{epoch}_{s}.png'))
            plt.close(fig)



