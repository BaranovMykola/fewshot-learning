import io
import itertools
import os

import tensorflow as tf
import matplotlib.pyplot as plt

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

        for s in range(self.steps):
            fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * 4, n_rows * 4))
            fig.suptitle('Sample examples', fontsize=16)

            for idx, (query, support, mask, classname, class_id) in enumerate(itertools.islice(self.test, n_rows)):
                pred_mask = self.model((query, support), training=False)
                pred_mask = pred_mask[0, :, :]
                query = query[0]
                support = support[0]
                classname = classname[0]
                class_id = class_id[0]
                mask = mask[0]

                name = f'{classname.numpy().decode()} ({class_id})'
                sample_masked_gt = tf.concat([query[0, :, :, :3], tf.expand_dims(mask, axis=-1)], axis=-1)
                sample_masked_pred = tf.concat([query[0, :, :, :3], tf.expand_dims(pred_mask, axis=-1)], axis=-1)

                TbCallback.show_sample(ax[idx, 0], sample_masked_gt, name)
                TbCallback.show_sample(ax[idx, 1], sample_masked_pred, name)
                # for i in range(1, n_cols):
                #     TbCallback.show_sample(ax[idx, i], support[i - 1, :, :, :], f'Support {i}')

            plt.savefig(os.path.join(self.root, f'epoch_{epoch}_{s}.png'))
        # images = []
        # for q, s, m, class_name, class_id in self.test:
        #     bs = q.shape[0]
        #     m_pred = self.model((q, s), training=False)
        #
        #     fig, ax = plt.subplots(nrows=bs, ncols=2)
        #
        #     for i in range(bs):
        #         name = f'{class_name[i].numpy().decode()} ({class_id[i]})'
        #
        #         sample_masked = tf.concat([q[i, 0, :, :, :3], tf.expand_dims(m[i], axis=-1)], axis=-1)
        #         TbCallback.show_sample(ax[i, 0], sample_masked, name)
        #
        #         sample_masked_pred = tf.concat([q[i, 0, :, :, :3], tf.expand_dims(m_pred[i], axis=-1)], axis=-1)
        #         TbCallback.show_sample(ax[i, 1], sample_masked_pred, name)

        #     buf = io.BytesIO()
        #     plt.savefig(buf, format='png')
        #     buf.seek(0)
        #     image = tf.image.decode_png(buf.getvalue(), channels=4)
        #     # Add the batch dimension
        #     image = tf.expand_dims(image, 0)
        #     images.append(image)
        #     # Add image summary
        #
        # images = tf.concat(images, axis=0)
        #
        # with self.file_writer.as_default():
        #     tf.summary.image('segmentation_samples', images, step=epoch)

            # for i in range(1, n_cols):
            #     TbCallback.show_sample(ax[idx, i], support[i-1, :, :, :], f'Support {i}')

    @staticmethod
    def show_sample(subplot, sample, title):
        img = sample[:, :, :3]
        mask = sample[:, :, 3]
        subplot.imshow(img.numpy())
        subplot.imshow(mask.numpy(), cmap='rainbow_alpha', alpha=0.4)
        subplot.title.set_text(title)



