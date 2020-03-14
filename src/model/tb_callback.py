import io
import os

import tensorflow as tf
import matplotlib.pyplot as plt

from src.visualization.maks import register_colormap

class TbCallback(tf.keras.callbacks.Callback):

    def __init__(self, model: tf.keras.Model, test: tf.data.Dataset, root: str):
        super().__init__()
        assert len(test.element_spec) == 5

        register_colormap()
        self.test = test.batch(2)
        self.model = model
        self.root = os.path.join(root, 'imgs')
        self.file_writer = tf.summary.create_file_writer(self.root)

    def on_epoch_end(self, epoch, logs=None):
        images = []
        for q, s, m, class_name, class_id in self.test:
            bs = q.shape[0]
            m_pred = self.model((q, s), training=False)

            fig, ax = plt.subplots(nrows=bs, ncols=2)

            for i in range(bs):
                name = f'{class_name[i].numpy().decode()} ({class_id[i]})'

                sample_masked = tf.concat([q[i, 0, :, :, :3], tf.expand_dims(m[i], axis=-1)], axis=-1)
                TbCallback.show_sample(ax[i, 0], sample_masked, name)

                sample_masked_pred = tf.concat([q[i, 0, :, :, :3], tf.expand_dims(m_pred[i], axis=-1)], axis=-1)
                TbCallback.show_sample(ax[i, 1], sample_masked_pred, name)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            images.append(image)
            # Add image summary

        images = tf.concat(images, axis=0)

        with self.file_writer.as_default():
            tf.summary.image('segmentation_samples', images, step=epoch)

            # for i in range(1, n_cols):
            #     TbCallback.show_sample(ax[idx, i], support[i-1, :, :, :], f'Support {i}')

    @staticmethod
    def show_sample(subplot, sample, title):
        img = sample[:, :, :3]
        mask = sample[:, :, 3]
        subplot.imshow(img.numpy())
        subplot.imshow(mask.numpy(), cmap='rainbow_alpha', alpha=0.4)
        subplot.title.set_text(title)



