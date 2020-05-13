import os
import pickle
import time
from pathlib import Path
from typing import Any

import tensorflow as tf

from src.model import IoU, F1Score, SampleVisualizerCallback
from src.utils import assert_gpu


class Estimator:
    model: tf.keras.Model

    def __init__(self, model: tf.keras.Model, dataset_file: Path, batch_size: int):
        self.model = model

        from src.dataset.proto.few_shot_sample import decode_few_shot_sample
        from src.dataset.image_decoder import ImageDecoder
        ds = tf.data.TFRecordDataset('./test.tfrecord')
        ds = ds.map(decode_few_shot_sample)
        decoder = ImageDecoder('./Data', (224, 224))

        def decode(q, s, g):
            # q, s, g = t
            q = tf.reshape(q, [])
            q = decoder.parse_image(q)
            q = tf.concat([q, tf.zeros(shape=q.shape[:2] + [1])], axis=-1)
            q = tf.expand_dims(q, axis=0)

            g = tf.reshape(g, [])
            g = decoder.parse_mask(g)

            images = tf.map_fn(decoder.parse_image, s[0, :, 0], dtype=tf.float32)
            masks = tf.map_fn(decoder.parse_mask, s[1, :, 0], dtype=tf.float32)
            support = tf.concat([images, masks], axis=-1)
            g = tf.reshape(g, shape=g.shape[:2])
            return (q, support), g

        ds = ds.map(decode)

        self.batch_size = batch_size
        ds = ds.batch(self.batch_size)

        self.train_data = ds
        # self.test_data_raw = self.dataset.test(for_fit=False)
        self.test_data = ds
        (self.q_shape, self.s_shape), m_shape = self.dataset_shape(self.train_data)

        self.logdir = self.get_logdir('./tensorboard')

    @property
    def metrics(self):
        return [tf.keras.metrics.Precision(thresholds=0.5),
                tf.keras.metrics.Recall(thresholds=0.5),
                IoU(),
                F1Score(threshold=0.5)]

    @property
    def loss(self):
        return tf.losses.binary_crossentropy

    @property
    def optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=0.0001)

    @property
    def callbacks(self):
        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)
        # mcb = SampleVisualizerCallback(self.test_data_raw, self.logdir, steps=5)
        f1_saver = tf.keras.callbacks.ModelCheckpoint(str(self.logdir / 'best_f1.ckpt'),
                                                      monitor='val_f1_score',
                                                      verbose=0,
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      mode='max',
                                                      save_freq='epoch')
        iou_saver = tf.keras.callbacks.ModelCheckpoint(str(self.logdir / 'best_iou.ckpt'), monitor='val_io_u', verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=True, mode='max', save_freq='epoch')
        return [tb_cb, f1_saver, iou_saver]

    def compile(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.model.build(input_shape=[self.q_shape, self.s_shape])

    def train(self):
        assert_gpu()
        self.model.fit(self.train_data,
                       steps_per_epoch=10,
                       validation_steps=10,
                       # steps_per_epoch=len(self.dataset.train_unrolled_df) // self.batch_size,
                       # validation_steps=len(self.dataset.test_unrolled_df) // self.batch_size,
                       validation_data=self.test_data,
                       epochs=228,
                       callbacks=self.callbacks)

    def _prepare_set(self, x: tf.data.Dataset) -> tf.data.Dataset:
        x = x.shuffle(24)
        x = x.batch(self.batch_size)
        x = x.prefetch(1)
        x = x.repeat()
        return x

    @staticmethod
    def dataset_shape(x: tf.data.Dataset) -> Any:
        (q, s), m = x.element_spec
        return (q.shape, s.shape), m.shape

    @staticmethod
    def get_logdir(root: str) -> Path:
        ts = time.gmtime()
        postfix = time.strftime("%Y-%m-%d_%H:%M:%S", ts)
        logdir  = Path(root) / postfix
        if not logdir.exists():
            logdir.mkdir()

        return logdir
