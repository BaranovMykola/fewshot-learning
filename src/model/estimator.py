import time
from pathlib import Path
from typing import Any, Tuple

import tensorflow as tf

from src.dataset.image_decoder import ImageDecoder
from src.dataset.proto.few_shot_sample import decode_few_shot_sample
from src.model import IoU, F1Score, SampleVisualizerCallback
from src.utils import assert_gpu


class Estimator:
    model: tf.keras.Model

    def __init__(self, model: tf.keras.Model, train_dataset: str, val_dataset: str, batch_size: int, data_root: str):
        self.model = model

        self.decoder = ImageDecoder(data_root, (224, 224))
        self.batch_size = batch_size

        self.train_data, self.train_data_len = self._make_dataset(train_dataset)
        self.val_data, self.val_data_len = self._make_dataset(val_dataset)
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
        mcb = SampleVisualizerCallback(self.val_data, self.logdir, steps=5)
        f1_saver = tf.keras.callbacks.ModelCheckpoint(str(self.logdir / 'best_f1.ckpt'),
                                                      monitor='val_f1_score',
                                                      verbose=0,
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      mode='max',
                                                      save_freq='epoch')
        iou_saver = tf.keras.callbacks.ModelCheckpoint(str(self.logdir / 'best_iou.ckpt'),
                                                       monitor='val_io_u',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=True, mode='max', save_freq='epoch')
        return [tb_cb, mcb, f1_saver, iou_saver]

    def compile(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.model.build(input_shape=[self.q_shape, self.s_shape])

    def train(self):
        assert_gpu()
        self.model.fit(self.train_data,
                       # steps_per_epoch=10,
                       # validation_steps=10,
                       steps_per_epoch=self.train_data_len // self.batch_size,
                       validation_steps=self.val_data_len // self.batch_size,
                       validation_data=self.val_data,
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

    def _make_dataset(self, path: str) -> Tuple[tf.data.Dataset, int]:
        ds = tf.data.TFRecordDataset(path)
        ds_len = sum([1 for _ in ds])
        ds = ds.map(decode_few_shot_sample)
        ds = ds.map(self._reshape_dataset)
        ds = ds.batch(self.batch_size)
        ds = ds.shuffle(128).repeat().prefetch(1)

        return ds, ds_len

    def _reshape_dataset(self, query: tf. Tensor, support: tf.Tensor, gt: tf.Tensor) -> Tuple[
                                                                                     Tuple[tf.Tensor, tf.Tensor],
                                                                                     tf.Tensor]:
        query = tf.reshape(query, [])
        query = self.decoder.parse_image(query)
        query = tf.concat([query, tf.zeros(shape=query.shape[:2] + [1])], axis=-1)
        query = tf.expand_dims(query, axis=0)

        gt = tf.reshape(gt, [])
        gt = self.decoder.parse_mask(gt)

        images = tf.map_fn(self.decoder.parse_image, support[0, :, 0], dtype=tf.float32)
        masks = tf.map_fn(self.decoder.parse_mask, support[1, :, 0], dtype=tf.float32)
        support = tf.concat([images, masks], axis=-1)
        gt = tf.reshape(gt, shape=gt.shape[:2])
        return (query, support), gt
