import os
import pickle
import sys
import time
from typing import Any

import tensorflow as tf

# import tensorflow_addons as tfa
# tf.config.experimental_run_functions_eagerly(True)

from src.model import Model, TbCallback, IoU, F1Score
from src.utils import assert_gpu
from src.dataset.fss import FssDataset

batch_size = 2


def prepare_set(x: tf.data.Dataset) -> tf.data.Dataset:
    x = x.shuffle(24)
    x = x.batch(batch_size)
    x = x.prefetch(1)
    x = x.repeat()
    return x


def dataset_shape(x: tf.data.Dataset) -> Any:
    (q, s), m = x.element_spec
    return (q.shape, s.shape), m.shape


def get_logdir(root: str) -> str:
    ts = time.gmtime()
    postfix = time.strftime("%Y-%m-%d_%H:%M:%S", ts)
    return os.path.join(root, postfix)


def main(args):
    assert_gpu()
    with open('./dataset.bin', 'rb') as f:
        dataset: FssDataset = pickle.load(f)
    model = Model()
    train = prepare_set(dataset.train(for_fit=True))
    test_raw = dataset.test(for_fit=False)
    test = prepare_set(dataset.test(for_fit=True))

    (q_shape, s_shape), m_shape = dataset_shape(train)
    model.compile(loss=tf.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=[tf.keras.metrics.Precision(thresholds=0.5), tf.keras.metrics.Recall(thresholds=0.5),
                           IoU(), F1Score(threshold=0.5)])
    model.build(input_shape=[q_shape, s_shape])

    logdir = get_logdir('./tensorboard')
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    mcb = TbCallback(model, test_raw, logdir, steps=5)
    f1_saver = tf.keras.callbacks.ModelCheckpoint(logdir, monitor='val_f1_score', verbose=0, save_best_only=True,
                                                  save_weights_only=True, mode='max', save_freq='epoch')
    iou_saver = tf.keras.callbacks.ModelCheckpoint(logdir, monitor='val_io_u', verbose=0, save_best_only=True,
                                                   save_weights_only=True, mode='max', save_freq='epoch')
    model.fit(train,
              # steps_per_epoch=10,
              # validation_steps=10,
              steps_per_epoch=len(dataset.train_unrolled_df) // batch_size,
              validation_steps=len(dataset.test_unrolled_df) // batch_size,
              validation_data=test,
              epochs=228,
              callbacks=[tb_cb, mcb, f1_saver, iou_saver])

    return 0


if __name__ == '__main__':
    code = main(None)
    sys.exit(code)
