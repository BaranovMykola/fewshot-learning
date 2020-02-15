import numpy as np
import tensorflow as tf

from src.dataset.fss import FssDataset
from src.model.model import Model


q = np.zeros([4, 1, 224, 224, 4], np.float32)
s = np.zeros([4, 5, 224, 224, 4], np.float32)
# tf.config.experimental_run_functions_eagerly(True)

model = Model()

res = model((q, s))

# dataset = FssDataset('./fewshots.csv')
# test = dataset.test
# test = test.map(lambda q, s, m, n, i: ((q, s), m))
# test = test.batch(1).take(1)

# i, o = next(iter(test))
# model(i)

model.compile(loss=tf.keras.losses.MSE, optimizer=tf.keras.optimizers.Adam())
tf_q = tf.data.Dataset.from_tensor_slices([q])
tf_s = tf.data.Dataset.from_tensor_slices([s])
tf_mask = tf.data.Dataset.from_tensor_slices([np.zeros([4, 224, 224, 1], np.float32)])
tf = tf.data.Dataset.zip((tf_q, tf_s, tf_mask))
tf = tf.map(lambda q, s, m: ((q, s), m))
# d = tf.data.Dataset.zip((tf, tf_mask))
# res = next(iter(d))
#
print('************* FIIIIIIIIIIIIIIIT ****************')
res = model.fit(tf, epochs=10)
# print(res)

pass