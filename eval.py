import pickle

import tensorflow as tf

from src.model.model import Model

model = Model()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
model.build(input_shape=[(None, 1, 224, 224, 4), (None, 5, 224, 224, 4)])
model.load_weights('./test_weights.h5')

resh = lambda q, s, m, n, i: ((q, s), m)

with open('./dataset.bin', 'rb') as f:
    dataset = pickle.load(f)

bs = 2

mod = lambda x: x.shuffle(24).batch(bs).map(resh).prefetch(1).take(1).repeat()
train = mod(dataset.train)
test = mod(dataset.test)

model.evaluate(test, steps=1)
