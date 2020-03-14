import pickle

import tensorflow as tf

from src.model import Model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
print(len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

with open('./dataset.bin', 'rb') as f:
    dataset = pickle.load(f)

bs = 2
def mod(x):
    resh = lambda q, s, m, n, i: ((q, s), m)
    # x = x.shuffle(24)
    x = x.batch(bs)
    x = x.map(resh)
    # x = x.take(20)
    x = x.prefetch(1)
    x = x.repeat()
    return x

train = mod(dataset.train)
test = mod(dataset.test)

model = Model()

(q,s), m = next(iter(train))

model.compile(loss=tf.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=[tf.keras.metrics.binary_accuracy])

q_shape = tuple(q.shape)
s_shape = tuple(s.shape)
model.build(input_shape=[q_shape, s_shape])

# model.load_weights('./test_w.h5')
model.fit(train,
          # steps_per_epoch=200,
          # validation_steps=20,
          steps_per_epoch=len(dataset.train_unrolled_df)//bs,
          validation_steps=len(dataset.test_unrolled_df)//bs,
          validation_data=test,
          epochs=10,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./tf_log')])
model.save_weights('./test_w.h5')
pass

