import pickle
import numpy as np

import tensorflow as tf
from train_new import Model

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

(q,s), m = next(iter(train))
model = Model()
model.compile(loss=tf.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=[tf.keras.metrics.binary_accuracy])

q_shape = tuple(q.shape)
s_shape = tuple(s.shape)
model.build(input_shape=[q_shape, s_shape])

model.load_weights('./test_w.h5')

for i, ((q, s), m) in enumerate(train.shuffle(100)):
    m_pred = model.call((q,s), training=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(m[0])
    ax[1].imshow(np.where(m_pred[0, :, :, 0]>0.5, 1, 0))
    plt.savefig(f'./dump/fig{i}.png')
    plt.close(fig)
    print(i)
    if i >= 99:
        break
