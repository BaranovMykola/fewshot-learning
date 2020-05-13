import tensorflow as tf

from src.dataset.proto.few_shot_sample import decode_few_shot_sample
from src.dataset.image_decoder import ImageDecoder

ds = tf.data.TFRecordDataset('./test.tfrecord')
ds = ds.map(decode_few_shot_sample)
decoder = ImageDecoder('./Data', (224, 224))

def decode(q, s, g):
    # q, s, g = t
    q = tf.reshape(q, [])
    q = decoder.parse_image(q)
    q = tf.concat([q, tf.zeros(shape=q.shape[:2]+[1])], axis=-1)

    g = tf.reshape(g, [])
    g = decoder.parse_mask(g)

    images = tf.map_fn(decoder.parse_image, s[0, :, 0], dtype=tf.float32)
    masks = tf.map_fn(decoder.parse_mask, s[1, :, 0], dtype=tf.float32)
    support = tf.concat([images, masks], axis=-1)
    return q, support, g

ds = ds.map(decode)

q, s, g = next(iter(ds))
import matplotlib.pyplot as plt

plt.imshow(s[0, :, :, -1])
plt.savefig('./test.png')

