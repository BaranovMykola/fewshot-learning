from typing import Tuple

import tensorflow as tf


class VggConvBlock(tf.keras.layers.Layer):

    def __init__(self, filters: int, kernel_size: Tuple[int, int], **kwargs):
        super().__init__(**kwargs)

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same')
        self.bn = tf.keras.layers.BatchNormalization(momentum=1.0)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.activations.relu

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=kwargs.get('training'))
        x = self.act(x)
        return x
