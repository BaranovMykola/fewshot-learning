from typing import List

import tensorflow as tf

from .vgg_conv_block import VggConvBlock


class Model(tf.keras.Model):
    KERAS_VGG16_WEIGHTS = './models/keras/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    UNET_SHORTCUTS = [2, 5, 9, 13, 17]

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

        self.vgg16: List[tf.keras.layers.Layer] = tf.keras.applications.VGG16(include_top=False,
                                                                              weights=Model.KERAS_VGG16_WEIGHTS,
                                                                              pooling=None).layers[2:]
        self.layer0 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), padding='same',
                                             activation=tf.keras.activations.relu, name='vgg_layer_0')
        self.layer1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                             activation=tf.keras.activations.relu, name='vgg_layer_1')
        self.vgg16_layers = [self.layer0, self.layer1] + self.vgg16
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample')
        self.up1 = tf.keras.Sequential([
            VggConvBlock(512, (3, 3)),
            VggConvBlock(512, (3, 3))
        ], name='up1')
        self.up2 = tf.keras.Sequential([
            VggConvBlock(512, (3, 3)),
            VggConvBlock(512, (3, 3)),
        ], name='up2')
        self.up3 = tf.keras.Sequential([
            VggConvBlock(256, (3, 3)),
            VggConvBlock(256, (3, 3)),
        ], name='up3')
        self.up4 = tf.keras.Sequential([
            VggConvBlock(128, (3, 3)),
            VggConvBlock(128, (3, 3)),
        ], name='up4')
        self.up5 = tf.keras.Sequential([
            VggConvBlock(64, (3, 3)),
            VggConvBlock(64, (3, 3)),
        ], name='up5')
        self.up6 = tf.keras.Sequential([
            VggConvBlock(61, (3, 3)),
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                                   activation=tf.keras.activations.sigmoid, name='mask_producer')
        ], name='up6')

    def call(self, inputs, training=None, mask=None):
        q, s = inputs
        fq, _ = self.process_set(q)
        fs, us = self.process_set(s)
        fc = tf.concat((fq, fs), axis=-1)

        m = self.decode(fc, us, training)
        return m

    @tf.function
    def process_set(self, batch):
        f, u = tf.map_fn(self.encode, batch, dtype=(tf.float32, [tf.float32, tf.float32, tf.float32, tf.float32,
                                                                 tf.float32]))
        _u = [tf.reduce_sum(x, axis=1) for x in u]
        _f = tf.reduce_sum(f, axis=1)
        return _f, _u

    @tf.function
    def encode(self, inputs):
        unet_features = []
        x = inputs
        for idx, layer in enumerate(self.vgg16_layers):
            x = layer(x)
            if idx in Model.UNET_SHORTCUTS:
                unet_features.append(x)
        return x, unet_features

    def decode(self, inputs, unet_features, training):
        x = inputs
        x = self.up1(x, training=training)
        x = self.upsample(x)
        x = tf.concat([x, unet_features[-1]], axis=-1)
        x = self.up2(x, training=training)
        x = self.upsample(x)
        x = tf.concat([x, unet_features[-2]], axis=-1)
        x = self.up3(x, training=training)
        x = self.upsample(x)
        x = tf.concat([x, unet_features[-3]], axis=-1)
        x = self.up4(x, training=training)
        x = self.upsample(x)
        x = tf.concat([x, unet_features[-4]], axis=-1)
        x = self.up5(x, training=training)
        x = self.upsample(x)
        x = tf.concat([x, unet_features[-5]], axis=-1)
        x = self.up6(x, training=training)
        x = x[:, :, :, 0]

        return x
