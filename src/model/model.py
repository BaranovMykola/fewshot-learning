from typing import Tuple

import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):

    KERAS_VGG16_WEIGHTS = './models/keras/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    UNET_SHORTCUTS = [2, 5, 9, 13, 17]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vgg16: tf.keras.Model = tf.keras.applications.VGG16(include_top=False,
                                                 weights=Model.KERAS_VGG16_WEIGHTS,
                                                 pooling=None)

        self.layer0 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), padding='same')
        self.layer1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')
        self.vgg16_layers = [self.layer0, self.layer1] + self.vgg16.layers[2:]
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.up1 = tf.keras.Sequential([
            ConvBlock(512, (3, 3)),
            ConvBlock(512, (3, 3))
        ])
        self.up2 = tf.keras.Sequential([
            ConvBlock(512, (3, 3)),
            ConvBlock(512, (3, 3)),
        ])
        self.up3 = tf.keras.Sequential([
            ConvBlock(256, (3, 3)),
            ConvBlock(256, (3, 3)),
        ])
        self.up4 = tf.keras.Sequential([
            ConvBlock(128, (3, 3)),
            ConvBlock(128, (3, 3)),
        ])
        self.up5 = tf.keras.Sequential([
            ConvBlock(64, (3, 3)),
            ConvBlock(64, (3, 3)),
        ])
        self.up6 = tf.keras.Sequential([
            ConvBlock(61, (3, 3)),
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same')
        ])

    # @tf.function
    def call(self, inputs, training=None, mask=None):
        q, s = inputs
        q_features, _ = self.process_set(q)
        s_features, unet_features = self.process_set(s)
        #
        #
        features = tf.concat([q_features, s_features], axis=-1)
        # # mask = self.upsample(features)
        __mask = self.decode(features, unet_features)

        # return mask
        return __mask

    # @tf.function
    def decode(self, input, unet_features):
        x = self.up1(input)
        x = self.upsample(x)
        x = tf.concat([x, unet_features[-1]], axis=-1)
        x = self.up2(x)
        x = self.upsample(x)
        x = tf.concat([x, unet_features[-2]], axis=-1)
        x = self.up3(x)
        x = self.upsample(x)
        x = tf.concat([x, unet_features[-3]], axis=-1)
        x = self.up4(x)
        x = self.upsample(x)
        x = tf.concat([x, unet_features[-4]], axis=-1)
        x = self.up5(x)
        x = self.upsample(x)
        x = tf.concat([x, unet_features[-5]], axis=-1)
        x = self.up6(x)

        return x

    #  TODO: Optimize with optional unet processing
    # @tf.function
    def process_set(self, batch):

        # feature_list = []
        # unet_features_list = []
        features, unet_features_list = tf.map_fn(self.t1, batch, dtype=(tf.float32, [tf.float32, tf.float32,
                                                                                     tf.float32, tf.float32,
                                                            tf.float32]))
        # for set_sample in batch:
        #     self.t1(set_sample)

        concated_features = tf.stack(features, axis=0)

        # f = []
        # arr = np.array(unet_features_list)
        # for i in range(len(Model.UNET_SHORTCUTS)):
        #     f.append(tf.stack(arr[:, i], axis=0))

        # res = [[tf.stack(t, axis=0) for t in l] for l in unet_features_list]
        # f = lambda x: x[0]
        ress = []
        for i in range(5):
            res = tf.map_fn(lambda x: x[i], unet_features_list, dtype=tf.float32)
            ress.append(res)


        # concated_unet_features = [tf.stack(x, axis=0) for x in unet_features_list]
        # return concated_features, f
        return concated_features, ress

    # @tf.function
    # def t2(self, i):
    #     return x[i]

    @tf.function
    def t1(self, set_sample):
        features, unet_features = self.encode(set_sample)
        unet_features = [tf.reduce_sum(x, axis=0) for x in unet_features]
        features = tf.reduce_sum(features, axis=0)
        return features, unet_features
        # return features, features

    #  TODO: Optimize with optional unet processing
    @tf.function
    def encode(self, inputs):
        unet_features = []
        x = inputs
        for idx, layer in enumerate(self.vgg16_layers):
            x = layer(x)
            print(idx, x.shape)
            if idx in Model.UNET_SHORTCUTS:
                unet_features.append(x)
        return x, unet_features


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, filters: int, kernel_size: Tuple[int, int], **kwargs):
        super().__init__(**kwargs)

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same')
        self.bn = tf.keras.layers.BatchNormalization(momentum=1.0)
        self.act = tf.keras.activations.relu

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x

