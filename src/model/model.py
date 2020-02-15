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

        self.vgg16_layers = self.vgg16.layers
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

    def call(self, inputs, training=None, mask=None):
        q, s = inputs
        q_features, _ = self.process_set(q)
        s_features, unet_features = self.process_set(s)


        features = tf.concat([q_features, s_features], axis=-1)
        # mask = self.upsample(features)
        mask = self.decode(features, unet_features)

        return features

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
    def process_set(self, batch):

        feature_list = []
        unet_features_list = []
        for set_sample in batch:
            features, unet_features = self.encode(set_sample)
            unet_features = [tf.reduce_sum(x, axis=0) for x in unet_features]
            unet_features_list.append(unet_features)
            features = tf.reduce_sum(features, axis=0)
            feature_list.append(features)

        concated_features = tf.stack(feature_list, axis=0)

        f = []
        arr = np.array(unet_features_list)
        for i in range(len(Model.UNET_SHORTCUTS)):
            f.append(tf.stack(arr[:, i], axis=0))


        # concated_unet_features = [tf.stack(x, axis=0) for x in unet_features_list]
        return concated_features, f

    #  TODO: Optimize with optional unet processing
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
        self.bn = tf.keras.layers.BatchNormalization(momentum=1)
        self.act = tf.keras.activations.relu

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x

