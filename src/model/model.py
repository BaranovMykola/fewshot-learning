import tensorflow as tf


class Model(tf.keras.Model):

    KERAS_VGG16_WEIGHTS = './models/keras/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vgg16: tf.keras.Model = tf.keras.applications.VGG16(include_top=False,
                                                 weights=Model.KERAS_VGG16_WEIGHTS,
                                                 pooling=None)

        self.vgg16_layers = self.vgg16.layers

        # self.vgg_layers = self.vgg16.la

    def call(self, inputs, training=None, mask=None):
        q, s = inputs
        q_features = self.process_set(q)
        s_features = self.process_set(s)

        features = tf.concat([q_features, s_features], axis=-1)

        return features

    def process_set(self, batch):

        feature_list = []
        for set in batch:
            features = self.encode(set)
            # feature_list.append(features)
            features = tf.reduce_sum(features, axis=0)
            # features = tf.reshape(features, [7, 7, 512])
            feature_list.append(features)

        concated_features = tf.stack(feature_list, axis=0)
        return concated_features

    def encode(self, inputs):
        x = inputs
        for layer in self.vgg16_layers:
            x = layer(x)
        return x
