import pickle
from typing import Tuple

import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)

from src.dataset.fss import FssDataset

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
print(len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

with open('./dataset.bin', 'rb') as f:
    dataset = pickle.load(f)


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, filters: int, kernel_size: Tuple[int, int], **kwargs):
        super().__init__(**kwargs)

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same')
        # self.bn = tf.keras.layers.BatchNormalization(momentum=1.0)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.activations.relu

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        # x = self.bn(x, training=kwargs['training'])
        x = self.act(x)
        return x

class Model(tf.keras.Model):
    KERAS_VGG16_WEIGHTS = './models/keras/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    UNET_SHORTCUTS = [2, 5, 9, 13, 17]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vgg16: tf.keras.Model = tf.keras.applications.VGG16(include_top=False,
                                                                 weights=Model.KERAS_VGG16_WEIGHTS,
                                                                 pooling=None).layers[2:]
        self.layer0 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), padding='same',
                                             activation=tf.keras.activations.relu)
        self.layer1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same',
                                             activation=tf.keras.activations.relu)
        self.vgg16_layers = [self.layer0, self.layer1] + self.vgg16
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
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                                   activation=tf.keras.activations.sigmoid
                                   )
        ])
        #
    def call(self, inputs, training=None, mask=None):
        q, s = inputs
        # a = tf.reduce_mean(q*self.w)
        # b = tf.reduce_mean(s*self.w)
        f, u = self.encode(q[0])
        # tf.print(tf.reduce_mean(f), tf.reduce_mean(u[0]))
        # tf.print(u[0])
        # a = tf.reduce_mean(f)
        m = self.decode(f, u, training)
        a = m
        # a = tf.reduce_mean(m)
        return a

    @tf.function
    def encode(self, inputs):
        unet_features = []
        x = inputs
        for idx, layer in enumerate(self.vgg16_layers):
            x = layer(x)
            if idx in Model.UNET_SHORTCUTS:
                unet_features.append(x)
        return x, unet_features
        # return x

    def decode(self, input, unet_features, training):
        x = input
        x = self.up1(input, training=training)
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
        # x = tf.squeeze(x)

        return x


bs = 2
def mod(x):
    resh = lambda q, s, m, n, i: ((q, s), m)
    # x = x.shuffle(24)
    x = x.batch(bs)
    x = x.map(resh)
    x = x.take(20)
    x = x.prefetch(1)
    x = x.repeat()
    return x

train = mod(dataset.train)
test = mod(dataset.test)

model = Model()


class MinLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        l = tf.keras.losses.mse(y_pred, tf.ones(y_pred.shape, y_pred.dtype))
        return l


(q,s), m = next(iter(train))

model.compile(loss=MinLoss(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001))
model.fit(train,
          steps_per_epoch=20,
          validation_steps=20,
          validation_data=train,
          epochs=5)
pass

