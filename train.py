# import pickle
#
# import numpy as np
# import tensorflow as tf
#
# from src.dataset.fss import FssDataset
# from src.model.model import Model
#
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# print(len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
#
# q = np.zeros([4, 1, 224, 224, 4], np.float32)
# s = np.zeros([4, 5, 224, 224, 4], np.float32)
# # tf.config.experimental_run_functions_eagerly(True)
#
# model = Model()
#
# res = model((q, s))
#
# resh = lambda q, s, m, n, i: ((q, s), m)
#
# with open('./dataset.bin', 'rb') as f:
#     dataset = pickle.load(f)
#
# bs = 2
#
# mod = lambda x: x.shuffle(24).batch(bs).map(resh).prefetch(1).repeat()
# train = mod(dataset.train)
# test = mod(dataset.test)
#
# def loss_fn_cstm(y_true, y_pred):
#     # bce = -(y_true * tf.math.log(y_pred) + (1-y_true) * tf.math.log(1-y_pred))
#     # _loss = tf.reduce_mean(tf.reduce_mean(bce, axis=-1), axis=-1)
#     return tf.keras.losses.binary_crossentropy(y_true, y_pred)
#
# # i, o = next(iter(test))
# # model(i)
#
# model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
#               optimizer=tf.keras.optimizers.Adam())
#               # metrics=[tf.keras.metrics.MeanIoU(2)])
#
# logdir = './tf_log'
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# # print('************* FIIIIIIIIIIIIIIIT ****************')
# his = model.fit(train,
#                 epochs=1,
#                 # steps_per_epoch=len(dataset.train_unrolled_df)//bs,
#                 # validation_steps=len(dataset.test_unrolled_df)//bs,
#                 steps_per_epoch=10,
#                 validation_steps=10,
#                 validation_data=train,
#                 callbacks=[tensorboard_callback])
# # model.save_weights('./test_weights.h5')
# #
# #
# # import matplotlib.pyplot as plt
# #
# # loss_fn = tf.keras.losses.mse
# # opt = tf.keras.optimizers.Adam()
# # model.build(input_shape=[(None, 1, 224, 224, 4), (None, 5, 224, 224, 4)])
# #
# # for (q, s), m in train:
# #     with tf.GradientTape() as tape:
# #         pred_m = model(inputs=(q, s))
# #         loss = tf.reduce_mean(loss_fn_cstm(m, pred_m))
# #         # model.weights
# #
# #     grads = tape.gradient(loss, model.trainable_variables)
# #     opt.apply_gradients(zip(grads, model.trainable_variables))
# #     # print(loss, np.min(pred_m), np.max(pred_m))
# #     print(loss)
# #
# # # model((q,s))
# #
# #
# # pass