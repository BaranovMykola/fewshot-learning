import tensorflow as tf


def assert_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    print(len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
