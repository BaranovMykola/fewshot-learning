import tensorflow as tf


def is_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    return len(physical_devices) > 0


def setup_gpu(eager: bool = False):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental_run_functions_eagerly(eager)

def assert_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    print(len(physical_devices))
