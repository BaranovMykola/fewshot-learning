import tensorflow as tf
import matplotlib.pyplot as plt

from src.utils import assert_shape, assert_two_shapes


def show_sample(subplot, image: tf.Tensor, mask: tf.Tensor, classname: tf.Tensor, class_id: tf.Tensor):
    assert_shape(image, [None, None, 3])
    assert_shape(mask, [None, None])
    assert_two_shapes(image.shape[:-1], mask.shape)
    assert_shape(classname, [])
    assert_shape(class_id, [])

    name = f'{classname.numpy().decode()} ({class_id})'

    subplot.imshow(image.numpy())
    subplot.imshow(mask.numpy(), cmap='rainbow_alpha', alpha=0.4)
    subplot.title.set_text(name)


def show_sample_batch(subplot: plt.Axes,
                      image: tf.Tensor,
                      mask: tf.Tensor,
                      classname: tf.Tensor,
                      class_id: tf.Tensor,
                      *,
                      sample_id: int):
    assert_shape(image, [None, 1, None, None, 3])
    assert_shape(mask, [1, None, None])
    assert_shape(classname, [None])
    assert_shape(class_id, [None])
    show_sample(subplot, image[sample_id, 0], mask[sample_id], classname[sample_id], class_id[sample_id])
