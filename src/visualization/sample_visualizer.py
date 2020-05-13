from typing import Optional

import tensorflow as tf
import matplotlib.pyplot as plt

from src.utils import assert_shape, assert_two_shapes


def show_sample(subplot, image: tf.Tensor, mask: tf.Tensor, classname: tf.Tensor, class_id: tf.Tensor):
    assert_shape(image, [None, None, 3])
    assert_shape(mask, [None, None])
    assert_two_shapes(image.shape[:-1], mask.shape)

    if classname is not None:
        assert_shape(classname, [])
    if class_id is not None:
        assert_shape(class_id, [])

    name = f'{classname.numpy().decode()} ({class_id})' if classname is not None else None

    subplot.imshow(image.numpy())
    subplot.imshow(mask.numpy(), cmap='rainbow_alpha', alpha=0.4)

    if name:
        subplot.title.set_text(name)


def show_sample_batch(subplot: plt.Axes,
                      image: tf.Tensor,
                      mask: tf.Tensor,
                      classname: Optional[tf.Tensor] = None,
                      class_id: Optional[tf.Tensor] = None,
                      *,
                      sample_id: int):
    assert_shape(image, [None, 1, None, None, 3])
    assert_shape(mask, [None, None, None])

    if classname is not None:
        assert_shape(classname, [None])
    if class_id is not None:
        assert_shape(class_id, [None])

    show_sample(subplot,
                image[sample_id, 0],
                mask[sample_id],
                classname[sample_id] if classname is not None else None,
                class_id[sample_id] if class_id is not None else None)
