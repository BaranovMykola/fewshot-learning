from typing import TYPE_CHECKING, Union, List, Tuple, Iterable

if TYPE_CHECKING:
    import tensorflow as tf

def assert_shape(tensor: "tf.Tensor", gt_shape: Union[List, Tuple]):
    assert len(tensor.shape) == len(gt_shape), f'Expected ndim #{len(gt_shape)} for tensor. Got {len(tensor.shape)}'
    assert_two_shapes(tensor.shape, gt_shape)


def assert_two_shapes(shape1: Iterable, shape2: Iterable):
    for i, (dim, gt_dim) in enumerate(zip(shape1, shape2)):
        if gt_dim is None:
            continue
        assert dim == gt_dim, f'Invalid shape of given tensor: ndim #{i} expected to be <{gt_dim}>. Got {dim}'
