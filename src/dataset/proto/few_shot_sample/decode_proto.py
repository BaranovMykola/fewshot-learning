from typing import Tuple

import tensorflow as tf


def decode_masked_img(tensor: tf.Tensor) -> tf.Tensor:
    fws = tf.io.decode_proto(bytes=tensor,
                             message_type='MaskedImage',
                             field_names=['image_path', 'mask_path'],
                             output_types=[tf.string, tf.string],
                             descriptor_source="./src/dataset/proto/few_shot_sample/few_shot_sample.desc")
    return fws[1]


def decode_few_shot_sample(tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    fws = tf.io.decode_proto(bytes=tensor,
                             message_type='FewShotSample',
                             field_names=['query_image_path', 'support', 'gt_mask_path'],
                             output_types=[tf.string, tf.string, tf.string],
                             descriptor_source="./src/dataset/proto/few_shot_sample/few_shot_sample.desc")
    return fws[1][0], decode_masked_img(fws[1][1]), fws[1][2]
