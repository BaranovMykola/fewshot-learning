from typing import Tuple

import tensorflow as tf


class ImageDecoder:

    def __init__(self, root: str, img_size: Tuple[int, int]):
        self.root = tf.convert_to_tensor(root, dtype=tf.string)
        self.size = img_size

    def parse_image(self, path: tf.Tensor):
        file_data = self._read_file(path)
        image = tf.io.decode_jpeg(file_data, channels=3)
        image = self._normalize_image(image)
        return image

    def parse_mask(self, path: tf.Tensor):
        file_data = self._read_file(path)
        image = tf.io.decode_png(file_data, channels=1)
        image = self._normalize_image(image)
        image = tf.where(image > 0.5, 1.0, 0.0)
        return image

    def _read_file(self, path: tf.Tensor) -> tf.Tensor:
        joined_path = tf.strings.join([self.root, path], separator='/')
        file_data = tf.io.read_file(joined_path)
        return file_data

    def _normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        image = tf.image.resize(image, self.size)
        image = tf.cast(image, tf.float32)/255.0
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
        return image
