import tensorflow as tf


class IoU(tf.keras.metrics.Metric):

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.intersection_area: tf.Variable = self.add_weight('union_area', dtype=tf.int64)
        self.union_area: tf.Variable = self.add_weight('union_area', dtype=tf.int64)
        self.threshold = tf.Variable(threshold)

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, **kwargs) -> None:
        y_pred = tf.where(y_pred > self.threshold, 1.0, 0.0)

        union_mask = tf.clip_by_value(y_true + y_pred, 0, 1)
        intersection_mask = y_true * y_pred

        intersection = tf.math.count_nonzero(intersection_mask)
        union = tf.math.count_nonzero(union_mask)

        self.union_area.assign_add(union)
        self.intersection_area.assign_add(intersection)

    def result(self) -> tf.Tensor:
        return self.intersection_area / self.union_area
