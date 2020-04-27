import tensorflow as tf


class IoU(tf.keras.metrics.Metric):

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        # self.intersection: tf.Variable = self.add_weight('union_area', dtype=tf.int64)
        # self.union: tf.Variable = self.add_weight('union_area', dtype=tf.int64)
        # self.
        # self.threshold = tf.Variable(initial_value=threshold)
        self.threshold: tf.Variable = self.add_weight('t', dtype=tf.float32)
        self.threshold.assign(threshold)
        self.iou: tf.Variable = self.add_weight('iou', dtype=tf.float64)
        self.counter: tf.Variable = self.add_weight('counter', dtype=tf.int32)

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, **kwargs) -> None:
        y_pred = tf.where(y_pred > self.threshold, 1.0, 0.0)

        union_mask = tf.clip_by_value(y_true + y_pred, 0, 1)
        intersection_mask = y_true * y_pred

        intersection = tf.math.count_nonzero(intersection_mask)
        union = tf.math.count_nonzero(union_mask)

        iou = intersection/union
        self.iou.assign_add(iou)
        self.counter.assign_add(1)

    def result(self) -> tf.Tensor:
        return self.iou/tf.cast(self.counter, tf.float64)

    def reset_states(self):
        self.iou.assign(0)
        self.counter.assign(0)
