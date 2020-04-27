import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):

    def __init__(self, threshold: float, **kwargs):
        super().__init__(**kwargs)
        self.precision = tf.keras.metrics.Precision(threshold)
        self.recall = tf.keras.metrics.Recall(threshold)

    def update_state(self, *args, **kwargs):
        self.precision(*args, **kwargs)
        self.recall(*args, **kwargs)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return (2*p*r)/(r+p)

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
