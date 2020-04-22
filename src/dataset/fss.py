from typing import Tuple, Optional

import tensorflow as tf
import numpy as np
import pandas as pd


class FssDataset:

    def __init__(self, csv_path: str, test_classes_frac: float = 0.2):
        np.random.seed(478)
        self.df = pd.read_csv(csv_path).sample(frac=1)

        # Take test classes
        classes_id = np.unique(self.df.class_id.to_numpy())
        np.random.shuffle(classes_id)
        test_classes = classes_id[:int(len(classes_id) * test_classes_frac)]

        # Split train/test sets
        self.test_df = self.df[self.df.class_id.isin(test_classes)]
        self.train_df = self.df[~self.df.class_id.isin(test_classes)]

        self.support_size: Optional[int] = None

        # Split query/support set
        self.test_q, self.test_s = FssDataset.split_query_support_set(self.test_df)
        self.train_q, self.train_s = FssDataset.split_query_support_set(self.train_df)

        # Generate unrolled sets
        self.test_unrolled_df, self.support_size_test = FssDataset.generate_df_with_support_samples(self.test_q,
                                                                                                    self.test_s)
        self.train_unrolled_df, self.support_size_train = FssDataset.generate_df_with_support_samples(self.train_q,
                                                                                                      self.train_s)

    def train(self, for_fit: bool = False) -> tf.data.Dataset:
        return FssDataset._dataset_from_unrolled_df(self.train_unrolled_df, self.support_size_train, for_fit)

    def test(self, for_fit: bool = False):
        return FssDataset._dataset_from_unrolled_df(self.test_unrolled_df, self.support_size_test, for_fit)

    @staticmethod
    def _dataset_from_unrolled_df(unrolled_df: pd.DataFrame, support_size: int, for_fit: bool) -> tf.data.Dataset:
        dataset = FssDataset.create(unrolled_df, support_size)
        if for_fit:
            dataset = dataset.map(FssDataset.reshape)

        return dataset

    @staticmethod
    def generate_df_with_support_samples(query_df: pd.DataFrame, support_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        support_df_unrolled = []
        items = []
        for idx, i in query_df.iterrows():
            class_id = i.class_id
            support_set = support_df[support_df.class_id == class_id]
            support_in = support_set.in_file.to_numpy()
            support_out = support_set.out_file.to_numpy()
            support = np.concatenate([support_in, support_out])
            support_df_unrolled.append(support)
            items.append(len(support) // 2)

        items = np.min(items)

        columns = [f'support_rgb_{x}' for x in range(items)] + [f'support_mask_{x}' for x in range(items)]
        support_df_unrolled = pd.DataFrame(support_df_unrolled, index=query_df.index, columns=columns)
        df = pd.merge(query_df, support_df_unrolled, left_index=True, right_index=True)
        df = df.sample(frac=1)
        return df, items

    @staticmethod
    def split_query_support_set(df: pd.DataFrame, s_num: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        classes_id = np.unique(df.class_id)

        s = None
        q = None
        for c in classes_id:
            class_df = df[df.class_id == c]
            support_set = class_df.sample(s_num)
            query_set = pd.concat([class_df, support_set]).drop_duplicates(keep=False)

            s = pd.concat([s, support_set]) if s is not None else support_set
            q = pd.concat([q, query_set]) if q is not None else query_set

        return q, s

    @staticmethod
    def _rgb_mask(rgb_p: np.ndarray, mask_p: np.ndarray) -> tf.data.Dataset:
        rgb = tf.data.Dataset.from_tensor_slices(rgb_p).map(FssDataset.read_rgb)
        mask = tf.data.Dataset.from_tensor_slices(mask_p).map(FssDataset.read_mask)
        sample = tf.data.Dataset.zip((rgb, mask)).map(FssDataset.concat)
        return sample

    @staticmethod
    def create(unrolled_df: pd.DataFrame, support_size: int) -> tf.data.Dataset:
        tf_q = tf.data.Dataset.from_tensor_slices(unrolled_df.in_file.to_numpy()).map(FssDataset.read_rgb)
        tf_q = tf_q.map(lambda x: tf.concat([x, tf.zeros([224, 224, 1], tf.float32)], axis=-1))

        tf_q = tf_q.map(lambda x: tf.expand_dims(x, axis=0))
        label_mask = tf.data.Dataset.from_tensor_slices(unrolled_df.out_file.to_numpy()).map(FssDataset.read_mask)
        label_mask = label_mask.map(tf.squeeze)

        s_rgb = [f'support_rgb_{x}' for x in range(support_size)]
        s_mask = [f'support_mask_{x}' for x in range(support_size)]
        s_rgb = [unrolled_df[x] for x in s_rgb]
        s_mask = [unrolled_df[x] for x in s_mask]
        s_f = zip(s_rgb, s_mask)
        tf_s = [FssDataset._rgb_mask(r, m) for r, m in s_f]
        tf_s = tf.data.Dataset.zip(tuple(tf_s)).map(lambda *x: tf.stack(x))

        classname = tf.data.Dataset.from_tensor_slices(unrolled_df['class'].to_numpy())
        class_id = tf.data.Dataset.from_tensor_slices(unrolled_df.class_id.to_numpy())
        ds = tf.data.Dataset.zip((tf_q, tf_s, label_mask, classname, class_id))

        return ds

    @staticmethod
    def read_rgb(path: tf.Tensor) -> tf.Tensor:
        file = tf.io.read_file(path)
        image = tf.io.decode_jpeg(file, channels=3)
        image = tf.image.resize(image, (224, 224)) / 225
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    @staticmethod
    def read_mask(path: tf.Tensor) -> tf.Tensor:
        file = tf.io.read_file(path)
        image = tf.io.decode_png(file, channels=1)
        image = tf.image.resize(image, (224, 224)) / 255
        image = tf.where(image > 0.5, 1, 0)
        image = tf.cast(image, tf.float32)
        return image

    @staticmethod
    def concat(rgb: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        return tf.concat([rgb, mask], axis=-1)

    @staticmethod
    def reshape(q, s, m, _, __):
        return (q, s), m
