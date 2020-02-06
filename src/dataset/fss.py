import tensorflow as tf
import numpy as np
import pandas as pd


class FssDataset:

    def __init__(self, csv_path, test_classes_frac=0.2):
        np.random.seed(478)
        self.df = pd.read_csv(csv_path).sample(frac=1)

        classes_id = np.unique(self.df.class_id.to_numpy())
        np.random.shuffle(classes_id)
        test_classes = classes_id[:int(len(classes_id)*test_classes_frac)]

        self.test_df = self.df[self.df.class_id.isin(test_classes)]
        self.train_df = self.df[~self.df.class_id.isin(test_classes)]

        self.test_q, self.test_s = FssDataset.q_s_set(self.test_df)
        self.train_q, self.train_s = FssDataset.q_s_set(self.train_df)

        self.test = FssDataset.generate_raw_df(self.test_q, self.test_s)
        self.train = FssDataset.generate_raw_df(self.train_q, self.train_s)

    @staticmethod
    def generate_raw_df(q, s):
        support_df = []
        for idx, i in q.iterrows():
            class_id = i.class_id
            support_set = s[s.class_id == class_id]
            support_in = support_set.in_file.to_numpy()
            support_out = support_set.out_file.to_numpy()
            support = np.concatenate([support_in, support_out])
            support_df.append(support)

        columns = [f'support_rgb_{x}' for x in range(s.shape[1])] + [f'support_mask_{x}' for x in range(s.shape[1])]
        support_df = pd.DataFrame(support_df, index=q.index, columns=columns)
        df = pd.merge(q, support_df, left_index=True, right_index=True)
        return df

    def test_support(self, class_id):
        return FssDataset._support(self.test_s, class_id)

    def train_support(self, class_id):
        return FssDataset._support(self.train_s, class_id)

    @staticmethod
    def _support(s, class_id):
        ss = s[s.class_id == class_id]

        if len(ss) == 0:
            raise RuntimeError(f'No support set for given class id: {class_id}')

        tfss = FssDataset.create(ss)
        tfss = tfss.batch(len(ss))
        support_set = next(iter(tfss))
        return support_set

    @staticmethod
    def q_s_set(df, s_frac=0.5):
        classes_id = np.unique(df.class_id)

        s = None
        q = None
        for c in classes_id:
            class_df = df[df.class_id == c]
            support_set = class_df.sample(frac=s_frac)
            query_set = pd.concat([class_df, support_set]).drop_duplicates(keep=False)

            s = pd.concat([s, support_set]) if s is not None else support_set
            q = pd.concat([q, query_set]) if q is not None else query_set

        return q, s

    @staticmethod
    def create(df):
        rgb = tf.data.Dataset.from_tensor_slices(df.in_file.to_numpy())
        mask = tf.data.Dataset.from_tensor_slices(df.out_file.to_numpy())
        classname = tf.data.Dataset.from_tensor_slices(df['class'].to_numpy())
        class_id = tf.data.Dataset.from_tensor_slices(df.class_id.to_numpy())

        rgb = rgb.map(FssDataset.read_rgb)
        mask = mask.map(FssDataset.read_mask)

        zipped = tf.data.Dataset.zip((rgb, mask, classname, class_id))
        zipped = zipped.map(FssDataset.concat)
        # rgb_q = tf.data.Dataset.

        return zipped

    @staticmethod
    def read_rgb(path):
        file = tf.io.read_file(path)
        image = tf.io.decode_jpeg(file, channels=3)
        image = tf.image.resize(image, (224, 224))
        return image

    @staticmethod
    def read_mask(path):
        file = tf.io.read_file(path)
        image = tf.io.decode_png(file, channels=1)
        image = tf.image.resize(image, (224, 224))
        return image

    @staticmethod
    def concat(rgb, mask, classname, class_id):
        return tf.concat([rgb, mask], axis=-1), classname, class_id
