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
        items = None
        for idx, i in q.iterrows():
            class_id = i.class_id
            support_set = s[s.class_id == class_id]
            support_in = support_set.in_file.to_numpy()
            support_out = support_set.out_file.to_numpy()
            support = np.concatenate([support_in, support_out])
            support_df.append(support)
            items = items or int(len(support)/2)

        columns = [f'support_rgb_{x}' for x in range(items)] + [f'support_mask_{x}' for x in range(items)]
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

        tfss = FssDataset.create(ss, 5)
        tfss = tfss.batch(len(ss))
        support_set = next(iter(tfss))
        return support_set

    @staticmethod
    def q_s_set(df, s_num=5):
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
    def _rgb_mask(rgb_p, mask_p):
        rgb = tf.data.Dataset.from_tensor_slices(rgb_p).map(FssDataset.read_rgb)
        mask = tf.data.Dataset.from_tensor_slices(mask_p).map(FssDataset.read_mask)
        sample = tf.data.Dataset.zip((rgb, mask)).map(FssDataset.concat)
        return sample


    @staticmethod
    def create(df, s):
        tf_q = FssDataset._rgb_mask(df.in_file.to_numpy(), df.out_file.to_numpy())
        tf_q = tf_q.map(lambda x: tf.expand_dims(x, axis=0))

        s_rgb = [f'support_rgb_{x}' for x in range(s)]
        s_mask = [f'support_mask_{x}' for x in range(s)]
        s_rgb = [df[x] for x in s_rgb]
        s_mask = [df[x] for x in s_mask]
        s_f = zip(s_rgb, s_mask)
        tf_s = [FssDataset._rgb_mask(r, m) for r, m in s_f]
        tf_s = tf.data.Dataset.zip(tuple(tf_s)).map(lambda *x: tf.stack(x))

        classname = tf.data.Dataset.from_tensor_slices(df['class'].to_numpy())
        class_id = tf.data.Dataset.from_tensor_slices(df.class_id.to_numpy())
        ds = tf.data.Dataset.zip((tf_q, classname, class_id, tf_s))

        return ds

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
    def concat(rgb, mask):
        return tf.concat([rgb, mask], axis=-1)
