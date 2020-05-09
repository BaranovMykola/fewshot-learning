import pickle

import tensorflow as tf
# import tqdm

from src.dataset.fss import FssDataset

dataset = FssDataset('./fewshots.csv')

# train = dataset.train().map(FssDataset.convert_to_example)
# test = dataset.train().map(FssDataset.convert_to_example)

with tf.io.TFRecordWriter('train.tfrecord') as writer:
    for sample in dataset.train():
        example = FssDataset.convert_to_example(*sample)
        writer.write(example)

# with open('./dataset.bin', 'wb+') as f:
#     pickle.dump(dataset, f)
