from pathlib import Path

import tensorflow as tf

from src.dataset import FssSamples
from src.dataset.dataset_splitter import NovelSplitter
from src.dataset.few_shot_dataset import FewShotDataset

def main():
    k = 5

    dataset = FssSamples.from_json_file(Path('./annotations/samples.json'))
    train, test = NovelSplitter(dataset, 123).split(0.2)
    train_k_shot = FewShotDataset(train, 123).split_to_query_support(k)
    test_k_shot = FewShotDataset(test, 123).split_to_query_support(k)

    with tf.io.TFRecordWriter('./test.tfrecord') as writer:
        for i in train_k_shot:
            example = i.convert_to_proto().SerializeToString()
            writer.write(example)


if __name__ == '__main__':
    main()
