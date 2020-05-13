import argparse
import shutil
from pathlib import Path
from typing import Iterable

import tensorflow as tf
import tqdm

from src.dataset import FssSamples
from src.dataset.dataset_splitter import NovelSplitter
from src.dataset.few_shot_dataset import FewShotDataset


def main(args):
    if args.dst.exists():
        is_delete = input(f'{args.dst} already exists. Overwrite? [yes/no]: ').lower()
        if not (is_delete == 'yes' or is_delete == 'y'):
            print('Aborting...')
            return 0

        shutil.rmtree(args.dst)

    args.dst.mkdir()

    dataset = FssSamples.from_json_file(args.annotation)

    if args.method == 'novel':
        train, val = NovelSplitter(dataset, 123).split(0.2)
        train_k_shot = [*FewShotDataset(train, 123).split_to_query_support(args.k)]
        val_k_shot = [*FewShotDataset(val, 123).split_to_query_support(args.k)]
    else:
        raise ValueError(f'Invalid method: {args.method}')

    with tf.io.TFRecordWriter(str(args.dst / 'train.tfrecord')) as writer:
        for i in tqdm.tqdm(train_k_shot, total=len(train_k_shot)):
            example = i.convert_to_proto().SerializeToString()
            writer.write(example)

    with tf.io.TFRecordWriter(str(args.dst / 'val.tfrecord')) as writer:
        for i in tqdm.tqdm(val_k_shot, total=len(val_k_shot)):
            example = i.convert_to_proto().SerializeToString()
            writer.write(example)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst', help='Destination directory', type=lambda x: Path(x), required=True)
    parser.add_argument('--annotation', help='Annotation json file', type=lambda x: Path(x), required=True)
    parser.add_argument('--method', help='Method of split', type=str, required=True)
    parser.add_argument('-k', help='support set size', type=int, required=True)
    arguments = parser.parse_args()

    main(arguments)
