from pathlib import Path

import tensorflow as tf

from src.dataset import FssSamples
from src.dataset.dataset_splitter import NovelSplitter

dataset = FssSamples.from_json_file(Path('./annotations/samples.json'))
train, test = NovelSplitter(dataset, 123).split(0.2)
