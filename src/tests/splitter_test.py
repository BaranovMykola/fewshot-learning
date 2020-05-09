import itertools
import unittest
import json

from src.dataset import FssSamples, Sample, SampleList, Category, CategoryList
from src.dataset.dataset_splitter import NovelSplitter

from src.tests.generate_test_samples import generate_test_samples


class SplitterTest(unittest.TestCase):

    def setUp(self) -> None:
        j = generate_test_samples(cats=20, samples=1000)
        self.dataset = FssSamples.from_json(j)

    def test_novel_splitter(self):
        seeds = [123, 456, 333, 534534534]
        ratios = [0.1, 0.5, 0.2, 0.8]
        for seed, ratio in itertools.product(seeds, ratios):
            log_values = str({'seed': seed, 'ratio': ratio})

            splitter = NovelSplitter(self.dataset, 123)
            train, val = splitter.split(ratio)

            expected_val_cats = int(len(self.dataset.categories)*ratio)
            expected_train_cats = len(self.dataset.categories) - expected_val_cats

            self.assertEqual(len(train.categories), expected_train_cats, log_values)
            self.assertEqual(len(val.categories), expected_val_cats, log_values)

            self.assertFalse(any([c in val.categories.ids for c in train.categories.ids]), log_values)
            self.assertFalse(any([s in val.samples.ids for s in train.samples.ids]), log_values)
