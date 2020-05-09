import itertools
import unittest
from typing import List

from src.dataset import FssSamples
from src.dataset.few_shot_dataset import FewShotDataset
from src.dataset.sample import QuerySupportSample
from src.tests.generate_test_samples import generate_test_samples


class FewShotDatasetTest(unittest.TestCase):

    def setUp(self) -> None:
        j = generate_test_samples(20, 120)
        self.dataset = FssSamples.from_json(j)

    def test_few_shot_dataset(self):
        seeds = [3123, 435, 4234, 423423]
        ks = [2, 3, 4, 5]
        for k, seed in itertools.product(ks, seeds):
            fw = FewShotDataset(self.dataset, seed)

            fw_sample_list: List[QuerySupportSample] = [*fw.split_to_query_support(k)]
            for fw_sample in fw_sample_list:
                self.assertEqual(len(fw_sample.support), k, f'Support len is not {k}')

                for fw_sample_other in fw_sample_list:
                    self.assertFalse(fw_sample.query.sample_id in fw_sample_other.support.ids,
                                     'Support set contains query image')

