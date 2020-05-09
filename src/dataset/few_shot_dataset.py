import random
from typing import Iterable

from src.dataset import FssSamples

from .sample import QuerySupportSample


class FewShotDataset:

    def __init__(self, dataset: FssSamples, seed: int):
        random.seed(seed)
        self.dataset = dataset

    def split_to_query_support(self, k: int) -> Iterable[QuerySupportSample]:
        all_categories = self.dataset.categories

        for cat in all_categories:
            sample_of_cat = self.dataset.samples[cat]

            if len(sample_of_cat) <= k:
                raise RuntimeError(f'Not enough samples to extract #{k} support set. All samples #{len(sample_of_cat)}')

            sample_of_cat_ids = sample_of_cat.ids
            random.shuffle(sample_of_cat_ids)

            support_ids = sample_of_cat_ids[:k]
            query_ids = sample_of_cat_ids[k:]

            support = sample_of_cat[support_ids]
            query = sample_of_cat[query_ids]

            for query_sample in query:
                yield QuerySupportSample(query_sample, support)
