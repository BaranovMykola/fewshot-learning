import random
from typing import Tuple, List

from src.dataset import FssSamples

from .base_splitter import BaseSplitter


class NovelSplitter(BaseSplitter):

    def _split_worker(self, val_ratio: float) -> Tuple:
        train_cats_count, val_cats_count = self._split_classes_count(val_ratio)
        train_cats_id, val_cats_id = self._extract_train_val_ids(train_cats_count)

        train_cats = self.dataset.categories[train_cats_id]
        test_cats = self.dataset.categories[val_cats_id]

        train_samples = self.dataset.samples[train_cats]
        test_samples = self.dataset.samples[test_cats]

        train = FssSamples(train_samples, train_cats)
        test = FssSamples(test_samples, test_cats)

        return train, test


    def _split_classes_count(self, val_ratio: float) -> Tuple[int, int]:
        cats_count = len(self.dataset.categories)
        val_cats_count = int(cats_count * val_ratio)
        train_cats_counts = cats_count - val_cats_count

        assert val_cats_count + train_cats_counts == cats_count, f'Invalid split: {val_cats_count},' \
                                                                 f'{train_cats_counts} from {cats_count}'

        self._logger.info(f'Discover {cats_count} in dataset. Split to #{val_cats_count} validation and '
                          f'#{train_cats_counts} train categories')

        return train_cats_counts, val_cats_count

    def _extract_train_val_ids(self, train_cats_count: int) -> Tuple[List[int], List[int]]:
        shuffled_categories = random.sample(self.dataset.categories.ids, k=len(self.dataset.categories))
        train_cats_id = shuffled_categories[:train_cats_count]
        val_cats_id = shuffled_categories[train_cats_count:]

        return train_cats_id, val_cats_id
