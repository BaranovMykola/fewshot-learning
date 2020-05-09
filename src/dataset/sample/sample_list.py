import itertools
from typing import List, Iterable, Union

from src.dataset.category import Category
from src.dataset.category.category_list import CategoryList
from src.dataset.sample import Sample


class SampleList:
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    @classmethod
    def from_json(cls, json_data: List) -> 'SampleList':
        return cls([Sample.from_json(s) for s in json_data])

    @classmethod
    def merge(cls, *args: 'SampleList') -> 'SampleList':
        samples = [s.samples for s in args]
        merged_samples = [*itertools.chain.from_iterable(samples)]
        return cls(merged_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterable[Sample]:
        return iter(self.samples)

    def __repr__(self) -> str:
        return f'<#{len(self)} sample>'

    def __getitem__(self, item: Union[Category, CategoryList, int, List[int]]) -> 'SampleList':
        if isinstance(item, CategoryList):
            return SampleList.merge(*[self[c] for c in item])

        if isinstance(item, Category):
            filtered_samples = [*filter(lambda x: x.cat_id == item.cat_id, self)]
            return SampleList(filtered_samples)

        if isinstance(item, list):
            filtered_items = [*filter(lambda s: s.sample_id in item, self)]
            return SampleList(filtered_items)

        filtered_items = [*filter(lambda s: s.sample_id == item, self)]

        if len(filtered_items) != 1:
            raise ValueError('Failed to extract sample due to invalid id')

        return filtered_items[0]

    @property
    def ids(self) -> List[int]:
        return [s.sample_id for s in self]

    @property
    def cat_ids(self) -> List[int]:
        return [s.cat_id for s in self]
