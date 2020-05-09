import itertools
from typing import List, Iterable, Union

from .category import Category
from .category_list import CategoryList
from .sample import Sample


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

    def __getitem__(self, item: Union[Category, CategoryList]) -> 'SampleList':
        if isinstance(item, CategoryList):
            return SampleList.merge(*[self[c] for c in item])

        if isinstance(item, Category):
            filtered_samples = [*filter(lambda x: x.cat_id == item.cat_id, self)]
            return SampleList(filtered_samples)

    @property
    def ids(self) -> List[int]:
        return [s.sample_id for s in self]
