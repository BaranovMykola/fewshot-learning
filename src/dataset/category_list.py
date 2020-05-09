from typing import List, Iterable

from .category import Category


class CategoryList:
    def __init__(self, categories: List[Category]):
        self._ids = [x.cat_id for x in self.categories]
        if len(self.ids) != len(set(self.ids)):
            raise ValueError(f'List of categories contains not unique ids')

        self.categories = categories

    @classmethod
    def from_json(cls, json_data: List):
        return cls([Category.from_json(c) for c in json_data])

    @property
    def ids(self) -> List[int]:
        return self._ids

    def __iter__(self) -> Iterable[Category]:
        return iter(self.categories)
