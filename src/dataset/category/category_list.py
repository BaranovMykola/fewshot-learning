from typing import List, Iterable, Union

from src.dataset.category import Category


class CategoryList:
    def __init__(self, categories: List[Category]):
        self._ids = [x.cat_id for x in categories]
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

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, cat_id: Union[int, slice, List[int]]) -> Union[Category, 'CategoryList']:
        if isinstance(cat_id, slice):
            return self._slice_by_selected_cats(sorted(self.ids)[cat_id])

        if isinstance(cat_id, Iterable):
            return self._slice_by_selected_cats([*cat_id])

        if cat_id not in self.ids:
            raise ValueError(f'Category {cat_id} does not exist')

        cats = [c for c in self if c.cat_id == cat_id]

        if len(cats) > 1:
            raise RuntimeError(f'More than one category found by id {cat_id}')

        return cats[0]

    def __repr__(self) -> str:
        return f'<#{len(self)} categories>'

    def _slice_by_selected_cats(self, selected_cats: List[int]) -> 'CategoryList':
        categories = [self[x] for x in selected_cats]
        return CategoryList(categories)
