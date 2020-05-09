import json
from pathlib import Path
from typing import Dict

from src.dataset.category.category_list import CategoryList
from src.dataset.sample.sample_list import SampleList


class FssSamples:
    JSON_SUFFIX = '.json'

    def __init__(self, samples: SampleList, categories: CategoryList):

        self._samples = samples
        self._categories = categories

        if not all([s.cat_id in self.categories.ids for s in self.samples]):
            raise RuntimeError('Some of images has no corresponded category in categories list')


    @classmethod
    def from_json_file(cls, json_file: Path):
        if not json_file.exists() or json_file.suffix != FssSamples.JSON_SUFFIX:
            raise OSError(f'Provided file is not json file: {json_file}. Check this file')

        with json_file.open() as f:
            json_data = json.load(f)

        return cls.from_json(json_data)

    @classmethod
    def from_json(cls, json_data: Dict):
        samples = SampleList.from_json(json_data['samples'])
        categories = CategoryList.from_json(json_data['categories'])
        return cls(samples, categories)

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self):
        return f'<#{len(self.samples)} samples> ,#{len(self.categories)} categories>'

    # def category_name(self, cat_id: int) -> :
    #     cat_names = [x for x in self.categories if x['id'] == cat_id]
    #     if len(cat_names) == 0:
    #         raise ValueError(f'No such category: {cat_id}')
    #     if len(cat_names) > 1:
    #         raise ValueError(f'multiple categories found: {cat_names}')
    #
    #     return cat_names[0]

    @property
    def samples(self) -> SampleList:
        return self._samples

    @property
    def categories(self) -> CategoryList:
        return self._categories
