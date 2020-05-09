from typing import List, Iterable

from .sample import Sample


class SampleList:
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    @classmethod
    def from_json(cls, json_data: List):
        return cls([Sample.from_json(s) for s in json_data])

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterable[Sample]:
        return iter(self.samples)

    def __repr__(self) -> str:
        return f'<#{len(self)} sample>'
