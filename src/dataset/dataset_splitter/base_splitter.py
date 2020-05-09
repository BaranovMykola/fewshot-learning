import logging
import random
from typing import TYPE_CHECKING, Tuple
from abc import ABCMeta, abstractmethod

if TYPE_CHECKING:
    from src.dataset import FssSamples


class BaseSplitter(metaclass=ABCMeta):

    def __init__(self, dataset: 'FssSamples', seed: int):
        random.seed(seed)

        self.dataset = dataset
        self._logger = logging.getLogger(__name__)

    def split(self, val_ratio: float) -> Tuple:
        if val_ratio <= 0 or val_ratio >= 1:
            raise ValueError(f'validation ratio must belongs to (0, 1). Got: {val_ratio}')

        return self._split_worker(val_ratio)

    @abstractmethod
    def _split_worker(self, val_ratio: float) -> Tuple:
        pass
