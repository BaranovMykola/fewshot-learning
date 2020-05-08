import sys
from pathlib import Path

from src.model import Model
from src.model.estimator import Estimator
from src.utils import is_gpu, setup_gpu


def main():
    if is_gpu():
        setup_gpu()

    estimator = Estimator(Model(), Path('./dataset.bin'), 2)
    estimator.compile()
    estimator.train()


    return 0

if __name__ == '__main__':
    code = main()
    sys.exit(code)
