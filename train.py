import sys
from pathlib import Path

from src.model import Model
from src.model.estimator import Estimator
from src.utils import is_gpu, setup_gpu


def main():
    if is_gpu():
        setup_gpu()

    estimator = Estimator(Model(),
                          train_dataset='./datasets/novel/train.tfrecord',
                          val_dataset='./datasets/novel/val.tfrecord',
                          batch_size=2,
                          data_root='./Data')
    estimator.compile()
    estimator.train()


    return 0

if __name__ == '__main__':
    code = main()
    sys.exit(code)
