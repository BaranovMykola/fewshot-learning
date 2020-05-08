import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import tensorflow as tf

# tf.config.experimental_run_functions_eagerly(True)

from src.model import Model, TbCallback, IoU, F1Score
from src.model.estimator import Estimator
from src.utils import assert_gpu, is_gpu, setup_gpu
from src.dataset.fss import FssDataset

def main(args):
    if is_gpu():
        setup_gpu()
        
    estimator = Estimator(Model(), Path('./dataset.bin'), 2)
    estimator.compile()
    estimator.train()

    return 0

if __name__ == '__main__':
    code = main(None)
    sys.exit(code)
