import numpy as np

from src.model.model import Model


q = np.zeros([4, 1, 224, 224, 3], np.float32)
s = np.zeros([4, 5, 224, 224, 3], np.float32)
model = Model()

model((q, s))
