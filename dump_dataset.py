import pickle

from src.dataset.fss import FssDataset

dataset = FssDataset('./fewshots.csv')
with open('./dataset.bin', 'wb+') as f:
    pickle.dump(dataset, f)
