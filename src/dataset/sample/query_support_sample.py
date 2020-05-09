from .sample import Sample
from .sample_list import SampleList


class QuerySupportSample:

    def __init__(self, query: Sample, support: SampleList):
        if len(set(support.cat_ids)) != 1:
            raise ValueError(f'Invalid support set. Set contains sample of different categories or set is empty')

        self.query = query
        self.support = support
