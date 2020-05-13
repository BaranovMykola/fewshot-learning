from src.dataset.proto.few_shot_sample import MaskedImage, FewShotSample

from .sample import Sample
from .sample_list import SampleList


class FewShotSampleStruct:

    def __init__(self, query: Sample, support: SampleList):
        cat_set = set(support.cat_ids)
        if len(cat_set) != 1:
            raise ValueError('Invalid support set. Set contains sample of different categories or set is empty')

        self.category = [*cat_set][0]
        if query.cat_id != self.category:
            raise ValueError('Query sample does not match support category')

        self.query = query
        self.support = support

    def __repr__(self) -> str:
        return str({'query': repr(self.query), 'support': self.support})

    def convert_to_proto(self) -> FewShotSample:

        masked_images = []
        for i in self.support:
            masked_image = MaskedImage()
            masked_image.image_path = i.image_path_str
            masked_image.mask_path = i.mask_path_str
            masked_images.append(masked_image)

        sample = FewShotSample(query_image_path=self.query.image_path_str,
                               support=masked_images,
                               gt_mask_path=self.query.mask_path_str)
        return sample
