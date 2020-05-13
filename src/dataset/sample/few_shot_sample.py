from src.dataset.proto import fss_dataset_pb2 as fss_proto

from .sample import Sample
from .sample_list import SampleList


class FewShotSample:

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

    def convert_to_proto(self) -> fss_proto.FewShotSample:

        masked_images = []
        for i in self.support:
            masked_image = fss_proto.MaskedImage()
            masked_image.image_path = i.image_path_str
            masked_image.mask_path = i.mask_path_str
            masked_images.append(masked_image)

        sample = fss_proto.FewShotSample(query_image_path=self.query.image_path_str,
                                         support=masked_images)
        return sample

        # input_image = self.query.convert_to_proto_image()
        # support_images = [x.convert_to_proto_masked_image() for x in self.support]
        #
        # context = fss_proto.FewShotSampleContext(category_id=self.category)
        # query = fss_proto.QuerySet(content=input_image)
        # support = fss_proto.SupportSet(content=support_images)
        #
        # input_proto = fss_proto.FewShotInput(query=query,
        #                                      support=support)
        # gt = self.query.convert_to_proto_gt()
        #
        # few_shot_sample = fss_proto.FewShotSample(context=context,
        #                                           input=input_proto,
        #                                           gt=gt)
        #
        # return few_shot_sample
        # # return input_image
