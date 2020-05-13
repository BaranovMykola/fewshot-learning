from typing import Dict

import tensorflow as tf

from src.dataset.tfrecords import bytes_feature, int64_feature
from src.dataset.proto import fss_dataset_pb2 as fss_proto


class Sample:

    def __init__(self, image_path_str: str, mask_path_str: str, category_id: int, sample_id: int):
        self.image_path_str = image_path_str
        self.mask_path_str = mask_path_str
        self.cat_id = category_id
        self.sample_id = sample_id

    @classmethod
    def from_json(cls, json_data: Dict):
        return cls(
            image_path_str=json_data['image'],
            mask_path_str=json_data['mask'],
            category_id=json_data['category_id'],
            sample_id=json_data['id']
        )

    def __repr__(self) -> str:
        return str(self.to_json())

    def to_json(self) -> Dict:
        return {
            'image': self.image_path_str,
            'mask': self.mask_path_str,
            'category_id': self.cat_id,
            'id': self.sample_id
        }

    # def convert_to_proto_image(self) -> fss_proto.Image:
    #     return fss_proto.Image(image_path=self.image_path_str,
    #                            sample_id = self.sample_id)
    #
    # def convert_to_proto_masked_image(self) -> fss_proto.MaskedImage:
    #     return fss_proto.MaskedImage(image_path=self.image_path_str,
    #                                  mask_path=self.mask_path_str,
    #                                  sample_id = self.sample_id)
    #
    # def convert_to_proto_gt(self) -> fss_proto.FewShotGt:
    #     return fss_proto.FewShotGt(mask_path=self.mask_path_str,
    #                                sample_id=self.sample_id)
