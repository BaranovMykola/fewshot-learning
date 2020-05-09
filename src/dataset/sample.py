from typing import Dict


class Sample:

    def __init__(self, image_path_str: str, mask_path_str: str, category_id: int, sample_id: int):
        self.image_path_str = image_path_str
        self.mask_path_str = mask_path_str
        self.category_id = category_id
        self.sample_id = sample_id

    @classmethod
    def from_json(cls, json_data: Dict):
        return cls(
            image_path_str=json_data['image'],
            mask_path_str=json_data['mask'],
            category_id=json_data['category_id'],
            sample_id=json_data['id']
        )
