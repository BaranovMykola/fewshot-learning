from typing import Dict


class Category:
    def __init__(self, name: str, cat_id: int):
        self.name = name
        self.cat_id = cat_id

    @classmethod
    def from_json(cls, json_data: Dict):
        return cls(
            name=json_data['name'],
            cat_id=json_data['id']
        )

    def __repr__(self) -> str:
        return str(self.to_json())

    def to_json(self) -> Dict:
        return {'name': self.name, 'id': self.cat_id}
