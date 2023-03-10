from typing import List
from pydantic import BaseModel


class RelatedObject(BaseModel):
    id: int
    r_score: float


class DetectedObject(BaseModel):
    id: int
    name: str
    confidence: float
    box: List[int]
    related_objects: List[RelatedObject] = []

    def add_related_object(self, obj_id, relation_score):
        self.related_objects.append(RelatedObject(id=obj_id, r_score=relation_score))


class ObjectDetectionRequest(BaseModel):
    file: bytes


class ObjectDetectionResponse(BaseModel):
    objects: List[DetectedObject]
