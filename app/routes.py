import cv2
import numpy as np
from fastapi import APIRouter
from modules.pre_process import pre_process
from modules.detect import detect
from modules.post_process import post_process
from app.models import DetectedObject, ObjectDetectionRequest, ObjectDetectionResponse

router = APIRouter()


@router.post("/api/image", response_model=ObjectDetectionResponse)
async def object_detection(request: ObjectDetectionRequest):
    contents = request.file
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_UNCHANGED)

    image = pre_process(image)
    boxes, confidences, class_ids = detect(image)
    objects = post_process(boxes, confidences, class_ids)

    detected_objects = []
    for obj_id, obj_class, obj_box, obj_score, obj_related in objects:
        detected_object = DetectedObject(id=obj_id, name=obj_class, box=obj_box, confidence=obj_score)
        for related_obj_id, relation_score in obj_related:
            detected_object.add_related_object(obj_id=related_obj_id, relation_score=relation_score)
        detected_objects.append(detected_object)

    response = ObjectDetectionResponse(objects=detected_objects)
    return response
