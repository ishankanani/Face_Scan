# image_upload/face_engine.py
from functools import lru_cache
from deepface import DeepFace

@lru_cache(maxsize=1)
def get_facenet_model():
    return DeepFace.build_model("Facenet")
