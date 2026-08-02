from functools import lru_cache
from transformers import pipeline
import torch
from PIL import Image, UnidentifiedImageError
import io
from app.core.config import VIT_MODEL_NAME


@lru_cache(maxsize=1)
def get_vision_pipeline():
    """
    Charge le modèle ViT une seule fois en mémoire (Singleton).
    """
    device = 0 if torch.cuda.is_available() else -1
    vision = pipeline(
        "image-classification",
        model=VIT_MODEL_NAME,
        device=device,
    )
    return vision


def analyze_image(image_bytes: bytes) -> dict:
    """
    Analyse une image et retourne un diagnostic (label + score de confiance).
    :param image_bytes: contenu binaire brut de l'image
    :return: dict avec 'label' et 'confidence'
    """
    vision_pipeline = get_vision_pipeline()
    try:
        # BytesIO transforme les données brutes en un fichier en mémoire
        # Pillow lit ce "fichier virtuel" puis il crée un objet Image.
        image = Image.open(io.BytesIO(image_bytes))
        #Pillow, vérifie que la structure interne de cette image est correcte.
        image.verify()

        # # Réouverture pour utilisation par le modèle et mettre toutes les images dans un format standard que le modèle IA comprend
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Le fichier fourni n'est pas une image valide.")

    predictions = vision_pipeline(image)
    top_prediction = predictions[0]  # le pipeline retourne les résultats triés par score décroissant

    return {
        "label": top_prediction["label"],
        "confidence": round(top_prediction["score"], 3),
    }