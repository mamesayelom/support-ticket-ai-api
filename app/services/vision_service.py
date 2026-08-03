from functools import lru_cache
from transformers import pipeline
import torch
from PIL import Image, UnidentifiedImageError
import io
from app.core.config import VIT_MODEL_NAME, MAX_CONFIDENCE, MIN_CONFIDENCE


@lru_cache(maxsize=1)
def get_vision_pipeline():
    """
    Charge le modèle ViT une seule fois en mémoire (Singleton).
    """
    device = 0 if torch.cuda.is_available() else -1
    vision = pipeline(
        "zero-shot-image-classification",
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

    # Classes métier données au modèle
    candidate_labels = [
        "a good quality product",
        "a damaged broken or defective product"
    ]

    # Analyse zero-shot
    predictions = vision_pipeline(
        image,
        candidate_labels=candidate_labels
    )
    # # Premier résultat = classe avec le meilleur score
    top_prediction = predictions[0]  

    label = top_prediction["label"]
    confidence = top_prediction["score"]


    # Conversion en statut métier
    if (
    label == "a good quality product"
    and confidence >= MAX_CONFIDENCE
    ):
        status = "conforme"

    elif (
        label == "a damaged broken or defective product"
        and confidence >= MAX_CONFIDENCE
    ):
        status = "defaut"

    else:
        status = "a_verifier"



    return {
        "status": status,
        "label": label,
        "confidence": round(confidence, 3)
    }