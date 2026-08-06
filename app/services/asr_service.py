# lru_cache est un outil Python qui permet de mettre en cache le résultat d'une fonction.
# Un cache est une mémoire temporaire.
# functools fait partie de la bibliothèque standard de Python
from functools import lru_cache
# Transformer est une architecture de réseau de neurones créée pour comprendre et traiter des données, surtout le texte, mais aussi les images et l'audio. Ce n'est pas un modèle précis, c'est plutôt une manière de construire un modèle d'IA.
# Un Transformer transforme une donnée d'entrée en une autre représentation plus utile.
# ex: Texte → compréhension numérique
# Un Transformer utilise un mécanisme appelé attention.
# L'attention permet au modèle de savoir quels mots sont importants par rapport aux autres.
# L'idée est :
# Quand un modèle lit une phrase, il ne doit pas donner la même importance à tous les mots. Certains mots sont plus importants que d'autres pour comprendre le sens.
# Prenons cette phrase : "Le chat mange la souris parce qu'il a faim."
# Un humain comprend que :
# "il" fait référence au chat ;
# "mange" est lié à chat et souris ;
# "faim" explique pourquoi il mange.
# Mais comment un ordinateur peut-il savoir cela ?
# Il regarde les relations entre les mots grâce à l'attention.
# Sans attention (ancienne méthode)
# Un ancien modèle comme un RNN lisait souvent les mots dans l'ordre : Le → chat → mange → la → souris → parce → qu'il → a → faim
# Le problème :
# Quand il arrive à faim il peut avoir oublié que chat était important plusieurs mots avant.
# Avec l'attention, le Transformer regarde tous les mots en même temps.
from transformers import pipeline

# sert à importer la bibliothèque PyTorch dans ton programme Python.
import torch
from app.core.config import WHISPER_MODEL_NAME


@lru_cache(maxsize=1)
def get_asr_pipeline():
    """
    Charge le modèle Whisper une seule fois en mémoire (Singleton).
    Grâce à @lru_cache(maxsize=1), les appels suivants renvoient
    directement l'instance déjà chargée au lieu de recharger le modèle.
    """

    # Cette ligne sert à savoir si un GPU NVIDIA est disponible.
    # Un GPU NVIDIA est un processeur graphique fabriqué par l'entreprise NVIDIA qui est utilisé pour faire des calculs très rapides, notamment pour l'intelligence artificielle.
    device = 0 if torch.cuda.is_available() else -1  # GPU si dispo, sinon CPU

    # Automatic Speech Recognition (Reconnaissance automatique de la parole)= transformer un fichier audio en texte.
    asr = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_NAME,
        # sert à dire à Hugging Face sur quel matériel (CPU ou GPU) il doit exécuter le modèle.
        device=device,
    )
    # la variable asr contient le modèle chargé en mémoire
    return asr


def transcribe_audio(file_path: str) -> str:
    """
    Transcrit un fichier audio en texte.
    :param file_path: chemin local du fichier audio temporaire
    :return: texte transcrit
    """
    # Récupérer le pipeline de reconnaissance vocale
    asr_pipeline = get_asr_pipeline()  # renvoie l'instance déjà en cache après le 1er appel
    # Envoyer le fichier audio au modèle Whisper pour qu'il le transforme en texte
    result = asr_pipeline(
        file_path,
        # pour gérer les longs audios
        # Timestamp signifie marqueur de temps.
        # Dans le contexte de Whisper (reconnaissance vocale), un timestamp indique à quel moment de l'audio une phrase ou un mot a été prononcé.
        # si l'audio est long, Whisper en a besoin pour faire la génération longue
        return_timestamps=True,
        generate_kwargs={
        "language": "fr",
        "task": "transcribe"
    }
    )
    # on retourne uniquement le texte transcrit par Whisper, en supprimant les espaces inutiles.
    return result["text"].strip()