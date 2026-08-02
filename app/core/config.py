# os est un module Python intégré qui permet d'interagir avec le système d'exploitation.

# Il sert notamment à :

# manipuler les chemins de fichiers ;
# créer des dossiers ;
# lire des variables d'environnement ;
# vérifier l'existence de fichiers.
import os

# Chemin racine du projet
# __file__ représente le fichier Python actuel (app/core/config.py)
# Premier dirname retire le nom du fichier
# Deuxième dirname retire encore un dossier
# Troisième dirname retire encore
# Donc BASE_DIR devient /home/user/mon-projet
# os.path.abspath(__file__): Transforme le chemin en chemin absolu (/home/user/mon-projet/app/core/config.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dossier temporaire pour les fichiers uploadés (audio/image)
# On construit le dossier temp_uploads qui contiendra (images, audio, documents PDF) temporairement car apres traitement on le supprime.
TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")

# Dossier contenant la base de connaissances (CGV/FAQ) pour le RAG
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "app", "data", "knowledge_base")

# Noms des modèles Hugging Face utilisés
WHISPER_MODEL_NAME = "openai/whisper-small"
VIT_MODEL_NAME = "google/vit-base-patch16-224"  # à adapter si tu utilises un modèle fine-tuné
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Extensions de fichiers autorisées
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Taille max autorisée par fichier (en bytes) — 10 Mo ici, à ajuster
MAX_FILE_SIZE = 10 * 1024 * 1024

# S'assurer que le dossier temporaire existe au démarrage
# Cette ligne crée temp_uploads/ si le dossier n'existe pas.
# avec exist_ok=True python dit si le dossier existe déjà, ignore l'erreur (FileExistsError)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)