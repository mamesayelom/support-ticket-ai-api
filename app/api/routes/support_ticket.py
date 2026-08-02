# UploadFile représente un fichier envoyé par un utilisateur
# File() indique à FastAPI : Ce paramètre vient d'un fichier envoyé dans la requête HTTP.
# Form() indique que la donnée vient d'un formulaire HTTP.
from fastapi import APIRouter, UploadFile, File, Form

#s on importe Optional depuis le module Python typing
# Optional permet d'indiquer qu'une variable ou un paramètre peut avoir une valeur ou être vide (None)
from typing import Optional
from app.models.schemas import SupportTicketResponse, TicketStatus, VisionDiagnostic

#importer pour communiquer avec le système
import os
# uuid sert à générer des identifiants uniques pour chaque fichier uploader sinon le deuxieme fichier ecrase le premier.
import uuid
from app.services.asr_service import transcribe_audio
from app.core.config import TEMP_UPLOAD_DIR, ALLOWED_AUDIO_EXTENSIONS, ALLOWED_IMAGE_EXTENSIONS
from app.services.vision_service import analyze_image

# sert à créer un groupe de routes.
router = APIRouter()

def save_temp_file(upload_file: UploadFile, allowed_extensions: set) -> str:
    # upload_file contient:
    # upload_file.filename qui le nom du fichier
    # upload_file.file qui est le contenu du fichier
    # audio.content_type qui est le type du fichier
    """
    Sauvegarde un fichier uploadé dans le dossier temporaire puis retourner son chemin.
    Lève une ValueError si l'extension n'est pas autorisée(fichier non autorisé).
    """
    # obtenir l'extension d'un fichier en séparant le nom d'un fichier et son extension.
    # splitext() signifie split extension → séparer l'extension.
    # Elle retourne un tuple contenant deux éléments : (nom_du_fichier, extension)
    # os.path est un module qui contient des fonctions pour manipuler les chemins
    ext = os.path.splitext(upload_file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise ValueError(f"Extension de fichier non autorisée : {ext}")

    # créer un nom de fichier temporaire unique
    temp_filename = f"{uuid.uuid4()}{ext}"
 
    # join() assemble plusieurs morceaux de texte pour former un chemin valide (temp_uploads/audio.wav)
    # temp_path contient le chemin
    # os.path.join() choisit automatiquement le bon séparateur selon le système d'exploitation (ex: temp_uploads\audio.wav sous windows)
    temp_path = os.path.join(TEMP_UPLOAD_DIR, temp_filename)

    # wb: Ouvre ce fichier pour écrire des données binaires.
    with open(temp_path, "wb") as f:
        # upload_file.file.read(): Lire tous les octets du fichier envoyé.
        # Écrire dans le fichier temporaire
        f.write(upload_file.file.read())

    # Retourner le chemin parce que le service Whisper a besoin de savoir où se trouve le fichier.
    return temp_path

def validate_extension(filename: str, allowed_extensions: set) -> None:
    """Valide l'extension d'un fichier sans l'écrire sur le disque."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise ValueError(f"Extension de fichier non autorisée : {ext}")

# FastAPI va valider automatiquement que ce qu'on retourne respecte bien le schéma Pydantic qu'on a défini avant.
@router.post("", response_model=SupportTicketResponse)
# FastAPI peut gérer plusieurs utilisateurs en même temps.
async def create_support_ticket(
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """
    Reçoit une réclamation client (audio, image et/ou texte) et retourne
    un diagnostic structuré pour aiguiller le ticket.
    """

    # texte produit par Whisper
    transcribed_text = None
    # le chemin du fichier temporaire
    temp_audio_path = None

    vision_diagnostic = None

    try:
        # --- Traitement audio (ASR) ---
        #transformers s'appuie en interne sur un outil externe appelé ffmpeg — un logiciel de traitement audio/vidéo qui n'est pas une librairie Python, mais un programme système
        if audio is not None:
            temp_audio_path = save_temp_file(audio, ALLOWED_AUDIO_EXTENSIONS)
            transcribed_text = transcribe_audio(temp_audio_path)

         # --- Traitement image (Vision) ---
        if image is not None:
            validate_extension(image.filename, ALLOWED_IMAGE_EXTENSIONS)
            # lit tout le contenu du fichier
            image_bytes = await image.read()
            vision_result = analyze_image(image_bytes)

            #VisionDiagnostic(
                #label="telephone",
                #confidence=0.95
            #)
            vision_diagnostic = VisionDiagnostic(**vision_result)


        # --- RAG seront branchés dans les prochaines étapes ---

        # Squelette temporaire — les services ASR/Vision/RAG seront branchés
        # dans les prochaines étapes.
        return SupportTicketResponse(
            transcribed_text=transcribed_text,
            vision_diagnostic=vision_diagnostic,
            rag_result=None,
            proposed_status=TicketStatus.A_VERIFIER,
            message="Audio traité." if transcribed_text else "Aucun audio fourni.",
        )

    finally:
        # --- Nettoyage garanti du fichier temporaire ---
        # On vérifie si on a créé un fichier temporaire et qu'il existe encore
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)







# note:
# transcribe_audio() a besoin d'un chemin de fichier (Whisper/ffmpeg lit depuis un fichier) → donc pour l'audio, écrire sur disque est nécessaire, pas juste une habitude.
# analyze_image() accepte des bytes directement (Image.open(io.BytesIO(image_bytes))) → donc pour l'image, écrire sur disque est inutile
        