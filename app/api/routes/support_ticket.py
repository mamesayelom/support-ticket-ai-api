from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from app.models.schemas import SupportTicketResponse, TicketStatus

router = APIRouter()

# FastAPI va valider automatiquement que ce qu'on retourne respecte bien le schéma Pydantic qu'on a défini avant.
@router.post("", response_model=SupportTicketResponse)
async def create_support_ticket(
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    """
    Reçoit une réclamation client (audio, image et/ou texte) et retourne
    un diagnostic structuré pour aiguiller le ticket.
    """
    # Squelette temporaire — les services ASR/Vision/RAG seront branchés
    # dans les prochaines étapes.
    return SupportTicketResponse(
        transcribed_text=None,
        vision_diagnostic=None,
        rag_result=None,
        proposed_status=TicketStatus.A_VERIFIER,
        message="Endpoint reçu — traitement IA à venir.",
    )