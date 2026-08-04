from pydantic import BaseModel
from typing import Optional
from enum import Enum

# avec enum, la classe contient une liste fixe de valeurs possibles.
# avec str, status = TicketStatus.REMBOURSABLE donne Remboursable
# sans str, status = TicketStatus.REMBOURSABLE donne status = TicketStatus.REMBOURSABLE
# on fait un héritage multiple, la classe devient à la fois une chaîne de caractères et une énumération
class TicketStatus(str, Enum):
    """Statuts possibles proposés pour un ticket de réclamation."""
    REMBOURSABLE = "Remboursable"
    A_VERIFIER = "À vérifier"
    REFUSE = "Refusé"

# Format de reponse
class VisionDiagnostic(BaseModel):
    """Résultat de l'analyse d'image (si une image a été fournie)."""
    status: str
    label: str          # ex: "produit endommagé", "conforme"
    confidence: float    # score de confiance du modèle (0 à 1)


class RagResult(BaseModel):
    """Règle interne trouvée via le RAG (CGV/FAQ)."""
    rule_text: str       # le passage pertinent trouvé dans la base de connaissances
    source: str          # ex: "CGV - Article 5" ou "FAQ - Retours"
    similarity_score: Optional[float] = None


class SupportTicketResponse(BaseModel):
    """Réponse finale renvoyée par l'endpoint POST /support-ticket."""
    transcribed_text: Optional[str] = None       # texte transcrit si audio fourni(un ticket peut arriver sans audio (donc transcribed_text = None) ou sans image (vision_diagnostic = None))
    vision_diagnostic: Optional[VisionDiagnostic] = None  # résultat image si image fournie
    rag_result: Optional[RagResult] = None        # règle interne trouvée
    assistant_response: Optional[str] = None 
    proposed_status: TicketStatus                 # statut final proposé
    message: str                                  # résumé lisible pour l'agent support