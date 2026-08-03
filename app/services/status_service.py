from app.models.schemas import TicketStatus


def determine_ticket_status(
    rag_result: dict | None,
    vision_result: dict | None,
    text: str | None
) -> TicketStatus:
    """
    Détermine le statut proposé du ticket selon :
    - la règle trouvée par le RAG
    - l'analyse image
    - la description client
    """

    if not rag_result:
        return TicketStatus.A_VERIFIER


    rule = rag_result["source"].lower()


    # -------------------------------------------------
    # Produit endommagé à la livraison
    # Article 1
    # -------------------------------------------------
    if "produit endommagé à la livraison" in rule:

        # Une preuve image est recommandée
        if vision_result:
            return TicketStatus.REMBOURSABLE

        return TicketStatus.A_VERIFIER



    # -------------------------------------------------
    # Produit non conforme à la description
    # Article 2
    # -------------------------------------------------
    if "produit non conforme" in rule:

        return TicketStatus.REMBOURSABLE



    # -------------------------------------------------
    # Retour simple
    # Article 5
    # -------------------------------------------------
    if "retour simple" in rule:

        return TicketStatus.A_VERIFIER



    # -------------------------------------------------
    # Produit défectueux après usage
    # Article 6
    # -------------------------------------------------
    if "défectueux après usage" in rule:

        return TicketStatus.A_VERIFIER



    # -------------------------------------------------
    # Cas de refus
    # Article 7
    # -------------------------------------------------
    if "refus de remboursement" in rule:

        return TicketStatus.REFUSE



    # Cas par défaut
    return TicketStatus.A_VERIFIER