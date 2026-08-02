from fastapi import FastAPI
from app.api.routes import support_ticket

app = FastAPI(
    title="Support Ticket AI API",
    description="API d'analyse automatique des réclamations clients (audio, image, texte) via ASR, Vision et RAG.",
    version="1.0.0",
)

# On rattache le routeur des tickets support
app.include_router(support_ticket.router, prefix="/support-ticket", tags=["Support Ticket"])


@app.get("/", tags=["Health"])
def health_check():
    """Endpoint simple pour vérifier que l'API tourne."""
    return {"status": "ok", "message": "Support Ticket AI API is running"}