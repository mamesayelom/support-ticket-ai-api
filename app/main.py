from fastapi import FastAPI
from app.api.routes import support_ticket

app = FastAPI(
    # permet de donner un nom a ton API
    title="Support Ticket AI API",
    # permet à quelqu'un qui utilise ton API de comprendre rapidement son rôle
    description="API d'analyse automatique des réclamations clients (audio, image, texte) via ASR, Vision et RAG.",
)

# On rattache le routeur des tickets support
app.include_router(
    # Le router contient toutes les routes defini dans le fichier support_ticket
    support_ticket.router,
    # Le préfixe est ajouté devant toutes les routes du routeur.
    prefix="/support-ticket",
    # Les tags servent uniquement à organiser la documentation Swagger.
    tags=["Support Ticket"]
)

@app.get("/", tags=["Health"])
def health_check():
    """Endpoint simple pour vérifier que l'API tourne."""
    return {"status": "ok", "message": "Support Ticket AI API is running"}