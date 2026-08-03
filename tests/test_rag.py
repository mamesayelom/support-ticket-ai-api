
from app.services.rag_service import search_relevant_rule

result = search_relevant_rule("mon produit est arrivé cassé")

print(result)
