from app.services.generation_service import generate_answer


question = "mon produit est arrivé cassé"

context = """
Article 1 - Produit endommagé à la livraison.
Si le produit reçu est endommagé, cassé, brisé ou abîmé dès la réception,
le client dispose de 14 jours pour signaler le problème.
Le remboursement est intégral.
"""


answer = generate_answer(question, context)

print("QUESTION :", question)
print("REPONSE :", answer)