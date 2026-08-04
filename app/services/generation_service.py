"""
Service de génération — Étape 4 du guide (Génération de la réponse).
"""

from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

GENERATION_MODEL_NAME = "bigscience/mt0-base"

# Prompt plus directif : on demande une phrase complète, adressée à l'agent
# support, avec un exemple de ton attendu.
PROMPT_TEMPLATE = """Tu es un assistant qui aide un agent du service client.

Contexte (règle applicable) : {context}

Réclamation du client : {question}

Rédige une phrase complète et claire expliquant à l'agent quelle action proposer au client, en te basant uniquement sur le contexte."""


@lru_cache(maxsize=1)
def get_generation_model():
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)
    return tokenizer, model


def generate_answer(question: str, context: str) -> str:
    tokenizer, model = get_generation_model()
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    output_tokens = model.generate(
        **inputs,
        max_new_tokens=100,
        min_new_tokens=15,        # force une réponse un minimum développée
        num_beams=4,               # recherche de phrase plus cohérente (au lieu d'un choix "gourmand" mot par mot)
        repetition_penalty=1.3,    # évite les répétitions de mots
        no_repeat_ngram_size=3,    # empêche de répéter un groupe de 3 mots
        early_stopping=True,
    )

    answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return answer.strip()