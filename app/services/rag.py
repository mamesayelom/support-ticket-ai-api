"""
Service RAG — suit exactement la démarche du guide appris en cours :
    Étape 1 : loader + splitter → chunks
    Étape 2 : embeddings + vector_store → indexation
    Étape 3 : vector_store.as_retriever() → retriever.invoke(query)

Deux différences uniquement, obligatoires pour notre contexte :
    - OpenAIEmbeddings() remplacé par HuggingFaceEmbeddings() (gratuit, local,
      pas de clé API), car on n'a pas de compte OpenAI payant pour ce projet.
    - On s'arrête à l'Étape 3 (retriever), pas d'Étape 4 (génération LLM),
      car la consigne demande de "récupérer la politique applicable",
      pas de générer une réponse en langage naturel.
"""

from typing import Optional
# Permet de manipuler les chemins
import os
# permet d'exécuter une fonction une seule fois et garde le résultat en mémoire
from functools import lru_cache
#Lire un PDF et transformer son contenu en documents LangChain.
from langchain_community.document_loaders import PyPDFLoader
# Découper un gros texte en petits morceaux.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# HuggingFaceEmbeddings charge un modèle Hugging Face comme modèle d'embedding.
from langchain_huggingface import HuggingFaceEmbeddings
# base de données vectorielle
from langchain_community.vectorstores import Chroma

# KNOWLEDGE_BASE_DIR: variable représentant le chemin vers le dossier qui contient les documents de connaissance de ton RAG.
from app.core.config import KNOWLEDGE_BASE_DIR, EMBEDDING_MODEL_NAME

# module Python qui permet d'utiliser les expressions régulières (regex)
import re
from langchain_core.documents import Document

# Score minimum accepté
# Plus le score est proche de 1,
# plus le texte est proche de la question.
MIN_SIMILARITY = 0.70

@lru_cache(maxsize=1)
def get_vector_store():
    """
    Construit le retriever (Le retriever est l'objet qui nous permet de trouver le contexte) une seule fois. Mis en cache pour ne pas refaire l'indexation à chaque
    requête HTTP.
    """

    # construire le chemin complet vers ton fichier PDF
    pdf_path = os.path.join(KNOWLEDGE_BASE_DIR, "cgv_faq.pdf")
    # loader est un objet qui sait lire un PDF
    loader = PyPDFLoader(pdf_path)
    # lit le PDF et retourne une liste de Document
    pages = loader.load()

    # Fusionne le texte de toutes les pages du PDF en une seule chaîne,
    # en séparant chaque page par un retour à la ligne. 
    full_text = "\n".join(
        page.page_content for page in pages
    )

    # Découpe le texte complet en sections (articles et questions de la FAQ)
    def split_articles(text):
        """
        sections = [
            "Article 1 - Produit endommagé",
            "Contenu article 1",
            "Article 2 - Produit non conforme",
            "Contenu article 2",
            "Article 3 - Retard livraison",
            "Contenu article 3"
        ]
        """
        # On découpe le texte en utilisant comme séparateurs les titres des articles, la section FAQ et les questions
        # sections est la liste des morceaux de texte obtenus après le découpage
        sections = re.split(
            # rticle \d+ - .+
            # Article → cherche le mot Article
            # \d+ → cherche un ou plusieurs chiffres (1, 2, 3...)
            # - .+ → récupère le titre après le tiret
            r"(Article \d+ - .+|FAQ - .+|Q: .+)",
            text
        )

        # Liste qui va contenir les documents
        documents = []

        for i in range(1, len(sections), 2):
            # Récupère le titre de l'article
            title = sections[i]
            # Récupère le contenu qui suit le titre.
            # Si aucun contenu n'existe après le titre, on met une chaîne vide.
            content = sections[i + 1] if i + 1 < len(sections) else ""

            # Si le titre commence par "FAQ -", ignore-le et passe directement au prochain élément.
            if title.startswith("FAQ -"):
                continue

            # Création d'un Document LangChain contenant :
            # - le titre + le contenu de l'article
            # - la source pour retrouver l'origine du document
            document = Document(
                page_content=f"{title}\n{content}".strip(),
                metadata={"source": title}
            )

            documents.append(document)

        return documents
    
    documents = split_articles(full_text)

    # charger le modèle d'embedding
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # on crée une base vectorielle à partir d'une liste de documents.
    # vector_store contient la base de données vectorielle
    vector_store = Chroma.from_documents(
        # donner les morceaux
        documents=documents,
        # le modèle qui transforme le texte en vecteur
        embedding=embeddings,
        # crées une collection appelée cgv_faq
        collection_name="cgv_faq",
        # sauvegarder la base Chroma sur le disque
        persist_directory=os.path.join(KNOWLEDGE_BASE_DIR, "store"),
        # On utilise cosine similarity
        #
        # Résultat :
        # 1 = très proche
        # 0 = pas proche
        collection_metadata={
            "hnsw:space": "cosine"
        }
    )

    return vector_store

# distance entre deux vecteur
DISTANCE_THRESHOLD = 10
def search_relevant_rule(query_text: str) -> Optional[dict]:
    """
    Utilise le retriever pour trouver la règle la plus pertinente.
    On s'arrête là (pas d'Étape 4 : génération LLM).

    La première fois retriever = get_retriever() va:
    charger le PDF
    découper le texte
    créer les embeddings
    créer Chroma
    créer le retriever

    les fois suivantes, il réutilise le même objet.
    """
    # un objet capable de chercher .
    vector_store = get_vector_store()

    
    # similarity_search_with_score renvoie une DISTANCE L2 (plus petit = plus pertinent)
    #results = vector_store.similarity_search_with_score(
       # query_text,
       # k=3
    #)

    # Recherche des 3 documents les plus proches
    # similarity_search_with_relevance_score renvoie une VRAIE similarité
    # normalisée entre 0 et 1 (contrairement à similarity_search_with_score
    # qui renvoie une distance brute, même en mode cosine).
    results = vector_store.similarity_search_with_relevance_scores(
        query_text,
        k=3
    )

    if not results:
        return None

    best_doc, score = results[0]

    # # Avec cosine :
    # score proche de 1 = meilleur
    if score < MIN_SIMILARITY:
        return None

    return {
        # Texte de l'article trouvé
        "rule_text": best_doc.page_content,
        # Nom de l'article
        "source": best_doc.metadata.get("source", "inconnu"),
        # Score de pertinence
        "similarity_score": round(float(score), 3),
    }