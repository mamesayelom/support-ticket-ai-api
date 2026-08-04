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


@lru_cache(maxsize=1)
def get_retriever():
    """
    Construit le retriever (Le retriever est l'objet qui nous permet de trouver le contexte) une seule fois. Mis en cache pour ne pas refaire l'indexation à chaque
    requête HTTP.
    """

    # construire le chemin complet vers ton fichier PDF
    pdf_path = os.path.join(KNOWLEDGE_BASE_DIR, "cgv_faq.pdf")
    # Charger le fichier PDF
    loader = PyPDFLoader(pdf_path)

    # objet capable de découper du texte
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        # tokenizer (transforme un texte en tokens) utiliser
        encoding_name="cl100k_base",
        # la taille maximale d'un morceau
        chunk_size=300,
        # combien de tokens doivent être répétés entre deux morceaux
        chunk_overlap=20,
    )

    # Lire le PDF, extrait son texte et le découpe en petits morceaux
    chunks = loader.load_and_split(text_splitter=splitter)

    # charger le modèle d'embedding
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # on crée une base vectorielle à partir d'une liste de documents.
    # vector_store contient la base de données vectorielle
    vector_store = Chroma.from_documents(
        # donner les morceaux
        documents=chunks,
        # le modèle qui transforme le texte en vecteur
        embedding=embeddings,
        # crées une collection appelée cgv_faq
        collection_name="cgv_faq",
        # sauvegarder la base Chroma sur le disque
        persist_directory=os.path.join(KNOWLEDGE_BASE_DIR, "store"),
    )

    # Créer le retriever ---
    retriever = vector_store.as_retriever(
        # méthode de recherche
        search_type="similarity",
        # Combien de résultats retourner
        search_kwargs={"k": 3},
    )

    return retriever


def search_relevant_rule(query_text: str) -> dict:
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
    retriever = get_retriever()

    # Envoie la question au retriever
    docs = retriever.invoke(query_text)

    best_doc = docs[0]  # k=1, donc un seul résultat

    return {
        "rule_text": best_doc.page_content,   # page_content contient le texte du chunk.      
        "source": f"page {best_doc.metadata.get('page', '?')}",   # récupère la page d'origine
        "similarity_score": None,  # LangChain .invoke() ne renvoie pas le score par défaut
    }












