# Support Ticket AI API

API backend/IA qui automatise la première analyse des réclamations clients reçues via messagerie (audio, image, texte), pour aiguiller instantanément les tickets vers l'équipe support.

Projet réalisé en solo par **Mame Sayelom**, dans le cadre de la formation Simplon Sénégal.

---

## Table des matières

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Architecture globale](#2-architecture-globale)
3. [Structure du projet](#3-structure-du-projet)
4. [Prérequis](#4-prérequis)
5. [Installation](#5-installation)
6. [Configuration](#6-configuration)
7. [Utilisation de l'API](#7-utilisation-de-lapi)
8. [Détail des services IA](#8-détail-des-services-ia)
9. [Logique du statut proposé](#9-logique-du-statut-proposé)
10. [Gestion des exceptions et robustesse](#10-gestion-des-exceptions-et-robustesse)
11. [Gestion de projet (Git Flow / Kanban)](#13-gestion-de-projet-git-flow--kanban)
12. [Glossaire](#14-glossaire)

---

## 1. Contexte et objectif

Dans une entreprise e-commerce, le service client reçoit des réclamations sous plusieurs formes : notes vocales, photos de produits endommagés, messages texte. Cette API automatise une première analyse de chaque réclamation pour :

- **Transcrire** l'audio en texte (ASR)
- **Analyser** l'image du produit (Vision)
- **Retrouver** la règle interne applicable (CGV/FAQ) via une recherche sémantique (RAG)
- **Générer** une phrase d'aide pour l'agent support (bonus, LLM léger)
- **Proposer** un statut de ticket (Remboursable / À vérifier / Refusé)

Le tout est exposé via un unique endpoint `POST /support-ticket`, démontrable via Swagger (`/docs`).

---

## 2. Architecture globale

```
                     ┌─────────────────────────┐
                     │   POST /support-ticket    │
                     │   (audio, image, texte)   │
                     └────────────┬───────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                          │
        ▼                         ▼                          ▼
 ┌──────────────┐        ┌──────────────┐          ┌──────────────────┐
 │  ASR Service  │        │ Vision Service│          │   (texte direct)  │
 │   (Whisper)   │        │     (ViT)      │          │                    │
 └──────┬───────┘        └──────┬───────┘          └────────┬───────────┘
        │ transcribed_text       │ vision_diagnostic          │ text
        └─────────────┬──────────┘                            │
                       │                                       │
                       └──────────────┬────────────────────────┘
                                      ▼
                            ┌───────────────────┐
                            │   RAG Service       │
                            │ (Embeddings+Chroma) │
                            └─────────┬───────────┘
                                      │ rag_result (rule_text, source)
                                      ▼
                            ┌───────────────────┐
                            │ Generation Service  │  
                            │   (mt0-base LLM)     │
                            └─────────┬───────────┘
                                      ▼
                            ┌───────────────────┐
                            │  Statut proposé      │
                            │
                            └───────────────────┘
```

Chaque service (ASR, Vision, RAG, Génération) est **indépendant**, dans son propre fichier, et charge son modèle **une seule fois en mémoire** (pattern Singleton via `@lru_cache`) pour éviter de saturer la RAM à chaque requête.

---

## 3. Structure du projet

```
support-ticket-ai-api/
├── app/
│   ├── main.py                          # Point d'entrée FastAPI
│   ├── api/
│   │   └── routes/
│   │       └── support_ticket.py        # Endpoint POST /support-ticket
│   ├── services/
│   │   ├── asr_service.py               # Whisper (transcription audio)
│   │   ├── vision_service.py            # ViT (analyse image)
│   │   ├── rag_service.py               # Recherche documentaire (RAG)
│   │   └── generation_service.py        # Génération LLM (bonus)
│   ├── models/
│   │   └── schemas.py                   # Schémas Pydantic (requêtes/réponses)
│   ├── core/
│   │   └── config.py                    # Configuration centralisée
│   └── data/
│       └── knowledge_base/
│           ├── cgv_faq.pdf               # Base de connaissances (CGV/FAQ)
│           └── store/                    # Index vectoriel Chroma (généré, non versionné)
├── tests/                                 # Scripts de test manuels
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4. Prérequis

### Logiciels

- Python 3.10 ou supérieur
- **ffmpeg** installé au niveau système (requis par Whisper pour lire les fichiers audio) :
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```

### Compte / accès

- Aucune clé API payante n'est nécessaire. Tous les modèles utilisés sont gratuits, open-source, et tournent en local (Hugging Face).

---

## 5. Installation

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd support-ticket-ai-api

# 2. Créer et activer un environnement virtuel
python3 -m venv venv
source venv/bin/activate        # Sur Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le serveur
uvicorn app.main:app --reload
```

L'API est alors accessible sur `http://127.0.0.1:8000`, et la documentation interactive Swagger sur `http://127.0.0.1:8000/docs`.

> ⚠️ Au premier lancement, les modèles IA (Whisper, ViT, embeddings, génération) se téléchargent automatiquement depuis Hugging Face (plusieurs Go au total). Cela peut prendre quelques minutes selon la connexion. Les lancements suivants seront rapides.

---

## 6. Configuration

Toute la configuration du projet est centralisée dans `app/core/config.py` :

| Paramètre | Valeur | Rôle |
|---|---|---|
| `WHISPER_MODEL_NAME` | `openai/whisper-small` | Modèle de transcription audio |
| `VIT_MODEL_NAME` | `openai/clip-vit-base-patch32` | Modèle de classification d'image |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Modèle d'embedding pour le RAG (multilingue, adapté au français) |
| `ALLOWED_AUDIO_EXTENSIONS` | `.mp3`, `.wav` | Extensions audio acceptées |
| `ALLOWED_IMAGE_EXTENSIONS` | `.png`, `.jpg`, `.jpeg` | Extensions image acceptées |
| `TEMP_UPLOAD_DIR` | `temp_uploads/` | Dossier temporaire pour les fichiers reçus |

Le modèle de génération (bonus) est défini séparément dans `app/services/generation_service.py` :

| Paramètre | Valeur |
|---|---|
| `GENERATION_MODEL_NAME` | `bigscience/mt0-base` |

---

## 7. Utilisation de l'API

### Endpoint : `POST /support-ticket`

Accepte une requête `multipart/form-data` avec 3 champs :

| Champ | Type | Description |
|---|---|---|
| `audio` | fichier (`.mp3`/`.wav`) | Note vocale du client |
| `image` | fichier (`.png`/`.jpg`) | Photo du produit |
| `text` | texte | Description écrite du client |

### Test via Swagger

1. Lancer le serveur (`uvicorn app.main:app --reload`)
2. Ouvrir `http://127.0.0.1:8000/docs`
3. Déplier `POST /support-ticket` → **Try it out**
4. Remplir les champs souhaités → **Execute**

---

## 8. Détail des services IA

### 8.1 ASR — Transcription audio (`asr_service.py`)

- **Modèle** : `openai/whisper-small` (Hugging Face `transformers`)
- **Chargement** : Singleton via `@lru_cache(maxsize=1)` — le modèle n'est chargé qu'une fois en mémoire, quel que soit le nombre de requêtes.
- **Flux** : le fichier audio est écrit temporairement sur disque (Whisper/ffmpeg a besoin d'un chemin de fichier), transcrit, puis **supprimé systématiquement** (bloc `finally`), même en cas d'erreur.

### 8.2 Vision — Analyse d'image (`vision_service.py`)

- **Modèle** : `openai/clip-vit-base-patch32`, un ViT **généraliste** (entraîné sur ImageNet).

- **Flux** : l'image est traitée **entièrement en mémoire** (`io.BytesIO`), sans écriture sur disque — plus rapide, et aucun nettoyage de fichier n'est nécessaire pour cette partie.

### 8.3 RAG — Recherche documentaire (`rag_service.py`)

Suit la démarche standard du RAG (Retrieval-Augmented Generation) :

1. **Indexation** (une seule fois, mise en cache) :
   - Chargement du PDF de CGV/FAQ (`PyPDFLoader`)
   - Découpage en chunks par regex, un chunk = un article ou une question de FAQ
   - Vectorisation avec `HuggingFaceEmbeddings` (modèle multilingue)
   - Stockage dans une base vectorielle **Chroma**

2. **Retrieval** (à chaque requête) :
   - La question (transcription audio ou texte client) est comparée aux chunks indexés
   - Le chunk le plus proche sémantiquement est retourné avec son score de similarité

### 8.4 Génération — `generation_service.py`

- **Modèle** : `bigscience/mt0-base` (580M paramètres), un modèle "instruction-tuned" multilingue, gratuit et local.
- **Fonctionnement** : à partir du texte de la règle trouvée par le RAG (`rule_text`) et de la réclamation du client, le modèle génère une phrase d'aide pour l'agent support.
- **Prompt anti-hallucination** : le modèle reçoit une consigne explicite lui demandant de répondre "je ne sais pas" si le contexte ne permet pas de répondre, plutôt que d'inventer une réponse.
- **Choix assumé** : un modèle de cette taille reste **nettement moins fiable** qu'un LLM comme GPT-4o-mini (utilisé dans le guide de référence avec une clé API OpenAI). Les réponses générées sont globalement cohérentes mais peuvent occasionnellement manquer de fluidité.

---

## 9. Logique du statut proposé

Implémentée dans `determine_ticket_status()`. Logique volontairement simple et explicable :

| Condition | Statut proposé |
|---|---|
| Aucune règle RAG trouvée | `À vérifier` |
| Règle trouvée, mais diagnostic Vision peu fiable | `À vérifier` |
| Règle trouvée, diagnostic fiable | `Remboursable` |

Cette heuristique est un premier niveau de décision automatique ; elle n'a pas vocation à remplacer un jugement humain final, d'où le statut `À vérifier` en cas de doute.
---

## 10. Gestion des exceptions et robustesse

- **Validation des extensions de fichiers** : tout fichier dont l'extension n'est pas dans `ALLOWED_AUDIO_EXTENSIONS` / `ALLOWED_IMAGE_EXTENSIONS` est rejeté avant tout traitement IA.
- **Nettoyage garanti des fichiers temporaires** : le fichier audio temporaire est supprimé dans un bloc `finally`, qui s'exécute que la requête réussisse ou échoue.
- **Gestion centralisée des erreurs** : les exceptions (extension invalide, fichier corrompu, etc.) sont interceptées et renvoient une réponse HTTP claire (400/422) plutôt qu'une erreur 500 brute.

---


## 11. Gestion de projet (Git Flow / Kanban)

- **Git Flow** : `main` (stable) / `develop` (intégration) / `feature/...` (une branche par fonctionnalité), fusionnées via Pull Request.
- **Kanban** : suivi des tâches sur GitHub Projects — lien : **[À COMPLÉTER]**
- **Vidéo de démonstration** (max 3 min) : **[À COMPLÉTER]**

---

## 12. Glossaire

- **ASR (Automatic Speech Recognition)** : reconnaissance vocale automatique, transcription audio → texte.
- **Embedding** : représentation d'un texte sous forme de vecteur numérique capturant son sens.
- **RAG (Retrieval-Augmented Generation)** : technique consistant à récupérer un contexte pertinent dans des documents avant de répondre à une question.
- **Singleton** : ici, pattern garantissant qu'un modèle IA n'est chargé qu'une seule fois en mémoire, via `@lru_cache`.
