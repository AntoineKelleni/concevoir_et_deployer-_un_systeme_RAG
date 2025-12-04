![Python](https://img.shields.io/badge/Python-3.11-blue)
![FAISS](https://img.shields.io/badge/FAISS-Indexing-green)
![Mistral](https://img.shields.io/badge/Mistral-API-orange)
![LangChain](https://img.shields.io/badge/LangChain-RAG-blueviolet)
![OpenAgenda](https://img.shields.io/badge/OpenAgenda-API-success)
![Status](https://img.shields.io/badge/Build-Passing-brightgreen)

<p align="center"> <img src="logo_OCR.jpg" alt="Logo Academy" width="200"> </p>

# Système RAG – ChatBOT : Recommandation d’événements culturels (OpenAgenda + Mistral + FAISS)

## Introduction
Ce projet met en place un pipeline complet permettant :
1. La récupération des événements OpenAgenda

2. Le nettoyage, la normalisation et le découpage (chunking)

3. La vectorisation des textes via les embeddings Mistral

4. L’indexation des vecteurs dans FAISS

5. L’intégration dans un chatbot RAG capable de faire des recommandations à partir d’une requête utilisateur




## Architecture globale du pipeline RAG

```
Étape 1 : Vérification de l’environnement
Étape 2 : Préprocessing OpenAgenda (nettoyage, chunking, embeddings)
Étape 3 : Construction de l’index FAISS
Étape 4 : Chatbot RAG (FAISS + Mistral)
```
Le pipeline est structuré en 4 blocs principaux :
```
Préprocessing OpenAgenda → Embeddings Mistral → Indexation FAISS → Chatbot RAG

```

## Technologies utilisées

```
Python – Scripts et pipeline
OpenAgenda API – Source des événements
Mistral – Embeddings et génération
LangChain – Intégration FAISS + RAG
FAISS – Recherche vectorielle
Pandas – Nettoyage des données
Pytest – Tests unitaires
```

# 1️⃣ Installation et configuration
Création de l’environnement

``` 
python -m venv .venv
..venv\Scripts\Activate.ps1 # Windows PowerShell
source .venv/bin/activate # macOS / Linux
```

Installation des dépendances

```
pip install --upgrade pip
pip install -r requirements.txt
```

Fichier .env

```
OPENAGENDA_API_KEY=xxxxxxxxxVOTRE_CLExxxxxxxxxx
OPENAGENDA_AGENDA_UID=56500817

OPENAGENDA_CITY=
OPENAGENDA_CITIES=Paris, Versailles, Meudon

MISTRAL_API_KEY=VOTRE_CLE_MISTRAL
```

Vérification de l’environnement

```
python test_install.py
```

# 2️⃣ Préprocessing OpenAgenda

Script :

```
python openagenda_preprocessing.py
```

Ce script réalise :

```

Appel API OpenAgenda (ou chargement du cache)

Normalisation des champs

Sauvegarde brute : openagenda_events_full_1an.csv

Filtrage des villes optionnel

Tri du plus récent au plus ancien

Chunking du texte des événements

Limitations (MAX_EVENTS / MAX_CHUNKS)

Calcul des embeddings Mistral

Sauvegardes finales :

openagenda_events_preprocessed.csv

openagenda_events_embeddings.npy
```

Exemple de structure des chunks :

```
uid
title
city
chunk_id
chunk_text
```

# 3️⃣ Indexation FAISS (LangChain + embeddings pré-calculés)

Construction de l’index :

```
python build_faiss_index.py
```

Actions réalisées :

```

Chargement CSV prétraité + embeddings .npy

Construction d’un vector store FAISS

Stockage des métadonnées (titre, ville, date, chunk_id…)

Sauvegarde :
data/faiss_openagenda_lc/

Test de recherche :
"concert à Paris"
```

Exemple de résultat de test :

```
Titre : Julien Clerc - Une vie
Ville : Versailles
Date : 2026-10-10
Chunk : 0
```

# 4️⃣ Chatbot RAG (FAISS + Mistral)

Lancement :

```
python chatbot_rag.py
```

Fonctionnement :

```

Encodage de la question utilisateur (embeddings)

Recherche sémantique dans FAISS

Création d’un contexte avec les meilleurs événements

Appel du modèle Mistral pour générer la réponse finale
```

Exemples d’utilisation :

```
Vous : Que faire ce week-end à Versailles ?
Vous : Propose-moi un concert en Île-de-France.
Vous : Je cherche une sortie culturelle en soirée.
```

# 5️⃣ Tests unitaires

Lancer :

```
pytest
```

Contenu des tests :

```
test_preprocessing.py :

Test du chunking

Test du format normalisé

Test alignement CSV / embeddings

test_faiss_index.py :

Test de la création d’un index FAISS

Test de la recherche vectorielle
```

Exemple :

```
3 passed in 12.11s
```

Structure du repository

```
project/
│── data/
│ ├── openagenda_events_full_1an.csv
│ ├── openagenda_events_preprocessed.csv
│ ├── openagenda_events_embeddings.npy
│ ├── faiss_openagenda_lc/
│
│── tests/
│ ├── test_preprocessing.py
│ └── test_faiss_index.py
│
│── openagenda_preprocessing.py
│── build_faiss_index.py
│── chatbot_rag.py
│── test_install.py
│── requirements.txt
│── .env
│── README.md
```
# Résultats attendus

```
Recherche sémantique fonctionnelle

Chatbot capable de recommander des événements

Embeddings cohérents

Index FAISS performant

Tests unitaires validés
```

# Exemples de réponses générées :

```
→ Propositions d’événements pertinents
→ Description résumée
→ Lieu + Date
→ Justification basée sur le contexte
```

# Conclusion

Ce POC démontre une architecture RAG complète :

```
Ingestion API

Préprocessing avancé

Vectorisation Mistral

Indexation FAISS

Chatbot Mistral guidé par la recherche

Tests unitaires garantissant la fiabilité
```

Ce système peut facilement être étendu vers une application web, une API REST ou un service de recommandation à plus grande échelle.