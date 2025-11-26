import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.faiss import FAISS


# ----------------------------------------------------
# Embeddings Mistral pour les requêtes (LangChain)
# ----------------------------------------------------
class MistralEmbeddings(Embeddings):
    """
    Wrapper LangChain pour utiliser l'API d'embeddings de Mistral.
    On l'utilise surtout pour embedder les requêtes utilisateur.
    """

    def __init__(self, api_key: str, model: str = "mistral-embed"):
        self.api_key = api_key
        self.model = model
        self.client = Mistral(api_key=api_key)

    def embed_documents(self, texts):
        # Pas utilisé dans notre pipeline (on a déjà pré-calculé les embeddings des docs),
        # mais on l'implémente pour rester compatible avec LangChain.
        embeddings = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            resp = self.client.embeddings.create(model=self.model, inputs=batch)
            for item in resp.data:
                embeddings.append(item.embedding)
        return embeddings

    def embed_query(self, text: str):
        # Utilisé par similarity_search(query=...) côté LangChain
        resp = self.client.embeddings.create(model=self.model, inputs=[text])
        return resp.data[0].embedding


# ----------------------------------------------------
# Chemins et constantes
# ----------------------------------------------------
DATA_DIR = "data"
PREPROCESSED_CSV_PATH = os.path.join(DATA_DIR, "openagenda_events_preprocessed.csv")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "openagenda_events_embeddings.npy")
LC_FAISS_DIR = os.path.join(DATA_DIR, "faiss_openagenda_lc")  # dossier de persistance LangChain


def main():
    # 1) Chargement config / .env
    load_dotenv()
    print("OK Chargement .env")

    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("NON OK MISTRAL_API_KEY manquante dans .env")

    # 2) Chargement des données prétraitées
    if not os.path.exists(PREPROCESSED_CSV_PATH):
        raise FileNotFoundError(f"NON OK Fichier CSV introuvable : {PREPROCESSED_CSV_PATH}")
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"NON OK Fichier embeddings introuvable : {EMBEDDINGS_PATH}")

    df = pd.read_csv(PREPROCESSED_CSV_PATH)
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    print(f"OK DataFrame prétraité : {len(df)} lignes (chunks)")
    print(f"OK Embeddings : shape = {embeddings.shape}")

    if embeddings.shape[0] != len(df):
        raise RuntimeError(
            f"NON OK Incohérence : {embeddings.shape[0]} embeddings pour {len(df)} lignes"
        )

    # 3) Préparation des paires (texte, vecteur) pour FAISS.from_embeddings
    texts = df["chunk_text"].tolist()
    text_embedding_pairs = list(zip(texts, embeddings.tolist()))

    # Métadonnées = toutes les colonnes (on enlève juste chunk_text pour alléger)
    metadatas = df.drop(columns=["chunk_text"], errors="ignore").to_dict(orient="records")

    # 4) Initialisation du wrapper d'embeddings pour LangChain (pour les requêtes)
    mistral_embeddings = MistralEmbeddings(api_key=mistral_api_key)

    # 5) Construction du vector store FAISS (LangChain) à partir d'embeddings pré-calculés
    print("OK Construction du vector store FAISS (LangChain) à partir des embeddings pré-calculés...")
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=mistral_embeddings,
        metadatas=metadatas,
    )

    print(f"OK Vector store FAISS créé avec {vectorstore.index.ntotal} vecteurs")

    # 6) Persistance sur disque (LangChain gère l'index FAISS + docstore)
    os.makedirs(LC_FAISS_DIR, exist_ok=True)
    vectorstore.save_local(LC_FAISS_DIR)
    print(f"OK Vector store sauvegardé dans {LC_FAISS_DIR}")

    # 7) Test de recherche (sanity check)
    print("OK Test de recherche : 'concert à Paris' ...")
    docs = vectorstore.similarity_search("concert à Paris", k=3)

    for i, doc in enumerate(docs):
        meta = doc.metadata
        print(f"\n--- Résultat {i+1} ---")
        print("Titre :", meta.get("title", ""))
        print("Ville :", meta.get("city", ""))
        print("Début :", meta.get("first_begin", ""))
        print("Chunk id :", meta.get("chunk_id", ""))
        print("Extrait chunk :", doc.page_content[:200].replace("\n", " "))

    print("\nOK Étape 3 terminée : index FAISS + métadonnées prêts pour l'étape 4.")


if __name__ == "__main__":
    main()
