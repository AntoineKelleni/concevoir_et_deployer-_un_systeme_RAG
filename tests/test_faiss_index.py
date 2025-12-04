# tests/test_faiss_index.py

import os

import numpy as np
import pandas as pd

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.faiss import FAISS

from build_faiss_index import (
    PREPROCESSED_CSV_PATH,
    EMBEDDINGS_PATH,
)


class DummyEmbeddings(Embeddings):
    """
    Implémentation minimale d'Embeddings pour les tests.
    On ne fait pas appel à une API externe.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def embed_documents(self, texts):
        # Retourne un vecteur constant par document
        return [ [0.0] * self.dim for _ in texts ]

    def embed_query(self, text: str):
        # Retourne un vecteur constant pour la requête
        return [0.0] * self.dim


def test_faiss_index_construction_and_search():
    """
    Vérifie que l'on peut construire un index FAISS à partir
    du CSV prétraité et des embeddings, puis faire une recherche.
    """
    assert os.path.exists(PREPROCESSED_CSV_PATH), f"{PREPROCESSED_CSV_PATH} manquant"
    assert os.path.exists(EMBEDDINGS_PATH), f"{EMBEDDINGS_PATH} manquant"

    df = pd.read_csv(PREPROCESSED_CSV_PATH)
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    assert len(df) == embeddings.shape[0], "Incohérence CSV / embeddings"

    texts = df["chunk_text"].tolist()
    text_embedding_pairs = list(zip(texts, embeddings.tolist()))

    metadatas = df.drop(columns=["chunk_text"], errors="ignore").to_dict(orient="records")

    dim = embeddings.shape[1]
    dummy_embeddings = DummyEmbeddings(dim=dim)

    # Construction de l'index FAISS (sans appel Mistral)
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=dummy_embeddings,
        metadatas=metadatas,
    )

    # On doit avoir autant de vecteurs que de lignes
    assert vectorstore.index.ntotal == len(df)

    # On teste une recherche de similarité (la requête donne un vecteur constant)
    docs = vectorstore.similarity_search("test", k=3)
    assert len(docs) == 3
    # Chaque doc doit avoir des métadonnées de base
    for doc in docs:
        meta = doc.metadata
        assert "title" in meta
        assert "city" in meta
