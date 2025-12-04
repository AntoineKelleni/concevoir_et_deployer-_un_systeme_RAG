# tests/test_preprocessing.py

import os

import numpy as np
import pandas as pd

from openagenda_preprocessing import (
    chunk_text,
    normalize_events,
    PREPROCESSED_CSV_PATH,
    EMBEDDINGS_PATH,
)


def test_chunk_text_basic():
    """
    Vérifie que chunk_text découpe bien un texte long
    et qu'il y a recouvrement entre les chunks.
    """
    text = "A" * 1500  # 1500 caractères
    chunks = chunk_text(text, max_chars=1000, overlap=200)

    # On doit obtenir au moins 2 chunks
    assert len(chunks) >= 2

    # Chaque chunk doit faire au plus max_chars caractères
    assert all(len(c) <= 1000 for c in chunks)

    # Le début du deuxième chunk doit chevaucher la fin du premier
    first, second = chunks[0], chunks[1]
    # On s'attend à ce que les 200 derniers chars du premier
    # correspondent aux 200 premiers du second
    assert first[-200:] == second[:200]


def test_normalize_events_structure():
    """
    Vérifie que normalize_events produit bien une ligne structurée
    avec toutes les clés attendues et un text_for_embedding non vide.
    """
    fake_events = [
        {
            "uid": "evt1",
            "slug": "slug-evt1",
            "title": {"fr": "Titre de test"},
            "description": {"fr": "Description courte"},
            "longDescription": {"fr": "Description longue"},
            "conditions": {"fr": "Entrée libre"},
            "keywords": {"fr": ["concert", "test"]},
            "location": {
                "city": "Paris",
                "name": {"fr": "Lieu de test"},
            },
            "timings": [
                {"begin": "2026-06-01T20:00:00+02:00", "end": "2026-06-01T22:00:00+02:00"}
            ],
        }
    ]

    rows = normalize_events(fake_events)
    assert len(rows) == 1

    row = rows[0]
    expected_keys = {
        "uid",
        "slug",
        "title",
        "description_short",
        "description_long",
        "conditions",
        "keywords",
        "city",
        "venue_name",
        "first_begin",
        "first_end",
        "text_for_embedding",
    }

    assert expected_keys.issubset(row.keys())
    assert row["uid"] == "evt1"
    assert row["city"] == "Paris"
    assert isinstance(row["text_for_embedding"], str)
    assert row["text_for_embedding"].strip() != ""


def test_preprocessed_csv_and_embeddings_alignment():
    """
    Vérifie que le CSV prétraité et le fichier .npy des embeddings
    sont cohérents (même nombre de lignes / vecteurs).
    """
    assert os.path.exists(PREPROCESSED_CSV_PATH), f"{PREPROCESSED_CSV_PATH} manquant"
    assert os.path.exists(EMBEDDINGS_PATH), f"{EMBEDDINGS_PATH} manquant"

    df = pd.read_csv(PREPROCESSED_CSV_PATH)
    emb = np.load(EMBEDDINGS_PATH)

    # Même nombre de lignes
    assert len(df) == emb.shape[0]

    # Colonnes minimales attendues
    for col in ["uid", "title", "city", "chunk_id", "chunk_text"]:
        assert col in df.columns, f"Colonne manquante dans le CSV : {col}"

    # Embeddings de dimension > 0
    assert emb.shape[1] > 0
