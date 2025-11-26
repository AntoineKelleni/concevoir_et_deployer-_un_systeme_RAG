import os
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral


# ----------------------------------------------------
# Chargement des variables d'environnement
# ----------------------------------------------------
load_dotenv()

print("OK Chargement .env")

OPENAGENDA_API_KEY = os.getenv("OPENAGENDA_API_KEY")
OPENAGENDA_AGENDA_UID = os.getenv("OPENAGENDA_AGENDA_UID")

# Filtre 1 ville (optionnel, pour API et/ou DataFrame)
OPENAGENDA_CITY = os.getenv("OPENAGENDA_CITY", "").strip()

# Filtre multi-villes (optionnel, appliqué dans le DataFrame)
OPENAGENDA_CITIES = os.getenv("OPENAGENDA_CITIES", "").strip()
if OPENAGENDA_CITIES:
    CITY_LIST = [c.strip() for c in OPENAGENDA_CITIES.split(",") if c.strip()]
else:
    CITY_LIST = []

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not OPENAGENDA_API_KEY:
    raise ValueError("NON OK OPENAGENDA_API_KEY manquante dans .env")
if not OPENAGENDA_AGENDA_UID:
    raise ValueError("NON OK OPENAGENDA_AGENDA_UID manquante dans .env")
if not MISTRAL_API_KEY:
    raise ValueError("NON OK MISTRAL_API_KEY manquante dans .env")

print("OK Variables d'environnement chargées")


# ----------------------------------------------------
# Chemins et constantes
# ----------------------------------------------------
OPENAGENDA_BASE_URL = "https://api.openagenda.com/v2/agendas"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

FULL_CSV_PATH = os.path.join(DATA_DIR, "openagenda_events_full_1an.csv")
PREPROCESSED_CSV_PATH = os.path.join(DATA_DIR, "openagenda_events_preprocessed.csv")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "openagenda_events_embeddings.npy")

# Limites de sécurité pour l'API Mistral
MAX_EVENTS = 80    # on réduit aussi un peu les events avant chunking
MAX_CHUNKS = 30    # très petit, mais quasi sûr de passer


# ----------------------------------------------------
# Fonctions utilitaires
# ----------------------------------------------------
def to_iso_z(dt: datetime) -> str:
    """Convertit un datetime en ISO 8601 au format attendu par OpenAgenda."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def build_query_params(city: str, start_date: str, after):
    """Construit les paramètres d'appel pour OpenAgenda."""
    params = [
        ("detailed", 1),
        ("monolingual", "fr"),
        ("size", 100),
        ("timings[gte]", start_date),
        ("relative[]", "passed"),
        ("relative[]", "current"),
        ("relative[]", "upcoming"),
        ("state", 2),
    ]

    if city:
        params.append(("city[]", city))

    if after:
        for value in after:
            params.append(("after[]", value))

    return params


# ----------------------------------------------------
# Récupération de tous les événements via l'API
# ----------------------------------------------------
def fetch_events_for_agenda(agenda_uid: str, city: str):
    """
    Récupère tous les événements d'un agenda sur 1 an d'historique
    + événements en cours et à venir.
    """
    now = datetime.now(timezone.utc)
    one_year_ago = now - timedelta(days=365)

    timings_gte = to_iso_z(one_year_ago)

    print(f"OK Fenêtre temporelle : >= {timings_gte}")

    all_events = []
    after = None
    page = 1

    headers = {"key": OPENAGENDA_API_KEY}

    while True:
        params = build_query_params(city, timings_gte, after)
        url = f"{OPENAGENDA_BASE_URL}/{agenda_uid}/events"

        print(f"OK Appel API OpenAgenda page {page}...")
        resp = requests.get(url, headers=headers, params=params, timeout=30)

        if resp.status_code != 200:
            print("NON OK Erreur API OpenAgenda")
            print("Code :", resp.status_code)
            print("Réponse :", resp.text)
            raise RuntimeError("Erreur API OpenAgenda")

        data = resp.json()
        events = data.get("events", [])
        after = data.get("after")

        print(f"OK Page {page}: {len(events)} événements récupérés")

        all_events.extend(events)

        if not after:
            print("OK Fin pagination")
            break

        page += 1

    print(f"OK Total {len(all_events)} événements récupérés")
    return all_events


# ----------------------------------------------------
# Normalisation des événements
# ----------------------------------------------------
def extract_text(field, lang="fr"):
    if isinstance(field, dict):
        return field.get(lang) or ""
    if isinstance(field, str):
        return field
    return ""


def normalize_events(events):
    rows = []

    for e in events:
        uid = e.get("uid")
        slug = e.get("slug")

        title = extract_text(e.get("title"))
        desc_short = extract_text(e.get("description"))
        desc_long = extract_text(e.get("longDescription"))
        conditions = extract_text(e.get("conditions"))

        keywords = ""
        kws = e.get("keywords")
        if isinstance(kws, dict):
            fr_list = kws.get("fr") or []
            if isinstance(fr_list, list):
                keywords = ", ".join(fr_list)

        location = e.get("location") or {}
        city = location.get("city") or ""
        venue_name = extract_text(location.get("name"))

        timings = e.get("timings") or []
        first_begin = timings[0].get("begin") if timings else None
        first_end = timings[0].get("end") if timings else None

        text_for_embedding = "\n\n".join(
            x
            for x in [
                title,
                desc_short,
                desc_long,
                f"Mots clés : {keywords}" if keywords else "",
                f"Conditions : {conditions}" if conditions else "",
            ]
            if x
        )

        rows.append(
            {
                "uid": uid,
                "slug": slug,
                "title": title,
                "description_short": desc_short,
                "description_long": desc_long,
                "conditions": conditions,
                "keywords": keywords,
                "city": city,
                "venue_name": venue_name,
                "first_begin": first_begin,
                "first_end": first_end,
                "text_for_embedding": text_for_embedding,
            }
        )

    print(f"OK Normalisation terminée : {len(rows)} lignes")
    return rows


# ----------------------------------------------------
# Chunking des textes
# ----------------------------------------------------
def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200):
    """
    Découpe un texte en chunks de taille max_chars avec recouvrement overlap.
    Approche simple basée sur la longueur en caractères, suffisante pour ce POC.
    """
    if not isinstance(text, str):
        return []

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        # recouvrement pour ne pas couper trop brutalement
        start += max_chars - overlap

    return chunks


# ----------------------------------------------------
# Calcul des embeddings Mistral
# ----------------------------------------------------
def compute_embeddings(texts):
    client = Mistral(api_key=MISTRAL_API_KEY)
    model = "mistral-embed"

    embeddings = []
    batch_size = 64

    print(f"OK Calcul embeddings : {len(texts)} textes")

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        print(f"OK Batch {i} -> {i + len(batch)}")

        try:
            resp = client.embeddings.create(model=model, inputs=batch)
        except Exception as e:
            print("NON OK Erreur API Mistral lors du calcul des embeddings")
            print("Détail erreur :", e)
            print(
                "NON OK Arrêt du calcul des embeddings. "
                "Réduisez MAX_EVENTS/MAX_CHUNKS ou réessayez plus tard si le quota est épuisé."
            )
            raise

        for item in resp.data:
            embeddings.append(item.embedding)

    print("OK Embeddings générés")
    return np.array(embeddings, dtype="float32")


# ----------------------------------------------------
# Script principal
# ----------------------------------------------------
def main():
    print("OK Début préprocessing OpenAgenda")

    # 1) Essayer de charger depuis le cache complet 1 an
    if os.path.exists(FULL_CSV_PATH):
        print(f"OK Dataset complet déjà présent : {FULL_CSV_PATH}")
        df_full = pd.read_csv(FULL_CSV_PATH)
        print(f"OK Chargé depuis le cache : {len(df_full)} événements")
    else:
        print("OK Aucun cache détecté, appel API OpenAgenda...")

        events = fetch_events_for_agenda(
            agenda_uid=OPENAGENDA_AGENDA_UID,
            city=OPENAGENDA_CITY  # vide = toutes villes
        )

        if not events:
            print("NON OK Aucun événement récupéré")
            return

        rows = normalize_events(events)
        df_full = pd.DataFrame(rows)

        df_full.to_csv(FULL_CSV_PATH, index=False)
        print(f"OK Sauvegarde brute : {len(df_full)} événements dans {FULL_CSV_PATH}")

    # À partir d'ici, on travaille uniquement en local sur df_full
    df = df_full.copy()

    # Filtre 1 ville (optionnel)
    if OPENAGENDA_CITY:
        df = df[df["city"] == OPENAGENDA_CITY].reset_index(drop=True)
        print(f"OK Filtre (1 ville) : {OPENAGENDA_CITY} -> {len(df)} événements")

    # Filtre multi-villes (optionnel)
    if CITY_LIST:
        df = df[df["city"].isin(CITY_LIST)].reset_index(drop=True)
        print(f"OK Filtre (multi-villes) : {CITY_LIST} -> {len(df)} événements")

    # Suppression textes vides
    df = df[df["text_for_embedding"].str.strip() != ""].reset_index(drop=True)
    print(f"OK DataFrame après nettoyage : {len(df)} lignes")

    # Conversion date de début en datetime pour trier (avec utc=True pour éviter le FutureWarning)
    df["first_begin_dt"] = pd.to_datetime(
        df["first_begin"], errors="coerce", utc=True
    )

    # Tri du plus récent au plus ancien
    df = df.sort_values("first_begin_dt", ascending=False).reset_index(drop=True)

    # Vérification des colonnes importantes pour la suite (FAISS)
    required_cols = [
        "uid",
        "title",
        "city",
        "venue_name",
        "first_begin",
        "first_end",
        "text_for_embedding",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"NON OK Colonne manquante dans le DataFrame : {col}")

    # Limiter le nombre d'événements avant chunking (sécurité)
    if len(df) > MAX_EVENTS:
        print(f"OK Limitation événements : {MAX_EVENTS} sur {len(df)} (les plus récents)")
        df = df.head(MAX_EVENTS).reset_index(drop=True)

    print(f"OK DataFrame retenu pour chunking : {len(df)} événements")

    # ------------------------------------------------
    # Création des chunks par événement
    # ------------------------------------------------
    rows_chunks = []
    for idx, row in df.iterrows():
        chunks = chunk_text(row["text_for_embedding"], max_chars=1200, overlap=200)
        if not chunks:
            continue

        for chunk_id, chunk in enumerate(chunks):
            base = row.to_dict()
            # On enlève la colonne technique utilisée uniquement pour le tri
            base.pop("first_begin_dt", None)
            base["chunk_id"] = chunk_id
            base["chunk_text"] = chunk
            rows_chunks.append(base)

    df_chunks = pd.DataFrame(rows_chunks)
    print(f"OK Chunking : {len(df)} événements -> {len(df_chunks)} chunks")

    # ------------------------------------------------
    # Limitation pour respecter le quota Mistral
    # ------------------------------------------------
    if len(df_chunks) > MAX_CHUNKS:
        print(f"OK Limitation embeddings (chunks) : {MAX_CHUNKS} sur {len(df_chunks)}")
        df_chunks = df_chunks.head(MAX_CHUNKS).reset_index(drop=True)

    print(f"OK DataFrame final pour embeddings : {len(df_chunks)} lignes")

    # ------------------------------------------------
    # Calcul des embeddings sur les chunks
    # ------------------------------------------------
    embeddings = compute_embeddings(df_chunks["chunk_text"].tolist())

    # Sauvegardes prétraitées pour l'étape 3 (FAISS)
    df_chunks.to_csv(PREPROCESSED_CSV_PATH, index=False)
    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"OK Données sauvegardées dans {PREPROCESSED_CSV_PATH}")
    print(f"OK Embeddings sauvegardés dans {EMBEDDINGS_PATH}")
    print("OK Préprocessing terminé")


if __name__ == "__main__":
    main()
