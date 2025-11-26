import os
from textwrap import shorten

from dotenv import load_dotenv
from mistralai import Mistral

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.embeddings import Embeddings


# ----------------------------------------------------
# Wrapper Embeddings Mistral (le même que pour l'étape 3)
# ----------------------------------------------------
class MistralEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "mistral-embed"):
        self.api_key = api_key
        self.model = model
        self.client = Mistral(api_key=api_key)

    def embed_documents(self, texts):
        embeddings = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            resp = self.client.embeddings.create(model=self.model, inputs=batch)
            for item in resp.data:
                embeddings.append(item.embedding)
        return embeddings

    def embed_query(self, text: str):
        resp = self.client.embeddings.create(model=self.model, inputs=[text])
        return resp.data[0].embedding


# ----------------------------------------------------
# Chargement config, vector store, client Mistral
# ----------------------------------------------------
def load_resources():
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("NON OK MISTRAL_API_KEY manquante dans .env")

    embeddings = MistralEmbeddings(api_key=api_key)

    faiss_dir = os.path.join("data", "faiss_openagenda_lc")
    if not os.path.isdir(faiss_dir):
        raise FileNotFoundError(
            f"NON OK Dossier FAISS introuvable : {faiss_dir}. "
            "Assure-toi d'avoir exécuté build_faiss_index.py avant."
        )

    vectorstore = FAISS.load_local(
        folder_path=faiss_dir,
        embeddings=embeddings,
        index_name="index",
        allow_dangerous_deserialization=True,
    )

    client = Mistral(api_key=api_key)

    return vectorstore, client


# ----------------------------------------------------
# Construction du prompt RAG
# ----------------------------------------------------
def build_rag_prompt(user_query: str, docs):
    """
    Construit un prompt pour Mistral à partir :
    - de la question utilisateur
    - des chunks retournés par FAISS (docs)
    """

    context_lines = []
    for doc in docs:
        meta = doc.metadata
        titre = meta.get("title", "")
        ville = meta.get("city", "")
        debut = meta.get("first_begin", "")
        chunk_id = meta.get("chunk_id", "")

        extrait = shorten(doc.page_content.replace("\n", " "), width=400, placeholder="...")

        block = (
            f"- Événement UID {meta.get('uid', '')} | chunk {chunk_id}\n"
            f"  Titre : {titre}\n"
            f"  Ville : {ville} | Début : {debut}\n"
            f"  Description (extrait) : {extrait}\n"
        )
        context_lines.append(block)

    context = "\n".join(context_lines)

    system_prompt = (
        "Tu es un assistant qui recommande des événements culturels en Île-de-France. "
        "Tu dois répondre en français, de manière claire et synthétique. "
        "Tu t'appuies uniquement sur les événements fournis dans le contexte. "
        "Si aucun événement n'est pertinent, tu le dis explicitement."
    )

    user_prompt = f"""
Question de l'utilisateur :
{user_query}

Voici une sélection d'événements pertinents (contexte) :

{context}

Consigne :
- Propose les événements les plus adaptés à la demande.
- Justifie en quelques phrases en t'appuyant sur les informations (titre, ville, date, description).
- Si c'est une recherche de type 'que faire ce week-end à Paris', propose plusieurs options.
- Si tu n'as pas assez d'informations, dis-le clairement.
"""

    return system_prompt, user_prompt.strip()


# ----------------------------------------------------
# Boucle d'interaction simple en ligne de commande
# ----------------------------------------------------
def chat_loop():
    vectorstore, client = load_resources()
    print("OK Chatbot RAG prêt (FAISS + Mistral). Tape 'quit' pour sortir.\n")

    while True:
        user_query = input("Vous: ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"quit", "exit"}:
            print("Bot: Au revoir !")
            break

        # 1) Recherche sémantique dans FAISS
        docs = vectorstore.similarity_search(user_query, k=5)

        # 2) Construction du prompt RAG
        system_prompt, user_prompt = build_rag_prompt(user_query, docs)

        # 3) Appel au modèle de chat Mistral
        try:
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"(Erreur lors de l'appel Mistral : {e})"

        print("\nBot:", answer, "\n")


if __name__ == "__main__":
    chat_loop()
