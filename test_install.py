import importlib
import os
from dotenv import load_dotenv

load_dotenv()

print("OK Chargement .env")

# Vérification des bibliothèques nécessaires
required_packages = [
    "langchain",
    "langchain_core",
    "langchain_community",
    "faiss",
    "numpy",
    "pandas",
    "requests",
    "dotenv",
    "mistralai"
]

print("OK Début vérification des imports...")

missing = []

for pkg in required_packages:
    try:
        importlib.import_module(pkg)
        print(f"OK Import {pkg}")
    except ImportError:
        print(f"NON OK Import {pkg}")
        missing.append(pkg)

if missing:
    print("\nNON OK Les bibliothèques suivantes sont manquantes :")
    for m in missing:
        print(" -", m)
    print("\nInstalle-les avec :")
    print("pip install " + " ".join(missing))
    raise SystemExit()

print("OK Tous les imports sont corrects")


# Vérification de la clé Mistral
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("NON OK MISTRAL_API_KEY manquante dans .env")

print("OK Clé Mistral chargée")

from mistralai import Mistral
client = Mistral(api_key=api_key)


# Mini test de requête API Mistral
try:
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "Test"}]
    )
    print("OK Appel API Mistral opérationnel")
except Exception as e:
    print("NON OK Appel API Mistral échoué")
    print("Erreur :", e)

print("OK environnement prêt !")
