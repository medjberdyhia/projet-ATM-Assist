"""Recherche sémantique dans la base Chroma. Sera réutilisée par l'agent et le serveur MCP."""
import sys

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_PATH, MODELE_EMBEDDING, NOM_COLLECTION

_modele = None
_collection = None


def _charger():
    """Charge le modèle et la base une seule fois (c'est lent), puis les réutilise."""
    global _modele, _collection
    if _modele is None:
        _modele = SentenceTransformer(MODELE_EMBEDDING)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(NOM_COLLECTION)
    return _modele, _collection


def rechercher(question: str, k: int = 3) -> list[dict]:
    """Retourne les k passages les plus proches du sens de la question."""
    modele, collection = _charger()
    embedding = modele.encode([question], normalize_embeddings=True).tolist()
    resultat = collection.query(query_embeddings=embedding, n_results=k)
    passages = []
    for texte, meta, distance in zip(
        resultat["documents"][0], resultat["metadatas"][0], resultat["distances"][0]
    ):
        passages.append({
            "texte": texte,
            "source": meta["source"],
            "fichier": meta["fichier"],
            "score": round(1 - distance, 3),  # 1 = très proche, 0 = sans rapport
        })
    return passages


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Qu'est-ce qu'un transpondeur ?"
    print(f"Question : {question}\n")
    for i, p in enumerate(rechercher(question), 1):
        print(f"[{i}] score {p['score']} | {p['fichier']}")
        print(f"    {p['texte'][:300]}...\n")