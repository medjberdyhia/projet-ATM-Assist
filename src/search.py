"""Recherche dans la base documentaire : vectorielle (le sens), lexicale BM25 (les mots
exacts) ou hybride (les deux fusionnées). Réutilisée par l'agent, le RAG et le serveur MCP."""
import math
import re
import sys
import unicodedata
from collections import defaultdict

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CHROMA_PATH, MODELE_EMBEDDING, NOM_COLLECTION, RECHERCHE_HYBRIDE

# Mots sans intérêt pour la recherche par mots exacts (écrits sans accents)
MOTS_VIDES = {
    "le", "la", "les", "l", "d", "de", "du", "des", "un", "une", "et", "en", "a", "au", "aux",
    "que", "qui", "quoi", "qu", "est", "sont", "pour", "par", "sur", "dans", "ce", "ces", "se",
    "sa", "son", "ses", "ou", "combien", "quel", "quelle", "quels", "quelles", "comment",
    "pourquoi", "il", "elle", "ils", "elles", "ne", "pas", "plus", "avec", "sans", "s", "c", "n", "y",
}

_etat = None  # (modèle, textes, métadonnées, embeddings, mots de chaque passage)


def tokeniser(texte: str) -> list[str]:
    """Découpe en mots : minuscules, sans accents, « 1 030 » -> « 1030 », pluriels simplifiés."""
    texte = re.sub(r"(?<=\d)[\s\u202f\xa0](?=\d{3}\b)", "", texte.lower())
    texte = "".join(c for c in unicodedata.normalize("NFKD", texte) if not unicodedata.combining(c))
    mots = [m for m in re.findall(r"[a-z0-9]+", texte) if m not in MOTS_VIDES]
    return [m[:-1] if len(m) > 3 and m.endswith(("s", "x")) else m for m in mots]


def scores_bm25(corpus: list[list[str]], requete: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Score BM25 de chaque passage pour la requête : un mot rare qui apparaît dans le
    passage compte beaucoup ; un mot fréquent partout compte peu ; les passages très longs
    sont légèrement pénalisés."""
    n = len(corpus)
    scores = np.zeros(n)
    if n == 0:
        return scores
    longueur_moyenne = sum(len(d) for d in corpus) / n or 1
    for terme in set(requete):
        contenant = [i for i, d in enumerate(corpus) if terme in d]
        if not contenant:
            continue
        idf = math.log((n - len(contenant) + 0.5) / (len(contenant) + 0.5) + 1)
        for i in contenant:
            tf = corpus[i].count(terme)
            scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(corpus[i]) / longueur_moyenne))
    return scores


def fusion_rrf(*classements: list[int], c: int = 60) -> list[int]:
    """Reciprocal Rank Fusion : un passage bien classé dans l'un OU l'autre classement
    remonte. Chaque classement est une liste d'indices, du meilleur au moins bon."""
    total = defaultdict(float)
    for classement in classements:
        for rang, idx in enumerate(classement, 1):
            total[idx] += 1 / (c + rang)
    return sorted(total, key=total.get, reverse=True)


def _charger():
    """Charge le modèle et la base une seule fois (c'est lent), puis les réutilise."""
    global _etat
    if _etat is None:
        modele = SentenceTransformer(MODELE_EMBEDDING)
        collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(NOM_COLLECTION)
        donnees = collection.get(include=["documents", "metadatas", "embeddings"])
        embeddings = np.array(donnees["embeddings"], dtype=float)
        mots = [tokeniser(t) for t in donnees["documents"]]
        _etat = (modele, donnees["documents"], donnees["metadatas"], embeddings, mots)
    return _etat


def rechercher(question: str, k: int = 3, hybride: bool | None = None) -> list[dict]:
    """Retourne les k meilleurs passages. `score` = similarité cosinus (0 à 1), même en mode
    hybride : c'est elle qui sert au seuil de pertinence."""
    if hybride is None:
        hybride = RECHERCHE_HYBRIDE
    modele, textes, metas, embeddings, mots = _charger()
    vecteur_question = modele.encode([question], normalize_embeddings=True)[0]
    cosinus = embeddings @ vecteur_question  # embeddings normalisés : produit scalaire = cosinus
    classement_vecteur = [int(i) for i in np.argsort(-cosinus)]
    if hybride:
        lexical = scores_bm25(mots, tokeniser(question))
        classement_lexical = [int(i) for i in np.argsort(-lexical) if lexical[i] > 0][:50]
        ordre = fusion_rrf(classement_vecteur[:50], classement_lexical)
    else:
        ordre = classement_vecteur
    return [
        {"texte": textes[i], "source": metas[i]["source"], "fichier": metas[i]["fichier"],
         "score": round(float(cosinus[i]), 3)}
        for i in ordre[:k]
    ]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Qu'est-ce qu'un transpondeur ?"
    print(f"Question : {question}\n")
    for i, p in enumerate(rechercher(question), 1):
        print(f"[{i}] score {p['score']} | {p['fichier']}")
        print(f"    {p['texte'][:300]}...\n")