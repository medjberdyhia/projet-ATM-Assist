"""Ingestion : lit les documents de data/, les découpe, calcule les embeddings
et les stocke dans une base vectorielle Chroma (dossier chroma_db/)."""
import re

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_PATH, DOSSIER_DATA, MODELE_EMBEDDING, NOM_COLLECTION

# Sections de fin d'article Wikipédia sans intérêt pour répondre à des questions
FIN_ARTICLE = re.compile(
    r"^=+ *(Voir aussi|Notes et références|Liens externes|Bibliographie|Articles connexes) *=+ *$",
    re.MULTILINE,
)


def lire_document(chemin):
    """Retourne (url_source, texte_utile) pour un fichier de data/."""
    premiere_ligne, _, texte = chemin.read_text(encoding="utf-8").partition("\n")
    source = premiere_ligne.replace("Source :", "").strip()
    fin = FIN_ARTICLE.search(texte)
    if fin:
        texte = texte[: fin.start()]
    return source, texte.strip()


def decouper(texte, taille=500, chevauchement=50):
    """Découpe un texte en passages d'environ `taille` caractères.
    Chaque passage reprend `chevauchement` caractères du précédent pour ne pas
    couper une idée en deux. On coupe de préférence à un espace ou un retour à la ligne."""
    morceaux = []
    debut = 0
    while debut < len(texte):
        fin = min(debut + taille, len(texte))
        if fin < len(texte):
            coupe = max(texte.rfind(" ", debut + taille // 2, fin),
                        texte.rfind("\n", debut + taille // 2, fin))
            if coupe != -1:
                fin = coupe
        morceau = texte[debut:fin].strip()
        if len(morceau) > 80:  # on ignore les passages trop courts (titres isolés...)
            morceaux.append(morceau)
        if fin >= len(texte):
            break
        debut = max(fin - chevauchement, debut + 1)
    return morceaux


def main():
    fichiers = sorted(DOSSIER_DATA.glob("*.txt"))
    if not fichiers:
        print("Aucun fichier .txt dans data/. Lance d'abord download_corpus.py")
        return

    print("Chargement du modèle d'embeddings (téléchargement à la 1re utilisation)...")
    modele = SentenceTransformer(MODELE_EMBEDDING)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(NOM_COLLECTION)  # on repart de zéro à chaque ingestion
    except Exception:
        pass
    collection = client.create_collection(NOM_COLLECTION, metadata={"hnsw:space": "cosine"})

    total = 0
    for fichier in fichiers:
        source, texte = lire_document(fichier)
        morceaux = decouper(texte)
        embeddings = modele.encode(morceaux, normalize_embeddings=True).tolist()
        collection.add(
            ids=[f"{fichier.stem}-{i}" for i in range(len(morceaux))],
            documents=morceaux,
            embeddings=embeddings,
            metadatas=[{"source": source, "fichier": fichier.name, "passage": i}
                       for i in range(len(morceaux))],
        )
        print(f"{fichier.name} : {len(morceaux)} passages")
        total += len(morceaux)
    print(f"\nTerminé : {total} passages dans la base ({len(fichiers)} documents).")


if __name__ == "__main__":
    main()