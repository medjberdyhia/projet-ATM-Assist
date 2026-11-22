"""Donne le rang du premier passage contenant un mot-clé parmi les k premiers résultats.
Usage : python eval/rang_mot_cle.py "question" "mot" [k]   (aucun appel à OpenRouter)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from search import rechercher  # noqa: E402

question, mot = sys.argv[1], sys.argv[2].lower()
k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
for rang, p in enumerate(rechercher(question, k=k), 1):
    if mot in p["texte"].lower():
        print(f"Premier passage contenant « {mot} » : rang {rang} (score {p['score']}, {p['fichier']})")
        print(p["texte"][:300])
        break
else:
    print(f"Aucun passage contenant « {mot} » parmi les {k} premiers.")