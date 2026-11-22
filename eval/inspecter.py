"""Affiche les réponses « suspectes » du cache : faux refus et réponses sans citation.
Aucun appel à OpenRouter : on lit seulement cache_generation.json."""
import json
import re

from eval_generation import DOSSIER_EVAL, MODELES, charger_cache, est_refus

questions = {q["id"]: q for q in json.loads((DOSSIER_EVAL / "questions.json").read_text(encoding="utf-8"))}
cache = charger_cache()

for modele in MODELES:
    for idq, q in questions.items():
        r = cache.get(f"{modele}|{idq}")
        if not r or q["hors_corpus"]:
            continue
        refus = est_refus(r["reponse"])
        sans_citation = not re.search(r"\[\d+\]|source", r["reponse"].lower())
        if refus or sans_citation:
            etiquette = "FAUX REFUS" if refus else "SANS CITATION"
            print(f"=== {etiquette} | {modele} | Q{idq} : {q['question']}")
            print(f"Réponse : {r['reponse'][:500]}")
            print(f"Meilleur score : {r['meilleur_score']} | Fichiers attendus : {q['fichiers_attendus']}")
            print(f"Début du contexte envoyé : {r['contexte'][:400]}\n")