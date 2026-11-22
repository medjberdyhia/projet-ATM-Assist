"""Évaluation des RÉPONSES : compare plusieurs modèles sur les questions de questions.json.
- Les réponses sont mises en cache (cache_generation.json) : relancer le script ne
  refait JAMAIS un appel déjà réussi, et reprend là où on s'est arrêté (quota atteint).
- Mesures automatiques (sans LLM juge) : refus, citations, nombres non fondés, latence."""
import json
import re
import statistics
import sys
import time
from pathlib import Path

DOSSIER_EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(DOSSIER_EVAL.parent / "src"))

from rag import repondre  # noqa: E402

# Identifiants FIXES (pas le routeur) pour que la comparaison soit reproductible.
# Remplace-les si l'un n'est plus disponible (openrouter.ai/models, filtre « Free »).
MODELES = ["recursive/flint-v1.0:free", "nvidia/nemotron-3.5-super-vl:free"]
PAUSE_ENTRE_APPELS = 4  # secondes : on reste sous la limite de requêtes par minute
CACHE = DOSSIER_EVAL / "cache_generation.json"

MOTIFS_REFUS = ["je ne trouve pas", "ne dispose pas de document", "pas dans les documents",
                "ne concerne pas", "aucun passage", "ne contient pas d'information"]


def est_refus(reponse: str) -> bool:
    debut = reponse.lower()[:250]
    return any(m in debut for m in MOTIFS_REFUS)


def nombres(texte: str) -> set[str]:
    """Nombres à 2 chiffres ou plus, sans espaces (1 090 -> 1090), hors marqueurs [1]."""
    texte = re.sub(r"\[\d+\]", " ", texte).replace("\u202f", " ").replace("\xa0", " ")
    trouves = re.findall(r"\d[\d ]*\d", texte)
    return {re.sub(r"\s", "", t) for t in trouves}


def nombres_non_fondes(reponse: str, contexte: str) -> list[str]:
    ctx = re.sub(r"\s", "", contexte.replace("\u202f", " ").replace("\xa0", " "))
    return sorted(n for n in nombres(reponse) if n not in ctx)


def charger_cache() -> dict:
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def collecter(questions: list[dict]) -> None:
    cache = charger_cache()
    for modele in MODELES:
        for q in questions:
            cle = f"{modele}|{q['id']}"
            if cle in cache:
                continue
            debut = time.perf_counter()
            res = repondre(q["question"], k=3, modele=modele)
            duree = round(time.perf_counter() - debut, 1)
            appel_llm = res["modele"] is not None
            reponse = res["reponse"]
            if reponse.startswith("Erreur") or reponse == "(réponse vide)":
                print(f"  ⚠ {modele} Q{q['id']} : {reponse}")
                if "limite de requêtes" in reponse:
                    print("Quota atteint : résultats déjà obtenus sauvegardés. Relance plus tard.")
                    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
                    return
                continue  # on ne met pas en cache : une relance réessaiera
            cache[cle] = {
                "reponse": reponse, "duree_s": duree, "appel_llm": appel_llm,
                "meilleur_score": res["meilleur_score"], "modele_servi": res["modele"],
                "contexte": " ".join(p["texte"] for p in res["passages"]),
            }
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  ✓ {modele} Q{q['id']} ({duree}s)")
            if appel_llm:
                time.sleep(PAUSE_ENTRE_APPELS)


def synthese(questions: list[dict]) -> str:
    cache = charger_cache()
    lignes = ["| Modèle | Réponses obtenues | Faux refus | Refus corrects (hors corpus) | Avec citation | Réponses avec nombre non fondé | Latence médiane |",
              "|---|---|---|---|---|---|---|"]
    for modele in MODELES:
        dans, hors = [], []
        for q in questions:
            r = cache.get(f"{modele}|{q['id']}")
            if r:
                (hors if q["hors_corpus"] else dans).append(r)
        faux_refus = sum(est_refus(r["reponse"]) for r in dans)
        refus_ok = sum(est_refus(r["reponse"]) for r in hors)
        cites = sum(bool(re.search(r"\[\d+\]|source", r["reponse"].lower())) for r in dans)
        non_fondes = sum(bool(nombres_non_fondes(r["reponse"], r["contexte"])) for r in dans)
        durees = [r["duree_s"] for r in dans if r["appel_llm"]]
        lat = f"{statistics.median(durees):.1f} s" if durees else "-"
        lignes.append(f"| {modele} | {len(dans)}/{sum(not q['hors_corpus'] for q in questions)} corpus, "
                      f"{len(hors)}/{sum(q['hors_corpus'] for q in questions)} hors | {faux_refus} | "
                      f"{refus_ok}/{len(hors)} | {cites}/{len(dans)} | {non_fondes}/{len(dans)} | {lat} |")
    return "\n".join(lignes)


def main() -> None:
    questions = json.loads((DOSSIER_EVAL / "questions.json").read_text(encoding="utf-8"))
    print("Collecte des réponses (les réponses déjà en cache ne sont pas redemandées)...")
    collecter(questions)
    tableau = synthese(questions)
    print("\n" + tableau)
    (DOSSIER_EVAL / "resultats_generation.md").write_text(tableau + "\n", encoding="utf-8")
    cache = charger_cache()
    print("\nNombres non fondés (à vérifier à la main) :")
    for cle, r in cache.items():
        modele, idq = cle.rsplit("|", 1)
        suspects = nombres_non_fondes(r["reponse"], r["contexte"])
        if suspects:
            print(f"  {modele} Q{idq} : {suspects}")


if __name__ == "__main__":
    main()