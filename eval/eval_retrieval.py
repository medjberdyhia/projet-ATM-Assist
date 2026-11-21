"""Évaluation de la RECHERCHE (sans appeler de LLM, donc sans quota) :
- le bon document est-il retrouvé ? (hit@1, hit@3, hit@5, MRR)
- quel seuil de pertinence sépare les questions du corpus des questions hors sujet ?"""
import json
import sys
from pathlib import Path

DOSSIER_EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(DOSSIER_EVAL.parent / "src"))

from search import rechercher  # noqa: E402

K = 5


def evaluer() -> dict:
    questions = json.loads((DOSSIER_EVAL / "questions.json").read_text(encoding="utf-8"))
    lignes = []
    for q in questions:
        passages = rechercher(q["question"], k=K)
        meilleur = passages[0]["score"] if passages else 0.0
        rang = None  # rang du premier passage venant d'un fichier attendu
        for i, p in enumerate(passages, 1):
            if p["fichier"] in q["fichiers_attendus"]:
                rang = i
                break
        lignes.append({
            "id": q["id"], "question": q["question"], "hors_corpus": q["hors_corpus"],
            "attendus": q["fichiers_attendus"], "top1": passages[0]["fichier"] if passages else None,
            "meilleur_score": meilleur, "rang": rang,
        })

    dans = [l for l in lignes if not l["hors_corpus"]]
    hors = [l for l in lignes if l["hors_corpus"]]
    n = len(dans)
    resume = {
        "n_questions_corpus": n,
        "hit@1": sum(1 for l in dans if l["rang"] == 1) / n,
        "hit@3": sum(1 for l in dans if l["rang"] and l["rang"] <= 3) / n,
        "hit@5": sum(1 for l in dans if l["rang"]) / n,
        "MRR": sum(1 / l["rang"] for l in dans if l["rang"]) / n,
        "score_min_corpus": min(l["meilleur_score"] for l in dans),
        "score_max_hors_corpus": max(l["meilleur_score"] for l in hors) if hors else None,
    }
    return {"resume": resume, "lignes": lignes}


def main() -> None:
    res = evaluer()
    r = res["resume"]
    print(f"{'#':>2} {'rang':>4} {'score':>6}  {'top1':<42} question")
    for l in res["lignes"]:
        rang = "-" if l["rang"] is None else str(l["rang"])
        etiquette = "(hors corpus)" if l["hors_corpus"] else ""
        print(f"{l['id']:>2} {rang:>4} {l['meilleur_score']:>6}  {l['top1']:<42} {l['question']} {etiquette}")
    print(f"\nQuestions du corpus : {r['n_questions_corpus']}")
    print(f"hit@1 = {r['hit@1']:.0%} | hit@3 = {r['hit@3']:.0%} | hit@5 = {r['hit@5']:.0%} | MRR = {r['MRR']:.2f}")
    print(f"Plus petit meilleur score (questions du corpus)  : {r['score_min_corpus']}")
    print(f"Plus grand meilleur score (questions hors corpus) : {r['score_max_hors_corpus']}")
    bas, haut = r["score_max_hors_corpus"], r["score_min_corpus"]
    if bas is not None and bas < haut:
        print(f"→ Un seuil entre {bas} et {haut} sépare parfaitement les deux groupes "
              f"(par exemple {round((bas + haut) / 2, 2)}).")
    else:
        print("→ Les deux groupes se chevauchent : aucun seuil ne les sépare parfaitement ; "
              "choisis un compromis et note-le dans le journal.")
    (DOSSIER_EVAL / "resultats_retrieval.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRésultats enregistrés : eval/resultats_retrieval.json")


if __name__ == "__main__":
    main()