"""Évaluation de la RECHERCHE (sans appeler de LLM, donc sans quota) :
- niveau DOCUMENT : le bon document est-il retrouvé ? (hit@1, hit@3, hit@5, MRR)
- niveau PASSAGE : un des passages retrouvés contient-il vraiment la réponse ? (mot_cle)
- quel seuil de pertinence sépare les questions du corpus des questions hors sujet ?"""
import json
import sys
from pathlib import Path

DOSSIER_EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(DOSSIER_EVAL.parent / "src"))

from search import rechercher  # noqa: E402

K = 5
MODE = sys.argv[1] if len(sys.argv) > 1 else "defaut"   # vecteur | hybride | defaut (valeur de config.py)
HYBRIDE = {"vecteur": False, "hybride": True}.get(MODE)  # None = valeur de config.py

def normaliser(texte: str) -> str:
    """Minuscules, sans espaces (y compris insécables) : « 1 030 » et « 1030 » deviennent égaux."""
    for espace in ("\u202f", "\xa0", " ", "\n", "\t"):
        texte = texte.replace(espace, "")
    return texte.lower()


def evaluer() -> dict:
    questions = json.loads((DOSSIER_EVAL / "questions.json").read_text(encoding="utf-8"))
    lignes = []
    for q in questions:
        passages = rechercher(q["question"], k=K, hybride=HYBRIDE)
        meilleur = max((p["score"] for p in passages), default=0.0)
        rang = None  # rang du premier passage venant d'un fichier attendu
        for i, p in enumerate(passages, 1):
            if p["fichier"] in q["fichiers_attendus"]:
                rang = i
                break
        rang_mot = None  # rang du premier passage qui contient vraiment le mot-clé de la réponse
        mot = q.get("mot_cle")
        if mot:
            for i, p in enumerate(passages, 1):
                if normaliser(mot) in normaliser(p["texte"]):
                    rang_mot = i
                    break
        lignes.append({
            "id": q["id"], "question": q["question"], "hors_corpus": q["hors_corpus"],
            "attendus": q["fichiers_attendus"], "top1": passages[0]["fichier"] if passages else None,
            "meilleur_score": meilleur, "rang": rang, "mot_cle": mot, "rang_mot_cle": rang_mot,
        })

    dans = [l for l in lignes if not l["hors_corpus"]]
    hors = [l for l in lignes if l["hors_corpus"]]
    avec_mot = [l for l in dans if l["mot_cle"]]
    n = len(dans)
    resume = {
        "n_questions_corpus": n,
        "hit@1": sum(1 for l in dans if l["rang"] == 1) / n,
        "hit@3": sum(1 for l in dans if l["rang"] and l["rang"] <= 3) / n,
        "hit@5": sum(1 for l in dans if l["rang"]) / n,
        "MRR": sum(1 / l["rang"] for l in dans if l["rang"]) / n,
        "n_questions_avec_mot_cle": len(avec_mot),
        "passage_hit@3": (sum(1 for l in avec_mot if l["rang_mot_cle"] and l["rang_mot_cle"] <= 3) / len(avec_mot)) if avec_mot else None,
        "passage_hit@5": (sum(1 for l in avec_mot if l["rang_mot_cle"]) / len(avec_mot)) if avec_mot else None,
        "score_min_corpus": min(l["meilleur_score"] for l in dans),
        "score_max_hors_corpus": max(l["meilleur_score"] for l in hors) if hors else None,
    }
    return {"resume": resume, "lignes": lignes}


def main() -> None:
    print(f"Mode de recherche : {MODE}")
    res = evaluer()
    r = res["resume"]
    print(f"{'#':>2} {'doc':>4} {'mot':>4} {'score':>6}  {'top1':<42} question")
    for l in res["lignes"]:
        doc = "-" if l["rang"] is None else str(l["rang"])
        mot = "" if not l["mot_cle"] else ("-" if l["rang_mot_cle"] is None else str(l["rang_mot_cle"]))
        etiquette = "(hors corpus)" if l["hors_corpus"] else ""
        print(f"{l['id']:>2} {doc:>4} {mot:>4} {l['meilleur_score']:>6}  {l['top1']:<42} {l['question']} {etiquette}")
    print("\n(doc = rang du 1er passage du bon document ; mot = rang du 1er passage contenant la réponse)")
    print(f"\nNIVEAU DOCUMENT ({r['n_questions_corpus']} questions du corpus)")
    print(f"hit@1 = {r['hit@1']:.0%} | hit@3 = {r['hit@3']:.0%} | hit@5 = {r['hit@5']:.0%} | MRR = {r['MRR']:.2f}")
    if r["passage_hit@5"] is not None:
        print(f"\nNIVEAU PASSAGE ({r['n_questions_avec_mot_cle']} questions avec mot-clé de la réponse)")
        print(f"passage hit@3 = {r['passage_hit@3']:.0%} | passage hit@5 = {r['passage_hit@5']:.0%}")
    print(f"\nPlus petit meilleur score (questions du corpus)  : {r['score_min_corpus']}")
    print(f"Plus grand meilleur score (questions hors corpus) : {r['score_max_hors_corpus']}")
    (DOSSIER_EVAL / f"resultats_retrieval_{MODE}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRésultats enregistrés : eval/resultats_retrieval.json")


if __name__ == "__main__":
    main()