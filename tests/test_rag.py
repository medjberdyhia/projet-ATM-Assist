import rag


def passage(score, fichier="transpondeur.txt"):
    return {"texte": "Un transpondeur répond au radar.", "source": "url", "fichier": fichier, "score": score}


def test_question_hors_corpus_refusee_sans_appeler_le_llm(monkeypatch):
    monkeypatch.setattr(rag, "rechercher", lambda q, k=3: [passage(0.10)])

    def ne_doit_pas_etre_appele(*args, **kwargs):
        raise AssertionError("Le LLM ne doit pas être appelé sous le seuil")

    monkeypatch.setattr(rag, "_appeler_llm", ne_doit_pas_etre_appele)
    resultat = rag.repondre("Quelle est la capitale du Japon ?")
    assert resultat["reponse"] == rag.REFUS
    assert resultat["modele"] is None


def test_question_du_corpus_appelle_le_llm_et_renvoie_les_sources(monkeypatch):
    monkeypatch.setattr(rag, "rechercher", lambda q, k=3: [passage(0.80), passage(0.70, "radar_secondaire.txt")])
    monkeypatch.setattr(rag, "_appeler_llm", lambda messages, modele_llm=None: ("Réponse [1]", "modele-test"))
    resultat = rag.repondre("Qu'est-ce qu'un transpondeur ?")
    assert resultat["reponse"] == "Réponse [1]"
    assert resultat["sources"] == ["radar_secondaire.txt", "transpondeur.txt"]
    assert resultat["modele"] == "modele-test"


def test_contexte_numerote_avec_la_source():
    contexte = rag.construire_contexte([passage(0.8), passage(0.7, "radar_secondaire.txt")])
    assert "[1] (source : transpondeur.txt)" in contexte
    assert "[2] (source : radar_secondaire.txt)" in contexte