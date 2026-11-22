from search import fusion_rrf, scores_bm25, tokeniser


def test_tokeniser_retire_accents_mots_vides_et_pluriels():
    assert tokeniser("Combien de pistes possède l'aéroport ?") == ["piste", "possede", "aeroport"]


def test_tokeniser_rassemble_les_nombres_avec_espace():
    assert "1030" in tokeniser("fréquence de 1 030 MHz")


def test_bm25_classe_en_premier_le_passage_qui_contient_le_mot_rare():
    corpus = [
        tokeniser("L'aéroport s'étale sur trois départements et une superficie de 32 km2."),
        tokeniser("L'aéroport, avec ses quatre pistes, son héliport et ses zones d'activités."),
        tokeniser("Les terminaux accueillent les passagers de l'aéroport."),
    ]
    scores = scores_bm25(corpus, tokeniser("Combien de pistes possède l'aéroport ?"))
    assert int(scores.argmax()) == 1


def test_bm25_renvoie_zero_si_aucun_mot_en_commun():
    scores = scores_bm25([tokeniser("radar secondaire")], tokeniser("pâte à crêpes"))
    assert scores.tolist() == [0.0]


def test_fusion_rrf_fait_remonter_un_passage_bien_classe_par_un_seul_classement():
    vecteur = [0, 1, 2, 3]
    lexical = [3]
    assert fusion_rrf(vecteur, lexical) == [3, 0, 1, 2]