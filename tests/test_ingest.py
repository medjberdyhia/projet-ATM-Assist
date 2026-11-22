from ingest import decouper, lire_document


def test_decouper_respecte_la_taille_maximale():
    texte = "Un transpondeur répond à une interrogation radar. " * 100
    morceaux = decouper(texte, taille=500, chevauchement=50)
    assert len(morceaux) > 1
    assert all(len(m) <= 500 for m in morceaux)


def test_decouper_garde_un_chevauchement_entre_passages():
    texte = "Un transpondeur répond à une interrogation radar. " * 100
    morceaux = decouper(texte, taille=500, chevauchement=50)
    assert morceaux[0][-20:] in morceaux[1]


def test_decouper_ignore_les_textes_trop_courts():
    assert decouper("Texte trop court.") == []


def test_lire_document_extrait_la_source_et_retire_la_fin_d_article(tmp_path):
    fichier = tmp_path / "doc.txt"
    fichier.write_text(
        "Source : https://exemple.org/page\n\nContenu utile.\n\n== Voir aussi ==\nLien inutile",
        encoding="utf-8",
    )
    source, texte = lire_document(fichier)
    assert source == "https://exemple.org/page"
    assert "Contenu utile" in texte
    assert "Lien inutile" not in texte