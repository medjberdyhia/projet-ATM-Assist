import asyncio

import search
import mcp_server
from fastmcp import Client


def appeler(outil, arguments):
    async def _run():
        async with Client(mcp_server.mcp) as client:
            return await client.call_tool(outil, arguments)

    return asyncio.run(_run())


def test_les_deux_outils_sont_exposes():
    async def _run():
        async with Client(mcp_server.mcp) as client:
            return [o.name for o in await client.list_tools()]

    assert set(asyncio.run(_run())) == {"search_docs", "list_documents"}


def test_search_docs_ne_renvoie_que_les_passages_pertinents(monkeypatch):
    faux = [
        {"texte": "pertinent", "source": "u", "fichier": "a.txt", "score": 0.8},
        {"texte": "hors sujet", "source": "u", "fichier": "b.txt", "score": 0.1},
    ]
    monkeypatch.setattr(search, "rechercher", lambda q, k=3: faux)
    resultat = appeler("search_docs", {"question": "test"})
    assert [p["fichier"] for p in resultat.data] == ["a.txt"]


def test_search_docs_borne_le_nombre_de_resultats(monkeypatch):
    demandes = []
    monkeypatch.setattr(search, "rechercher", lambda q, k=3: demandes.append(k) or [])
    appeler("search_docs", {"question": "test", "k": 1000})
    assert demandes == [10]


def test_list_documents_liste_les_fichiers_txt(monkeypatch, tmp_path):
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(mcp_server, "DOSSIER_DATA", tmp_path)
    assert appeler("list_documents", {}).data == ["a.txt", "b.txt"]