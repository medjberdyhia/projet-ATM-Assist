"""Serveur MCP (FastMCP) : expose la base documentaire à n'importe quel client MCP
(agent LangGraph, Claude Desktop, MCP Inspector...). Chaque appel est journalisé."""
import logging
import time

from fastmcp import FastMCP

from config import DOSSIER_DATA, RACINE, SEUIL_SCORE

# IMPORTANT : un serveur MCP en mode « stdio » communique par la sortie standard.
# On n'utilise donc JAMAIS print() ici : les logs vont dans un fichier.
logging.basicConfig(
    filename=RACINE / "mcp_calls.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("atm-assist-mcp")

mcp = FastMCP(
    name="atm-assist",
    instructions="Base documentaire sur la gestion du trafic aérien (documents publics). "
    "Utilise search_docs pour répondre à une question factuelle, "
    "list_documents pour connaître les documents disponibles.",
)


@mcp.tool
def search_docs(question: str, k: int = 3) -> list[dict]:
    """Recherche sémantique dans la base documentaire sur la gestion du trafic aérien.
    Retourne les k passages les plus proches de la question, avec leur fichier source
    et un score de pertinence entre 0 et 1 (1 = très proche)."""
    from search import rechercher  # import tardif : le démarrage du serveur reste rapide

    debut = time.perf_counter()
    k = max(1, min(k, 10))  # on borne k pour éviter les abus
    try:
        passages = rechercher(question, k=k)
    except Exception as e:  # noqa: BLE001
        log.error("search_docs ERREUR question=%r : %s", question, e)
        raise
    pertinents = [p for p in passages if p["score"] >= SEUIL_SCORE]
    log.info("search_docs question=%r k=%d -> %d pertinents, meilleur=%s, %.2fs",
             question, k, len(pertinents), passages[0]["score"] if passages else None,
             time.perf_counter() - debut)
    return pertinents


@mcp.tool
def list_documents() -> list[str]:
    """Liste les noms de fichiers des documents disponibles dans la base."""
    fichiers = sorted(p.name for p in DOSSIER_DATA.glob("*.txt"))
    log.info("list_documents -> %d documents", len(fichiers))
    return fichiers


if __name__ == "__main__":
    mcp.run()  # transport stdio par défaut