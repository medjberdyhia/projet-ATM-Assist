"""Agent ReAct (LangGraph) : le LLM décide lui-même quand chercher dans les documents,
avec quelle formulation, et combien de fois, avant de répondre en citant ses sources."""
import os
import sys

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import DOSSIER_DATA, MODELE_AGENT, SEUIL_SCORE
from rag import REFUS, construire_contexte  # charge aussi le fichier .env
from search import rechercher

SYSTEM_PROMPT = (
    "Tu es un assistant qui aide des ingénieurs à retrouver des informations sur la "
    "gestion du trafic aérien, à partir d'une base de documents. "
    "Pour toute question factuelle, utilise l'outil search_docs. Tu peux le rappeler "
    "avec une autre formulation si le premier résultat est insuffisant. "
    "Utilise list_documents si on te demande ce que contient la base. "
    "Réponds en français, de façon claire et concise, UNIQUEMENT à partir des résultats "
    "des outils, et cite le nom du fichier source entre parenthèses, par exemple "
    "(source : transpondeur.txt). "
    f"Si les résultats ne contiennent pas la réponse, réponds exactement : « {REFUS} » "
    "N'invente rien."
)


@tool
def search_docs(question: str) -> str:
    """Recherche dans la base documentaire (gestion du trafic aérien) et retourne les
    passages les plus pertinents avec leur source. À utiliser pour toute question
    factuelle sur le contrôle aérien, les radars, les transpondeurs, Eurocontrol, etc."""
    passages = rechercher(question, k=4)
    if not passages or max(p["score"] for p in passages) < SEUIL_SCORE:
        return "Aucun passage pertinent trouvé."
    return construire_contexte(passages)


@tool
def list_documents() -> str:
    """Liste les documents disponibles dans la base documentaire."""
    return "\n".join(sorted(p.name for p in DOSSIER_DATA.glob("*.txt")))


def construire_agent():
    cle = os.getenv("OPENROUTER_API_KEY")
    if not cle:
        raise ValueError("OPENROUTER_API_KEY introuvable : vérifie ton fichier .env")
    modele = ChatOpenAI(
        model=MODELE_AGENT,
        api_key=cle,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=4000,   # de la place pour les modèles qui « réfléchissent »
        timeout=120,
        max_retries=2,
        extra_body={"reasoning": {"effort": "low"}},  # réflexion courte (ignoré si non supporté)
    )
    return create_agent(modele, tools=[search_docs, list_documents], system_prompt=SYSTEM_PROMPT)


def _texte(contenu) -> str:
    """Certains modèles renvoient une liste de blocs au lieu d'une chaîne."""
    if isinstance(contenu, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in contenu)
    return str(contenu or "")


def poser_question(question: str, agent=None) -> dict:
    """Retourne {'reponse': str, 'outils': [(nom, arguments), ...]}."""
    agent = agent or construire_agent()
    try:
        resultat = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": 12},  # évite les boucles infinies
        )
    except Exception as e:  # noqa: BLE001
        return {"reponse": f"Erreur : {e}", "outils": []}
    outils = [
        (appel["name"], appel["args"])
        for message in resultat["messages"]
        for appel in (getattr(message, "tool_calls", None) or [])
    ]
    reponse = _texte(resultat["messages"][-1].content).strip() or "(réponse vide)"
    return {"reponse": reponse, "outils": outils}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Qu'est-ce qu'un transpondeur ?"
    sortie = poser_question(question)
    print(f"Question : {question}\n")
    for nom, args in sortie["outils"]:
        print(f"  → outil appelé : {nom}({args})")
    print(f"\n{sortie['reponse']}")