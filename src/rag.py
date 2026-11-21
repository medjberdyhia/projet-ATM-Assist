"""RAG simple : retrouve des passages pertinents puis demande au LLM de répondre
uniquement à partir de ces passages, en citant ses sources."""
import os
import sys
import time

from dotenv import load_dotenv
from openai import APIError, AuthenticationError, OpenAI, RateLimitError

from config import MODELE_LLM, RACINE, SEUIL_SCORE
from search import rechercher

load_dotenv(RACINE / ".env")

REFUS = "Je ne trouve pas la réponse dans les documents."

SYSTEM_PROMPT = (
    "Tu es un assistant qui aide des ingénieurs à retrouver des informations sur la "
    "gestion du trafic aérien. Réponds en français, de façon claire et concise. "
    "Réponds UNIQUEMENT à partir des extraits fournis. Cite les extraits utilisés "
    "avec leur numéro entre crochets, par exemple [1]. "
    f"Si les extraits ne contiennent pas la réponse, dis exactement : « {REFUS} » "
    "N'invente rien."
)


def _client() -> OpenAI:
    cle = os.getenv("OPENROUTER_API_KEY")
    if not cle:
        raise ValueError("OPENROUTER_API_KEY introuvable : vérifie ton fichier .env")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=cle, timeout=90)


def construire_contexte(passages: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (source : {p['fichier']})\n{p['texte']}" for i, p in enumerate(passages, 1)
    )


def _appeler_llm(messages: list[dict], essais: int = 3) -> tuple[str, str]:
    """Appelle le LLM. Certains modèles gratuits « réfléchissent » longtemps et renvoient
    une réponse vide : on laisse plus de place (max_tokens) et on réessaie.
    Retourne (texte, nom_du_modele_qui_a_répondu)."""
    modele = MODELE_LLM
    for essai in range(1, essais + 1):
        try:
            reponse = _client().chat.completions.create(
                model=MODELE_LLM, messages=messages, temperature=0, max_tokens=2000
            )
        except AuthenticationError:
            return "Erreur : clé API invalide ou révoquée.", modele
        except RateLimitError:
            return "Erreur : limite de requêtes gratuites atteinte, attends une minute.", modele
        except APIError as e:
            return f"Erreur API OpenRouter : {e}", modele
        modele = getattr(reponse, "model", MODELE_LLM)
        texte = (reponse.choices[0].message.content or "").strip()
        if texte:
            return texte, modele
        print(f"  (réponse vide du modèle {modele}, essai {essai}/{essais})", file=sys.stderr)
        time.sleep(2)
    return "(réponse vide)", modele


def repondre(question: str, k: int = 3) -> dict:
    """Retourne {'reponse', 'sources', 'passages', 'meilleur_score', 'modele'}."""
    passages = rechercher(question, k=k)
    meilleur = passages[0]["score"] if passages else 0.0
    sources = sorted({p["fichier"] for p in passages})

    # Garde-fou : si aucun passage n'est assez proche, inutile d'interroger le LLM
    if meilleur < SEUIL_SCORE:
        return {"reponse": REFUS, "sources": [], "passages": passages,
                "meilleur_score": meilleur, "modele": None}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extraits :\n{construire_contexte(passages)}\n\nQuestion : {question}"},
    ]
    texte, modele = _appeler_llm(messages)
    return {"reponse": texte, "sources": sources, "passages": passages,
            "meilleur_score": meilleur, "modele": modele}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Qu'est-ce qu'un transpondeur ?"
    resultat = repondre(question)
    print(f"Question : {question}\n")
    print(resultat["reponse"])
    print("\nSources consultées :", ", ".join(resultat["sources"]) or "aucune")
    print(f"Meilleur score de recherche : {resultat['meilleur_score']} | Modèle : {resultat['modele']}")