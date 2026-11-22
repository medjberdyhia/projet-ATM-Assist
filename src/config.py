from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_DATA = RACINE / "data"
CHROMA_PATH = RACINE / "chroma_db"
NOM_COLLECTION = "atm_docs"
MODELE_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"
MODELE_LLM = "openrouter/free"
SEUIL_SCORE = 0.40  # en dessous, on considère que les documents ne couvrent pas la question
MODELE_AGENT = "openrouter/free"
RECHERCHE_HYBRIDE = True