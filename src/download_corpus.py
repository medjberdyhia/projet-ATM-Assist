"""Télécharge des pages Wikipédia FR (texte brut) dans data/."""
import re
import time
import unicodedata
from pathlib import Path

import requests

TITRES = [
    "Contrôle de la circulation aérienne",
    "Gestion du trafic aérien",
    "Radar secondaire",
    "Automatic dependent surveillance-broadcast",
    "Transpondeur",
    "Espace aérien",
    "Règles de vol aux instruments",
    "Eurocontrol",
    "Ciel unique européen",
    "Aéroport de Paris-Charles-de-Gaulle",
]

API = "https://fr.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "atm-assist/0.1 (projet etudiant, https://github.com/medjberdyhia/projet-ATM-Assist)"}
DOSSIER = Path("data")


def nom_fichier(titre: str) -> str:
    titre = unicodedata.normalize("NFKD", titre).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", titre.lower()).strip("_") + ".txt"


def telecharger(titre: str):
    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": 1, "redirects": 1, "titles": titre,
    }
    reponse = requests.get(API, params=params, headers=HEADERS, timeout=30)
    reponse.raise_for_status()
    page = next(iter(reponse.json()["query"]["pages"].values()))
    return page.get("title"), page.get("extract")


def main() -> None:
    DOSSIER.mkdir(exist_ok=True)
    for titre in TITRES:
        titre_reel, texte = telecharger(titre)
        if not texte or len(texte) < 500:
            print(f"IGNORÉ (page absente ou trop courte) : {titre}")
            continue
        url = "https://fr.wikipedia.org/wiki/" + titre_reel.replace(" ", "_")
        (DOSSIER / nom_fichier(titre_reel)).write_text(
            f"Source : {url}\n\n{texte}", encoding="utf-8"
        )
        print(f"OK : {titre_reel} ({len(texte)} caractères)")
        time.sleep(1)  # on reste poli avec le serveur


if __name__ == "__main__":
    main()