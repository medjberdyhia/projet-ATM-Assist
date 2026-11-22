import sys
from pathlib import Path

# Permet aux tests d'importer les modules de src/ (config, ingest, rag, mcp_server...)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
