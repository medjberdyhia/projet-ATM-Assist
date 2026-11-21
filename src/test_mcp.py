"""Teste le serveur MCP comme le ferait un vrai client : liste des outils, puis appels."""
import asyncio

from fastmcp import Client

from mcp_server import mcp


async def main() -> None:
    async with Client(mcp) as client:
        outils = await client.list_tools()
        print("Outils exposés :", [o.name for o in outils])

        docs = await client.call_tool("list_documents", {})
        print("\nlist_documents ->", docs.data)

        res = await client.call_tool("search_docs", {"question": "Qu'est-ce qu'un transpondeur ?", "k": 2})
        print("\nsearch_docs ->")
        for p in res.data:
            print(f"  {p['score']} | {p['fichier']} | {p['texte'][:80]}...")

        hors_sujet = await client.call_tool("search_docs", {"question": "Quelle est la capitale du Japon ?"})
        print("\nQuestion hors sujet ->", hors_sujet.data)


if __name__ == "__main__":
    asyncio.run(main())