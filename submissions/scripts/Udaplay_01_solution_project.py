import os
import sys
import json
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 0.  Set up environment and imports
# ---------------------------------------------------------------------------

# Load environment variables from .env file if present
load_dotenv()

# Verify API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_OPENAI_API_KEY = os.getenv("CHROMA_OPENAI_API_KEY")

if not OPENAI_API_KEY or not CHROMA_OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY and/or CHROMA_OPENAI_API_KEY are missing.\n"
        "Create a .env file or export them in your shell before running."
    )

# Configure Vocareum endpoints automatically if voc- prefix is detected
if OPENAI_API_KEY.startswith("voc-"):
    os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

# Make sure our project libraries are importable
PROJECT_LIB = (
    Path(__file__).parent.parent / "projects" / "building-agents" / "src" / "project" / "starter"
)
if str(PROJECT_LIB) not in sys.path:
    sys.path.append(str(PROJECT_LIB))

from lib.documents import Document, Corpus  # noqa: E402
from lib.vector_db import VectorStoreManager, VectorStore  # noqa: E402

# ---------------------------------------------------------------------------
# 1.  Load and explore game data
# ---------------------------------------------------------------------------

def load_games() -> List[Dict]:
    """Load all JSON game files into memory."""
    games_dir = PROJECT_LIB / "games"
    games: List[Dict] = []
    for json_file in sorted(games_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as fp:
            games.append(json.load(fp))
    print(f"Loaded {len(games)} game files from {games_dir.relative_to(Path.cwd())}")
    if games:
        print("Example game keys:", list(games[0].keys()))
    return games

# ---------------------------------------------------------------------------
# 2.  Convert raw game dicts into Document objects for embedding
# ---------------------------------------------------------------------------

def create_game_document(game_data: Dict, index: int) -> Document:
    """Convert a single game dictionary into a Document."""
    name = game_data.get("Name", "Unknown")
    platform = game_data.get("Platform", "Unknown")
    genre = game_data.get("Genre", "Unknown")
    publisher = game_data.get("Publisher", "Unknown")
    release_year = game_data.get("YearOfRelease", "Unknown")
    description = game_data.get("Description", "No description available")

    content = "\n".join(
        [
            f"Game: {name}",
            f"Platform: {platform}",
            f"Genre: {genre}",
            f"Publisher: {publisher}",
            f"Release Year: {release_year}",
            f"Description: {description}",
        ]
    )

    metadata = {
        "name": name,
        "platform": platform,
        "genre": genre,
        "publisher": publisher,
        "release_year": str(release_year),
        "description": description,
    }

    clean_name = (
        name.lower()
        .replace(" ", "_")
        .replace(":", "")
        .replace("-", "_")
        .replace("'", "")
    )
    doc_id = f"game_{index:03d}_{clean_name}"

    return Document(id=doc_id, content=content, metadata=metadata)


def build_corpus(games: List[Dict]) -> Corpus:
    docs = [create_game_document(game, i) for i, game in enumerate(games)]
    corpus = Corpus(docs)
    print(f"Created {len(corpus)} Document objects (all IDs unique ✔️)")
    return corpus

# ---------------------------------------------------------------------------
# 3.  Vector store setup (ChromaDB + OpenAI embeddings)
# ---------------------------------------------------------------------------

import chromadb  # noqa: E402
from chromadb.utils import embedding_functions  # noqa: E402


class VocareumVectorStoreManager:
    """Thin wrapper around ChromaDB to support Vocareum endpoints."""

    def __init__(self, openai_api_key: str):
        # Use persistent client so data survives between script runs
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = self._create_embedding_function(openai_api_key)

    def _create_embedding_function(self, api_key: str):
        if api_key.startswith("voc-"):
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key, api_base="https://openai.vocareum.com/v1"
            )
        return embedding_functions.OpenAIEmbeddingFunction(api_key=api_key)

    # Public helpers --------------------------------------------------------
    def create_store(self, name: str, force: bool = False) -> VectorStore:
        if force:
            try:
                self.client.delete_collection(name=name)
            except Exception:
                pass  # ignore if collection didn't previously exist
        collection = self.client.get_or_create_collection(
            name=name, embedding_function=self.embedding_function
        )
        return VectorStore(collection)

    def get_store(self, name: str) -> VectorStore | None:
        try:
            return VectorStore(self.client.get_collection(name=name))
        except Exception:
            return None

# ---------------------------------------------------------------------------
# 4.  Index documents into ChromaDB
# ---------------------------------------------------------------------------

def index_documents(corpus: Corpus, store_name: str = "udaplay_games") -> VectorStore:
    vector_manager = VocareumVectorStoreManager(CHROMA_OPENAI_API_KEY)
    vec_store = vector_manager.create_store(store_name, force=True)
    print("Adding documents to vector store – this may take a moment…")
    vec_store.add(corpus)
    print(f"Successfully indexed {len(corpus)} documents into '{store_name}'")
    return vec_store

# ---------------------------------------------------------------------------
# 5.  Helper to pretty-print search results
# ---------------------------------------------------------------------------

def display_search_results(query: str, results: Dict):
    print("=" * 70)
    print(f"Query → {query}")
    print("=" * 70)
    if results["documents"] and results["documents"][0]:
        for i, (doc, distance, meta) in enumerate(
            zip(results["documents"][0], results["distances"][0], results["metadatas"][0])
        ):
            similarity = 1 - distance
            print(
                f"[{i+1}] {meta['name']} ({meta['release_year']}, {meta['platform']}) – "
                f"sim={similarity:.3f}"
            )
            print(f"    Genre: {meta['genre']} | Publisher: {meta['publisher']}")
            print(f"    Description: {meta['description'][:120]}…\n")
    else:
        print("No results found.\n")

# ---------------------------------------------------------------------------
# 6.  Demo searches (semantic + metadata filtering)
# ---------------------------------------------------------------------------

def run_demo_searches(store: VectorStore):
    demo_queries = [
        "Pokemon games from the 90s",
        "First 3D Mario platformer",
        "Mortal Kombat fighting game",
        "RPG games by Nintendo",
        "Games released in 1999",
    ]
    for q in demo_queries:
        res = store.query(query_texts=[q], n_results=3)
        display_search_results(q, res)

    # Metadata-only example (distances not included in get method)
    nintendo = store.get(where={"publisher": "Nintendo"}, limit=5)
    print("\nNintendo-published titles (metadata filter):")
    for idx, meta in enumerate(nintendo["metadatas"], start=1):
        print(f"  {idx}. {meta['name']} ({meta['release_year']}) – {meta['platform']}")

    # Mixed example
    filtered = store.query(
        query_texts=["adventure game"], n_results=3, where={"platform": "Nintendo 64"}
    )
    print("\nAdventure games on Nintendo 64:")
    for meta in filtered["metadatas"][0]:
        print(f"  – {meta['name']} ({meta['genre']})")

# ---------------------------------------------------------------------------
# 7.  Main entrypoint
# ---------------------------------------------------------------------------

def main():
    games = load_games()
    corpus = build_corpus(games)
    vec_store = index_documents(corpus)
    run_demo_searches(vec_store)
    print("\nAll done – vector database ready for Part 2! ✔️")


if __name__ == "__main__":
    main() 