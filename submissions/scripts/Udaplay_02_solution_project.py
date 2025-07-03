"""UdaPlay Part 2: AI Agent Development (Script Version)

This script is a .py replica of the `Udaplay_02_solution_project.ipynb` notebook. You can
run it incrementally (e.g., with `python -i` or by executing sections) and iterate more
rapidly. Once satisfied, you can copy/paste back into a notebook if desired.

Usage:
    python submissions/Udaplay_02_solution_project.py

Prerequisites:
    1. Create a `.env` file in the project root with:
        OPENAI_API_KEY="your-openai-api-key"
        CHROMA_OPENAI_API_KEY="your-openai-api-key"  # can be same as OPENAI_API_KEY
        TAVILY_API_KEY="your-tavily-api-key"

    2. Run `Udaplay_01_solution_project.py` (or the corresponding notebook) first to
       build the `udaplay_games` vector store.

    3. Ensure the following packages are installed (matching versions from requirements):
        chromadb, openai, python-dotenv, requests, pydantic, pdfplumber

This script will:
    • Connect to the existing vector store
    • Define three tools: retrieve_game, evaluate_retrieval, game_web_search
    • Build a custom UdaPlayAgent that follows the RAG → Evaluate → Web Search workflow
    • Demonstrate the agent with sample queries when run as __main__
"""

# ---------------------------------------------------------------------------
# Imports & Environment Setup
# ---------------------------------------------------------------------------
import os
import sys
import json
from typing import List, Dict, Optional

from dotenv import load_dotenv
import requests

# Add the project's lib directory to PYTHONPATH
PROJECT_LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "projects", "building-agents", "src", "project", "starter")
if PROJECT_LIB_PATH not in sys.path:
    sys.path.append(PROJECT_LIB_PATH)

from lib.llm import LLM
from lib.agents import Agent, AgentState
from lib.tooling import tool, Tool
from lib.vector_db import VectorStoreManager
from lib.state_machine import StateMachine, Step, EntryPoint, Termination
from lib.messages import AIMessage, UserMessage, SystemMessage, ToolMessage

# Load env vars
load_dotenv()

# Configure for Vocareum if using voc- keys
if os.environ.get("OPENAI_API_KEY", "").startswith("voc-"):
    print("Detected Vocareum OpenAI API key - configuring for Vocareum endpoint")
    os.environ['OPENAI_API_BASE'] = 'https://openai.vocareum.com/v1'

# Verify essential API keys
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found in environment"
assert os.getenv("CHROMA_OPENAI_API_KEY"), "CHROMA_OPENAI_API_KEY not found in environment"
assert os.getenv("TAVILY_API_KEY"), "TAVILY_API_KEY not found in environment"

print("✅ Environment variables loaded.")

# ---------------------------------------------------------------------------
# Vector Store Connection (using same VocareumVectorStoreManager as Part 1)
# ---------------------------------------------------------------------------
import chromadb
from chromadb.utils import embedding_functions

class VocareumVectorStoreManager:
    """Same vector store manager as Part 1 to ensure compatibility."""
    
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

    def get_store(self, name: str):
        try:
            from lib.vector_db import VectorStore
            return VectorStore(self.client.get_collection(name=name))
        except Exception:
            return None

vector_manager = VocareumVectorStoreManager(openai_api_key=os.getenv("CHROMA_OPENAI_API_KEY"))

vector_store = vector_manager.get_store("udaplay_games")

if vector_store:
    test_results = vector_store.get(limit=1)
    print(f"✅ Connected to vector store. Sample docs: {len(test_results['ids'])}")
else:
    print("❌ Could not locate 'udaplay_games' vector store.")
    print("Available collections:", [c.name for c in vector_manager.client.list_collections()])
    print("Please run Part 1 first: python submissions/Udaplay_01_solution_project.py")
    raise RuntimeError("Could not locate 'udaplay_games' vector store. Run Part 1 first.")

# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------
@tool
def retrieve_game(query: str, n_results: int = 3) -> Dict:
    """Search the vector database for game information."""
    try:
        results = vector_store.query(query_texts=[query], n_results=n_results)
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for doc, distance, metadata in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            ):
                similarity = 1 - distance
                formatted_results.append(
                    {
                        "name": metadata["name"],
                        "platform": metadata["platform"],
                        "genre": metadata["genre"],
                        "publisher": metadata["publisher"],
                        "release_year": metadata["release_year"],
                        "description": metadata["description"],
                        "similarity_score": similarity,
                    }
                )
        return {
            "query": query,
            "results": formatted_results,
            "num_results": len(formatted_results),
        }
    except Exception as e:
        return {
            "error": f"Error retrieving game information: {e}",
            "query": query,
            "results": [],
        }


@tool
def evaluate_retrieval(query: str, retrieved_results: str = "") -> Dict:
    """Evaluate the quality of retrieved results using an LLM."""
    evaluator = LLM(model="gpt-4o-mini", temperature=0.1)
    
    # Try to parse retrieved_results if it's a string
    if isinstance(retrieved_results, str):
        try:
            import json
            retrieved_results_dict = json.loads(retrieved_results)
            results_text = (
                "\n\n".join(
                    [
                        f"Game {i+1}: {r['name']}\n"
                        f"Platform: {r['platform']}\n"
                        f"Year: {r['release_year']}\n"
                        f"Genre: {r['genre']}\n"
                        f"Publisher: {r['publisher']}\n"
                        f"Description: {r['description']}\n"
                        f"Relevance Score: {r['similarity_score']:.3f}"
                        for i, r in enumerate(retrieved_results_dict.get("results", []))
                    ]
                )
                if retrieved_results_dict.get("results")
                else "No results found."
            )
        except:
            # If parsing fails, just use the string representation
            results_text = retrieved_results
    else:
        # If it's already a dict, format it normally
        results_text = (
            "\n\n".join(
                [
                    f"Game {i+1}: {r['name']}\n"
                    f"Platform: {r['platform']}\n"
                    f"Year: {r['release_year']}\n"
                    f"Genre: {r['genre']}\n"
                    f"Publisher: {r['publisher']}\n"
                    f"Description: {r['description']}\n"
                    f"Relevance Score: {r['similarity_score']:.3f}"
                    for i, r in enumerate(retrieved_results.get("results", []))
                ]
            )
            if retrieved_results.get("results")
            else "No results found."
        )

    evaluation_prompt = f"""
Evaluate if the following search results adequately answer the user's query.

User Query: "{query}"

Retrieved Results:
{results_text}

Please provide:
1. A quality score from 0-10 (10 being perfect)
2. A brief explanation of your evaluation
3. Whether web search is needed (true/false)

Respond in JSON format:
{{
    "quality_score": <number>,
    "explanation": "<your evaluation>",
    "needs_web_search": <true/false>,
    "missing_information": "<what's missing>"
}}
"""
    try:
        response = evaluator.invoke(evaluation_prompt)
        evaluation = json.loads(response.content)
        # For counting results, handle both string and dict cases
        if isinstance(retrieved_results, str):
            try:
                retrieved_results_dict = json.loads(retrieved_results)
                num_results = len(retrieved_results_dict.get("results", []))
            except:
                num_results = 0
        else:
            num_results = len(retrieved_results.get("results", []))
            
        return {
            "query": query,
            "quality_score": evaluation["quality_score"],
            "explanation": evaluation["explanation"],
            "needs_web_search": evaluation["needs_web_search"],
            "missing_information": evaluation.get("missing_information", ""),
            "num_results_evaluated": num_results,
        }
    except Exception as e:
        return {
            "query": query,
            "quality_score": 0,
            "explanation": f"Error during evaluation: {e}",
            "needs_web_search": True,
            "missing_information": "Unable to evaluate results",
        }


@tool
def game_web_search(query: str) -> Dict:
    """Perform a web search via Tavily API for additional game info."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": tavily_api_key,
        "query": f"{query} video game",
        "search_depth": "advanced",
        "include_answer": True,
        "include_raw_content": False,
        "max_results": 5,
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        formatted_results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "score": r.get("score", 0),
            }
            for r in data.get("results", [])
        ]
        return {
            "query": query,
            "answer": data.get("answer", ""),
            "results": formatted_results,
            "num_results": len(formatted_results),
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Web search error: {e}", "query": query, "results": []}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", "query": query, "results": []}


# ---------------------------------------------------------------------------
# Custom Agent Definition
# ---------------------------------------------------------------------------
class UdaPlayAgent(Agent):
    """Agent that follows RAG → Evaluate → Web Search workflow."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        instructions = (
            "You are UdaPlay, an AI research assistant specializing in video game information.\n\n"
            "Workflow:\n"
            "1. Use retrieve_game to search the internal database.\n"
            "2. Use evaluate_retrieval to assess result quality.\n"
            "3. If results are insufficient, use game_web_search.\n"
            "4. Return comprehensive, cited answers.\n\n"
            "Answering Guidelines:\n"
            "- Always cite sources.\n"
            "- Provide specific game details (platform, year, publisher, etc.).\n"
            "- If sources conflict, mention both and explain.\n"
            "- Maintain conversation context across queries."
        )
        super().__init__(
            model_name=model_name,
            instructions=instructions,
            tools=[retrieve_game, evaluate_retrieval, game_web_search],
            temperature=temperature,
        )

    def invoke(self, query: str, session_id: Optional[str] = None):
        print("=" * 60)
        print(f"Processing query: '{query}'")
        print("=" * 60)
        result = super().invoke(query, session_id)
        final_state = result.get_final_state()
        if final_state and "total_tokens" in final_state:
            print(f"💬 Total tokens used: {final_state['total_tokens']}")
        return result


# ---------------------------------------------------------------------------
# Demo / Test Harness
# ---------------------------------------------------------------------------

def _display_response(run_result):
    final_state = run_result.get_final_state()
    if final_state and "messages" in final_state:
        for msg in reversed(final_state["messages"]):
            if getattr(msg, "content", None) and not hasattr(msg, "tool_call_id"):
                print("\n🤖 Agent Response:\n" + msg.content)
                break


def _run_demo_queries():
    agent = UdaPlayAgent()
    session_id = "demo_session"

    queries = [
        "When was Pokémon Gold and Silver released?",
        "Which one was the first 3D platformer Mario game?",
        "Was Mortal Kombat X released for PlayStation 5?",
        "What other Pokemon games were released around the same time?",
    ]

    for q in queries:
        res = agent.invoke(q, session_id=session_id)
        _display_response(res)

    runs = agent.get_session_runs(session_id)
    total_tokens = sum(r.get_final_state().get("total_tokens", 0) for r in runs if r.get_final_state())
    print(f"\n📊 Processed {len(runs)} queries | Total tokens: {total_tokens}")


if __name__ == "__main__":
    _run_demo_queries() 