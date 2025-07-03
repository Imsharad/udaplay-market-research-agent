# Project: Market Research Agent

## RAG

| Criteria | Submission Requirements |
|----------|------------------------|
| Prepare and process a local dataset of video game information for use in a vector database and RAG pipeline | • The submission includes the notebook (Udaplay_01_solution_project.ipynb) that loads, processes, and formats the provided game JSON files.<br>• The processed data is added to a persistent vector database (e.g., ChromaDB) with appropriate embeddings.<br>• The notebook or script demonstrates that the vector database can be queried for semantic search. |

## Agent Development

| Criteria | Submission Requirements |
|----------|------------------------|
| Implement agent tools for internal retrieval, evaluation, and web search fallback. | • The submission includes at least three tools:<br>&nbsp;&nbsp;- A tool to retrieve game information from the vector database.<br>&nbsp;&nbsp;- A tool to evaluate the quality of retrieved results.<br>&nbsp;&nbsp;- A tool to perform web search using an API (e.g., Tavily).<br>• Each tool is implemented as a function/class and is integrated into the agent workflow.<br>• The agent:<br>&nbsp;&nbsp;- first attempts to answer using internal knowledge,<br>&nbsp;&nbsp;- evaluates the result,<br>&nbsp;&nbsp;- and falls back to web search if needed. |
| Build a stateful agent that manages conversation and tool usage. | • The agent is implemented as a class or function that maintains conversation state.<br>• The agent can handle multiple queries in a session, remembering previous context.<br>• The agent's workflow is implemented as a state machine or similar abstraction.<br>• The agent produces clear, structured, and well-cited answers. |
| Demonstrate and report on the agent's performance with example queries. | • The submission includes the notebook (Udaplay_02_solution_project.ipynb) that runs the agent on at least three example queries (e.g., about game release dates, platforms, or publishers).<br>• The output for each query includes the agent's reasoning, tool usage, and final answer.<br>• The report includes at least the response with citation, if any |
