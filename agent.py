from langgraph.graph import START, StateGraph, MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
import sqlite3
from pathlib import Path
from tools import tools

load_dotenv()

Path('data').mkdir(exist_ok=True)

DEFAULT_MODEL = "mistralai:mistral-small-latest"

ALLOWED_MODELS = {
    'mistralai:mistral-small-latest',
    'mistralai:mistral-medium-latest',
    'mistralai:mistral-large-latest',
    'mistralai:ministral-8b-latest',
    'mistralai:ministral-3b-latest',
    'mistralai:ministral-14b-latest',
    'mistralai:codestral-latest',
}

SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named PrathGPT similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- If the user asks about latest news, current events, recent updates, today's information, current prices, current people, current versions, new releases, or anything time-sensitive, use Tavily Search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and mention that the answer is based on web search results.
- Be clear, helpful, and concise.
"""

# Returns the compiled agent workflow
def build_agent(model_name: str):
    """
    Build one LangGraph agent for a selected Mistral model.
    """

    # Initialize ChatGoogleGenerativeAI
    llm = init_chat_model(model_name)

    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    tool_node = ToolNode(tools) # Executes tool calls

    graph = StateGraph(MessagesState)

    graph.add_node('chat_node', chatbot_node)
    graph.add_node('tools', tool_node)

    graph.add_edge(START, 'chat_node')

    # Conditional Routing:
    # tools_condition inspects state['messages'][-1].
    # If tool_calls exist -> returns 'tools'
    # If no tool_calls -> returns '__end__' (LangGraph's internal END node), completing the turn.
    graph.add_conditional_edges('chat_node', tools_condition) 

    # ReAct Cycle: After a tool executes, return to chat_node so the LLM can interpret results
    graph.add_edge('tools', 'chat_node')

    # check_same_thread=False allows Streamlit's multi-threaded worker runtime to share the DB connection
    conn = sqlite3.connect(database='data/langgraph_checkpoints.sqlite', check_same_thread=False)

    # SqliteSaver persists state snapshots per thread_id to support conversational memory & HITL pause/resume
    checkpoint = SqliteSaver(conn)

    return graph.compile(checkpointer=checkpoint)

_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """

    if model_name not in _AGENT_CACHE:
        _AGENT_CACHE[model_name] = build_agent(model_name)

    return _AGENT_CACHE[model_name]