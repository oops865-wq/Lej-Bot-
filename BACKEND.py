from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun



# Loading the models with API key
load_dotenv()
# Note: Ensure gpt-4.1-mini is available in your region/tier
llm = ChatOpenAI(model="gpt-4.1-mini")
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Documents loader - Ensure this file exists in your directory!
try:
    loader = PyPDFLoader("intro-to-ml.pdf")
    docs = loader.load()
    # splitting chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    # Creating vector store (Corrected order: documents first, then embeddings)
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})
except Exception as e:
    print(f"Warning: PDF loading failed: {e}. RAG tool might not work.")

# *********************** TOOLS SECTION ********************************************

search_tool = DuckDuckGoSearchRun(region='us-en')

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform basic arithmetic operations on two numbers.
    Supported operations: add, subtract, multiply, divide
    """
    try:
        result = None
        if operation == "add":
            result = first_num + second_num
        elif operation == "subtract":
            result = first_num - second_num
        elif operation == 'multiply':
            result = first_num * second_num
        elif operation == "divide":
            if second_num == 0:
                return {'error': "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"Error": f"Unsupported operation {operation}"}
        
        return {'first_num': first_num, 'second_num': second_num, 'operation': operation, 'result': result}
    except Exception as e:
        return {"Error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA').
    """
    # Note: Replace with your actual Alpha Vantage API key
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=YOUR_API_KEY"
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query: str):
    """
    Retrieve relevant information from the uploaded PDF for the current context.
    """
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        'query': query,
        'context': context, # Fixed typo
        'metadata': metadata
    }

tools = [calculator, get_stock_price, search_tool, rag_tool]
# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# *********************** GRAPH SETUP ********************************************

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    # Use llm_with_tools so the model can actually decide to use tools
    response = llm_with_tools.invoke(messages) 
    return {'messages': [response]}

tools_node = ToolNode(tools)

# Creating database for persistence
conn = sqlite3.connect(database='Chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

# Define the flow
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tools_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

# Compile the graph
Chatbot = graph.compile(checkpointer=checkpointer)

def retrive_all_threads():
    all_threads = set()
    # Iterate through checkpoints to find unique thread IDs
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)