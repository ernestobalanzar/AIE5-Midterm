from typing import TypedDict, Annotated, List
from typing_extensions import List, TypedDict

from dotenv import load_dotenv
import chainlit as cl
import operator

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader, PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

openai_chat_model = ChatOpenAI(model="gpt-4o-mini")

path = "data/"

# Load HTML files
html_loader = DirectoryLoader(path, glob="*.html", loader_cls=BSHTMLLoader)
html_docs = html_loader.load()

# Load PDF files
pdf_loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
pdf_docs = pdf_loader.load()

# Combine both document lists
docs = html_docs + pdf_docs

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 850,
    chunk_overlap  = 50,
    length_function = len
)
split_documents =  text_splitter.split_documents(docs)


# Load your fine-tuned model

finetune_embeddings = HuggingFaceEmbeddings(model_name="finetuned_caregiver_ft")

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="ai_across_years2",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="ai_across_years2",
    embedding=finetune_embeddings,
)

_ = vector_store.add_documents(documents=split_documents)


retriever = vector_store.as_retriever(search_kwargs={"k": 5})


def retrieve(state):
    retrieved_docs = retriever.invoke(state["question"])
    return {"context": retrieved_docs}


#####

RAG_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say "I don't know, would you like to talk to a care coach?", don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""

rag_prompt = PromptTemplate.from_template(RAG_template) 

def generate(state):
    docs_content = "\n\n".join([doc.page_content for doc in state["context"]])
    messages = rag_prompt.format_prompt(context=docs_content, question=state["question"])
    respose = openai_chat_model.invoke(messages)
    return {"response": respose.content}

class State(TypedDict):
  question: str
  context: List[Document]
  response: str

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve");
graph = graph_builder.compile();

@tool
def ai_rag_tool(question: str) -> str:
    """Answer questions about ALS based on the retrieved documents. Input should be a fully formed question."""
    response = graph.invoke({"question": question})
    return{
        "messages": [HumanMessage(content=response["response"])],
        "context": response["context"],
    }

tavily_tool = TavilySearchResults(max_results=5)

tool_belt = [
    tavily_tool,
    ai_rag_tool,
]

model = ChatOpenAI(model="gpt-4o", temperature=0)

model = model.bind_tools(tool_belt)

class AgentState(TypedDict):
  messages: Annotated[list, add_messages]

def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages, config={"tool_choice": "auto"})  # Ensure it knows it can use tools
    return {"messages": [response]}

tool_node = ToolNode(tool_belt)

uncompiled_graph = StateGraph(AgentState)

uncompiled_graph.add_node("agent", call_model)
uncompiled_graph.add_node("action", tool_node)

uncompiled_graph.set_entry_point("agent")

def should_continue(state):
    last_message = state["messages"][-1]
    print(f"Checking if model wants to call a tool: {last_message}")  # Debugging

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"Model wants to call a tool: {last_message.tool_calls}")
        return "action"

    print("No tool calls detected, ending execution.")
    return END


### Add conditional edges to the graph
uncompiled_graph.add_conditional_edges(
    "agent",
    should_continue
)

uncompiled_graph.add_edge("action", "agent")
compiled_graph = uncompiled_graph.compile()




@cl.on_chat_start
async def start():
  cl.user_session.set("compiled_graph", compiled_graph)

@cl.on_message
async def handle(message: cl.Message):
    """Handle user messages, invoke the agent graph, and send responses."""
    
    compiled_graph = cl.user_session.get("compiled_graph")  # Retrieve the stored graph

    # Initialize the state for the agent graph
    state = {"messages": [HumanMessage(content=message.content)]}

    # Invoke the agent graph asynchronously
    response = await compiled_graph.ainvoke(state)

    # Extract the model's response and send it to the user
    final_message = response["messages"][-1].content  # Extract last message content

    await cl.Message(content=final_message).send()