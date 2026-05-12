# Main file for implementation of RAG system

# Wikipedia RAG implementation

# Set the environment to start logging traces in LogSmith
import os
import time
import wikipedia 
from dotenv import load_dotenv 
from typing import Literal
from pydantic import BaseModel, Field

load_dotenv()  
my_user_agent = os.getenv("USER_AGENT")
os.environ["USER_AGENT"] = my_user_agent
wikipedia.set_user_agent(my_user_agent)

from langchain.chat_models import init_chat_model 
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_core.vectorstores import InMemoryVectorStore 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain.tools import tool 
from langchain.agents import create_agent 
from langchain_community.document_loaders import WikipediaLoader
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# COMPONENTS
model = init_chat_model("google_genai:gemini-2.5-flash-lite")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = InMemoryVectorStore(embeddings)

topic = "human evolution"

# Wikipedia loader
# Loader attempts to connect to wiki up to 3 times if the first attempt fails
for attempt in range(3):  # try up to 3 times
    try:
        #user agent so wikipedia doesn't block the request
        docs = WikipediaLoader(query=topic, load_max_docs=5).load() # grading, adding more topics/guardrails, url where the data comes from (relevancy, accuracy, reliability)       
        break  # success, exit the loop
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        if attempt < 2:
            print("Retrying in 5 seconds...")
            time.sleep(5)
        else:
            raise  # all 3 attempts failed, show the error


print(f"Loaded {len(docs)} documents")
print(f"Total characters: {sum(len(doc.page_content) for doc in docs)}")
print(f"First document title: {docs[0].metadata.get('title', 'N/A')}")
print(docs[0].page_content[:500])   # prints the firsrt 500 characters (?)

# Splitting Documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,  # chunk size in characters
    chunk_overlap = 200, # chunk overlap in characters
    add_start_index = True, # track index in original document
)
all_splits = text_splitter.split_documents(docs)

# Embded and store all the documents
document_ids = vector_store.add_documents(documents=all_splits)
print(f"\nStored {len(document_ids)} chunks in vector store.")

#
# RAG AGENT 
#

# tool for the agent
@tool #(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information Wikipedia articles about human evolution."""
    retrieved_docs = vector_store.similarity_search(query, k=3)

    return "\n\n".join(
        f"Source: {doc.metadata.get('title', 'Unknown')}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

retriever_tool = retrieve_context

#
# LANGCHAIN GRADING SYSTEM
#

# grading schema
class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

# topic guardrail
class GradeQuestion(BaseModel):
    """Check if the question is on topic."""
    binary_score: str = Field(
        description="'yes' if question is about human evolution, 'no' if off-topic"
    )

TOPIC_GUARD_PROMPT = (
    "You are checking whether a user question is related to human evolution, "
    "hominids, prehistoric humans, or closely related topics. \n"
    "Here is the user question: {question} \n"
    "Give a binary score 'yes' if it is on topic, or 'no' if it is off-topic "
    "(e.g. about animal evolution in general, cooking, sports, etc.)."
)

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n"
    "Here is the retrieved document:\n\n{context}\n\n"
    "Here is the user question: {question}\n"
    "If the document contains keyword(s) or semantic meaning related to the user question, "
    "grade it as relevant.\n"
    "Give a binary score 'yes' if relevant, or 'no' if not relevant."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent.\n"
    "Here is the initial question:\n"
    "---\n{question}\n---\n"
    "This question is about human evolution. Formulate an improved, more specific question:"
)

GENERATE_PROMPT = (
    "You are an assistant specializing in human evolution. "
    "Use the following retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question}\n"
    "Context: {context}"
)

#
# Graph Nodes
#

def generate_query_or_respond(state: MessagesState):
    """Decide to retrieve or respond directly. Also enforces topic guardrail."""
    question = state["messages"][-1].content

    # Topic guardrail (check if question is on topic before doing anything)
    prompt = TOPIC_GUARD_PROMPT.format(question=question)
    grade = (
        model
        .with_structured_output(GradeQuestion)
        .invoke([{"role": "user", "content": prompt}])
    )

    if grade.binary_score == "no":
        # Respond directly without retrieving
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(content=(
                "I'm only able to answer questions about human evolution, hominids, "
                "and closely related topics. Your question appears to be outside that scope. "
                "Please ask something related to human evolution!"
            ))]
        }

    # On-topic — let the model decide whether to call the retriever tool
    response = model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Check if retrieved documents are actually relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        model
        .with_structured_output(GradeDocuments)
        .invoke([{"role": "user", "content": prompt}])
    )

    if response.binary_score == "yes":
        print("--- Documents are RELEVANT, generating answer ---")
        return "generate_answer"
    else:
        print("--- Documents NOT relevant, rewriting question ---")
        return "rewrite_question"
    
def rewrite_question(state: MessagesState):
    """Rewrite the question to improve retrieval."""
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = model.invoke([{"role": "user", "content": prompt}])
    from langchain_core.messages import HumanMessage
    return {"messages": [HumanMessage(content=response.content)]}


def generate_answer(state: MessagesState):
    """Generate a final answer from retrieved context."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}   

#
# ASSEMBLE THE GRAPH
#

def route_on_tool_calls(state: MessagesState):
    """Route based on whether the model requested a tool call."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "retrieve"
    return END


workflow = StateGraph(MessagesState)

# nodes explicitly named
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    route_on_tool_calls,
    {"retrieve": "retrieve", END: END},
)

workflow.add_conditional_edges("retrieve", grade_documents)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()


#
# RUN THE AGENT WITH THE QUERY
#

def run_query(query: str):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    for chunk in graph.stream({"messages": [{"role": "user", "content": query}]}):
        for node, update in chunk.items():
            print(f"\n[Node: {node}]")
            update["messages"][-1].pretty_print()

# On-topic test
run_query("Where do most hominids originate, and what evidence supports that?")

# Off-topic test — should be rejected by guardrail
run_query("How do dolphins evolve their echolocation?")
