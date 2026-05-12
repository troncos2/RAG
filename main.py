# Main file for implementation of RAG system

# MPTL Blog RAG practice - Changes

# Set the environment to start logging traces in LogSmith
import getpass
import os
import bs4  # Loading documents
from langchain.chat_models import init_chat_model # chat model
from langchain_google_genai import GoogleGenerativeAIEmbeddings   # Embeddings model
from langchain_core.vectorstores import InMemoryVectorStore # Vector store model
from langchain_community.document_loaders import WebBaseLoader # loading documents
from langchain_text_splitters import RecursiveCharacterTextSplitter # text splitter


os.environ["LANGSMITH_TRACING"] = "true" 
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()


# Get the componenets
# select chat model (Gemini)
os.environ["GOOGLE_API_KEY"] = "..." # <<<<< HERE 

model = init_chat_model("google_genai:gemini-2.5-flash-lite")


# select embeddings model
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# select a vector store
vector_store = InMemoryVectorStore(embeddings)


# Loading documents
# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))   # html tags
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),    # blog link
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")
# put in LangSmith password when running

print(docs[0].page_content[:500])   # prints the firsrt 500 characters (?)

# Splitting Documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,  # chunk size in characters
    chunk_overlap = 200, # chunk overlap in characters
    add_start_index = True, # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

# Embded and store all the documents
document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])
