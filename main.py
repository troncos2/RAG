import os
import time
import wikipedia 
from dotenv import load_dotenv 

load_dotenv()  
my_user_agent = os.getenv("USER_AGENT")
os.environ["USER_AGENT"] = my_user_agent
wikipedia.set_user_agent(my_user_agent)

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = InMemoryVectorStore(embeddings)

print("Please wait while Lucy wakes up...")
for attempt in range(3):
    try:
        docs = WikipediaLoader(query="human evolution", load_max_docs=10).load()       
        break  
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        if attempt < 2: time.sleep(5)
        else: raise  


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,   
    chunk_overlap = 200, 
    add_start_index = True,
)
all_splits = text_splitter.split_documents(docs)

vector_store.add_documents(documents=all_splits)

time.sleep(10)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

def format_docs(docs):
    return "\n\n".join(
        [f"Source: {d.metadata.get('title')} | URL: {d.metadata.get('source')}\n{d.page_content}" 
         for d in docs]
    )

template = """You are Lucy, a helpful AI specializing in human evolution. 
Use the provided context to answer the question accurately. 

If the user greets you, reply warmly and offer help.
Do not include a greeting in every answer.
If the answer isn't in the context, politely state that you don't know.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_pipe = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print("\n" + "="*60)
print("TOPIC: Human Evolution RAG System")
print("="*60 + "\n")

print("Hello, I'm Lucy! Ask me anything about hominids or evolution!")
print("If you wish to end our chat, type 'exit' or 'quit' to end.")

while True:
    user_query = input("user: ")
    if user_query.lower() in ['quit', 'exit']:
        print("\nLucy: Goodbye! Keep exploring!\n")
        break
    if not user_query.strip():
        continue
        
    print("\nLucy is searching and thinking...")
    try:
        response = rag_pipe.invoke(user_query)
        print("\n" + "-" * 60)
        print(f"Lucy:\n{response}")
        print("-" * 60 + "\n")
    except Exception as e:
        print(f"\n An error occurred: {e}")