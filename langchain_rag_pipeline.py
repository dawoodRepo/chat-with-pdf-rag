from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from google import genai
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# step # 1
# load the document
loader = PyMuPDFLoader("story.pdf")
documents = loader.load()

# step # 2
# split the document and make chunks of it
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
# filter out empty pages from chunks
chunks = [chunk for chunk in chunks if chunk.page_content.strip()]



# --------------CUSTOM CLASS FOR EMBEDDINGS-----------------
# whenever LangChain doesn't support a model officially, 
# you can always wrap it yourself like this!
# So it's like a bridge — LangChain thinks it's talking to its own embedding class, 
# but under the hood it's calling Gemini's API directly.

class GeminiEmbeddings(Embeddings):
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.client.models.embed_content(model="gemini-embedding-2", contents=text).embeddings[0].values for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return self.client.models.embed_content(model="gemini-embedding-2", contents=text).embeddings[0].values
# ----------------------------------------------------------


# step # 3
# create embeddings for all the chunks using our custom class
embeddings = GeminiEmbeddings()


# step # 4
# store these embeddings with its correspinding chunks in chroma database
if os.path.exists("./chroma_langchain"):
    vectorstore = Chroma(
        persist_directory="./chroma_langchain",
        embedding_function=embeddings,
        collection_name="story_langchain"
    )
else:
    # only chunk when needed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_langchain",
        collection_name="story_langchain"
    )


# step # 5
# print total chunks
# print(f"stored {vectorstore._collection.count()} chunks!")


# step # 6
# now user have a question regarding the pdf
question = "who's the author of this story?"


# step # 7
# it converts the user question into embeddings 
# and then search the vector store for relevant embeddings
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})    # k=3 means fetch the 3 most relevant chunks using cosine similarity
relevant_chunks = retriever.invoke(question)
# print("relevent chunks:")
# for chunk in relevant_chunks:
#     print(chunk.page_content)
#     print("---")


# step # 8
# extract the text from the returned chunks and join them into one context string
context = "\n\n".join([doc.page_content for doc in relevant_chunks])


# step # 9
# creating the Groq client using the OpenAI SDK
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


# step # 10
# Now send the context and question to the LLM and print the answer:
response = groq_client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {"role": "system", "content": f"Answer the question based only on this context:\n\n{context}"},
        {"role": "user", "content": question}
    ]
)
print(response.choices[0].message.content)