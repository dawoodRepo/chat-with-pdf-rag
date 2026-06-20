import fitz
from google import genai
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import os

load_dotenv()

# ----------- load and chunk the PDF -----------
doc = fitz.open("story.pdf")
text = ""
for page in doc:
    text += page.get_text()

def split_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = split_chunks(text)

# ----------- embedding function -----------
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text):
    result = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    return result.embeddings[0].values

# ----------- store in chroma (only embed once) -----------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="story")

if collection.count() == 0:
    print("embedding chunks for the first time...")
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(ids=[str(i)], embeddings=[embedding], documents=[chunk])
        print(f"embedded chunk {i+1}/{len(chunks)}")
else:
    print(f"loaded {collection.count()} chunks from disk!")

# ----------- search chroma -----------
question = "what happens at the end of the lottery?"
question_embedding = get_embedding(question)

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

# ----------- send to LLM -----------
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

context = "\n\n".join(results["documents"][0])

response = groq_client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {"role": "system", "content": f"Answer the question based only on this context:\n\n{context}"},
        {"role": "user", "content": question}
    ]
)

print(response.choices[0].message.content)