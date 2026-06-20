from google import genai
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embeddings(text):
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    return result.embeddings[0].values

e1 = get_embeddings("I love cats")
e2 = get_embeddings("I adore kittens")
e3 = get_embeddings("Stock market crashed today")

print("cats vs kittens:", cosine_similarity(e1, e2))
print("cats vs stocks:", cosine_similarity(e1, e3))