import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

index = faiss.read_index("agri.index")
with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(question, k=3):
    q_emb = embedder.encode([question])
    _, ids = index.search(np.array(q_emb), k)
    return [all_chunks[i] for i in ids[0]]

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

instruction = (
    "You are a helpful assistant that answers questions strictly based on the provided context. "
    "Extract the most relevant pest control methods for rice from the provided data. "
    "Summarize in 2-3 short bullet points only. Be concise. No intro or explanation needed and use phrases if needed. "
    "If the context does not contain sufficient information, respond with general, agricultural guidance from your own knowledge, give concrete response instead of asking to user to refer other sources. "
    "If the question is not related to agriculture or farming, respond with 'I am sorry, I don't have the information to answer that question.' "
    "Don't include any aestrisk marks or other markdown formatting in your response."
)


def answer_question(question: str) -> str:
    context = "\n\n".join(retrieve(question))
    prompt = f"""
{instruction}

Context:
{context}

Question:
{question}

Answer:
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
        ),
    )
    return response.text.replace("*", "").strip()


if __name__ == "__main__":
    while True:
        user_q = input("Ask a question (or 'exit' to quit the application): ").strip()
        if not user_q or user_q.lower() == "exit":
            break
        print("\n" + answer_question(user_q) + "\n")