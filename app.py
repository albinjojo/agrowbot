from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import faiss
import numpy as np
import os
import uuid

from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# ------------------------------------------------------------------
# App & config
# ------------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
CORS(app)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------------------------------------
# Session storage (in-memory)
# ------------------------------------------------------------------
# { session_id: [ {role, content}, ... ] }
chat_sessions = {}
MAX_TURNS = 6  # user+assistant pairs

# ------------------------------------------------------------------
# Load RAG assets (UNCHANGED)
# ------------------------------------------------------------------
index = faiss.read_index("agri.index")

with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------------------------------------
# System instruction (UNCHANGED)
# ------------------------------------------------------------------
instruction = (
    "You are a helpful assistant that answers questions primarily based on the "
    "provided context. Prefer Kerala-specific agricultural practices, pest control "
    "methods, crop varieties, and recommendations whenever they are available in the "
    "context. If Kerala-specific information is present, do not use general or global "
    "practices. If the user greets (e.g., hello, hi), respond with a short greeting and "
    "ask how you can help with agriculture or farming. Summarize in 2-3 short points only. "
    "Be concise. No intro or explanation needed and use phrases if needed. "
    "If the context does not contain sufficient specific information, respond with "
    "general agricultural guidance if the question is related to agriculture or farming. "
    "Assume the end user is a farmer and prioritize clear, practical, field-ready "
    "guidance over general or theoretical explanations. "
    "If the question is not related to agriculture or farming at all, respond with "
    "\"I am sorry, I don't have the information to answer that question.\" "
    "Do not include asterisk marks or markdown formatting."
)

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def retrieve(question: str, k: int = 4):
    """Retrieve top-k relevant chunks using FAISS."""
    q_emb = embedder.encode([question])
    _, ids = index.search(np.array(q_emb), k)
    return [all_chunks[i] for i in ids[0]]


def get_or_create_session(session_id: str | None) -> str:
    """Create a new session if session_id is None or unknown."""
    if not session_id or session_id not in chat_sessions:
        session_id = f"thread_{uuid.uuid4().hex}"
        chat_sessions[session_id] = []
    return session_id


def get_trimmed_history(session_id: str):
    """Return last MAX_TURNS user+assistant messages."""
    history = chat_sessions.get(session_id, [])
    return history[-MAX_TURNS * 2:]


def append_to_history(session_id: str, role: str, content: str):
    """Append message and trim history safely."""
    history = chat_sessions.setdefault(session_id, [])
    history.append({"role": role, "content": content})
    if len(history) > MAX_TURNS * 2:
        del history[:-MAX_TURNS * 2]

# ------------------------------------------------------------------
# API endpoint
# ------------------------------------------------------------------
@app.route("/ask", methods=["POST"])
def ask():
    # Safe JSON parsing (browser-proof)
    data = request.get_json(silent=True) or {}

    question = data.get("question", "").strip()
    session_id = data.get("session_id")

    if not question:
        return jsonify({"error": "Question is required."}), 400

    # Session handling
    session_id = get_or_create_session(session_id)

    # RAG retrieval (UNCHANGED)
    context = "\n\n".join(retrieve(question))

    # Build messages
    messages = [{"role": "system", "content": instruction}]
    messages.extend(get_trimmed_history(session_id))
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{question}"
    })

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=messages,
            temperature=0.4,
        )

        answer = response.output_text.strip()

    except Exception as exc:
        return jsonify({
            "error": "Failed to get response from language model.",
            "details": str(exc),
            "session_id": session_id
        }), 502

    # Save conversation
    append_to_history(session_id, "user", question)
    append_to_history(session_id, "assistant", answer)

    return jsonify({
        "answer": answer,
        "session_id": session_id
    })


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
