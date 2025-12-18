import os, re, io, hashlib
import shutil
import logging
import subprocess
from PIL import Image
import streamlit as st
from datetime import datetime

# -------------------------------
import pymupdf4llm
# -------------------------------
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
# -------------------------------
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
# -------------------------------
from rag import RAGSystem

# ==================================================
# CONFIG
# ==================================================
RESET_DB_PASSWORD = "admin123"
VECTOR_DB_PATH = "./PDF_ChromaDB"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100

# ==================================================
# LOGGING
# ==================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ==================================================
# HELPERS
# ==================================================
def vector_db_exists(path=VECTOR_DB_PATH):
    return os.path.exists(path) and any(os.scandir(path))

def get_pdf_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def is_valid_pan(pan: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan))

def looks_like_pan(text: str) -> bool:
    text = text.strip().upper()
    return text.isalnum() and 8 <= len(text) <= 10

def find_pan_in_documents(pan: str):
    if not vector_db_exists():
        return set()

    db = Chroma(
        collection_name="pdf_content",
        persist_directory=VECTOR_DB_PATH,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"),
    )

    data = db.get(include=["documents", "metadatas"])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    found = set()
    for d, m in zip(docs, metas):
        if pan in d.upper() and m and "source" in m:
            found.add(m["source"])
    return found

def get_db_metadata():
    if not vector_db_exists():
        return set(), [], 0, None

    db = Chroma(
        collection_name="pdf_content",
        persist_directory=VECTOR_DB_PATH,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"),
    )

    metas = db.get(include=["metadatas"]).get("metadatas", [])
    hashes, files, times = set(), set(), []

    for m in metas:
        if not m:
            continue
        hashes.add(m.get("hash"))
        files.add(m.get("source"))
        if m.get("indexed_at"):
            times.append(m["indexed_at"])

    last = (
        datetime.fromisoformat(max(times)).strftime("%Y-%m-%d %H:%M:%S")
        if times else None
    )

    return hashes, sorted(files), len(metas), last

def remove_tags(text):
    return re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()

# ==================================================
# PDF EXPORT
# ==================================================
def generate_pdf():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    y = h - 40

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(w / 2, y, "Conversation History")
    y -= 40
    c.setFont("Helvetica", 12)

    for msg in st.session_state.messages:
        c.drawString(40, y, f"{msg['role'].title()}: {msg['content']}")
        y -= 18
        if y < 40:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = h - 40

    c.save()
    buf.seek(0)
    return buf

# ==================================================
# MODELS
# ==================================================
def get_available_models():
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return [
            l.split()[0]
            for l in r.stdout.splitlines()
            if l and "embed" not in l.lower() and "NAME" not in l
        ]
    except:
        return []

# ==================================================
# UI HEADER & STATS (Aligned with provided image)
# ==================================================
image = Image.open("imgs/ChatPDF3.png")
st.set_page_config(page_title="PAN & DocChat AI", page_icon=image)
st.subheader("💬 PAN & DocChat AI")

# Footer/Stats logic positioned at the top
_, indexed_files, chunk_count, last_indexed = get_db_metadata()
pdf_count = len(indexed_files)

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📄 PDFs indexed: {pdf_count}")
with col2:
    st.caption(f"🧩 Total chunks: {chunk_count}")
with col3:
    st.caption(f"🕒 Last indexed: {last_indexed}" if last_indexed else "🕒 Last indexed: —")
st.markdown("---")

available_models = get_available_models()

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.image(image)
    st.header("📄 Upload PDFs")

    if vector_db_exists():
        st.success("📚 Existing DB loaded")
    else:
        st.info("📭 No DB found")

    pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# ==================================================
# SESSION STATE
# ==================================================
st.session_state.setdefault("messages", [])
st.session_state.setdefault("processing_complete", vector_db_exists())

# ==================================================
# SIDEBAR CONTROLS
# ==================================================
with st.sidebar:
    hashes, files, chunks, last_time = get_db_metadata()

    if pdfs and st.button("▶ Start Processing"):
        os.makedirs("./tmp", exist_ok=True)
        docs, skipped = [], []

        for pdf in pdfs:
            data = pdf.getbuffer().tobytes()
            h = get_pdf_hash(data)

            if h in hashes:
                skipped.append(pdf.name)
                continue

            path = f"./tmp/{pdf.name}"
            open(path, "wb").write(data)

            text = pymupdf4llm.to_markdown(path)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            )

            now = datetime.now().isoformat()
            for ch in splitter.split_text(text):
                docs.append(
                    Document(
                        page_content=ch,
                        metadata={"source": pdf.name, "hash": h, "indexed_at": now},
                    )
                )

        if docs:
            Chroma.from_documents(
                docs,
                OllamaEmbeddings(model="nomic-embed-text:latest"),
                collection_name="pdf_content",
                persist_directory=VECTOR_DB_PATH,
            )
            st.success(f"Indexed {len(docs)} chunks")
            st.rerun()

        if skipped:
            st.warning(f"Skipped duplicates: {', '.join(skipped)}")

    st.markdown("---")
    st.subheader("📂 Indexed PDFs")
    if files:
        st.selectbox("Documents", files)
    else:
        st.caption("None yet")

    st.markdown("---")
    pwd = st.text_input("Reset DB Password", type="password")
    if st.button("Reset Vector DB"):
        if pwd == RESET_DB_PASSWORD:
            RAGSystem("pdf_content").delete_collection()
            st.session_state.messages.clear()
            st.success("DB reset")
            st.rerun()
        else:
            st.error("Wrong password")

    st.markdown("---")
    selected_model = st.selectbox("Model", available_models)
    n_results = st.slider("Retrieved chunks", 1, 15, 5)

# ==================================================
# RAG
# ==================================================
rag = RAGSystem("pdf_content", VECTOR_DB_PATH, n_results)

# ==================================================
# CHAT UI WITH CUSTOM ALIGNMENT CSS
# ==================================================
st.markdown("""
    <style>
        /* Container for each chat message */
        [data-testid="stChatMessage"] {
            display: flex;
            margin-bottom: 10px;
            width: 100%;
        }

        /* USER MESSAGE: Align to Right */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse;
            text-align: right;
        }

        /* ASSISTANT MESSAGE: Align to Left (Default) */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            flex-direction: row;
            text-align: left;
        }

        /* Ensure message bubbles don't span full width */
        [data-testid="stChatMessageContent"] {
            width: fit-content;
            max-width: 85%;
        }
    </style>
""", unsafe_allow_html=True)

# Loop through messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("Ask me...")
if query:
    q = query.strip().upper()
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ---------- PAN LOGIC ----------
    if looks_like_pan(q) and not is_valid_pan(q):
        resp = "❌ Invalid PAN format. Expected `ABCDE1234F`."

    elif is_valid_pan(q):
        found = find_pan_in_documents(q)
        resp = (
            f"❌ PAN **{q}** not found."
            if not found
            else f"✅ PAN **{q}** found in: **{', '.join(found)}**"
        )

    # ---------- RAG ----------
    else:
        with st.spinner("Thinking..."):
            ans, t, _, _, _ = rag.generate_response(query, selected_model)
            resp = remove_tags(ans)

    st.session_state.messages.append({"role": "assistant", "content": resp})
    with st.chat_message("assistant"):
        st.markdown(resp)
