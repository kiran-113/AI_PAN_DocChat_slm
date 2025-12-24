import os, re, io, hashlib
import shutil
import logging
import subprocess
from PIL import Image
import streamlit as st
from datetime import datetime
import fitz  # PyMuPDF
from rapidfuzz import process, fuzz

# ==================================================
# ✅ PAGE-WISE EXTRACTION HELPERS
# ==================================================
def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text("text")
        pages.append({
            "page_number": page_number + 1,
            "text": text
        })
    return pages

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

PAN_REGEX = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"

def find_pan_pagewise(pdf_path):
    results = []
    pages = extract_pages(pdf_path)

    for page in pages:
        page_no = page["page_number"]
        text = page["text"]
        chunks = chunk_text(text)

        for chunk in chunks:
            matches = re.findall(PAN_REGEX, chunk)
            for pan in matches:
                results.append({
                    "pan": pan,
                    "page": page_no,
                    "snippet": chunk.strip()[:300]
                })
    return results

# ==================================================
# LANGCHAIN / VECTOR IMPORTS
# ==================================================
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit

from rag import RAGSystem

# ==================================================
# CONFIG
# ==================================================
RESET_DB_PASSWORD = "admin123"
VECTOR_DB_PATH = "./PDF_ChromaDB"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100
NAME_REGEX = r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3}\b"

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

def looks_like_pan(text: str) -> bool:
    text = text.strip().upper()
    return text.isalnum() and 8 <= len(text) <= 10

def is_valid_pan(pan: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan))

# ==================================================
# 🔎 PAN SEARCH — GROUPED, PAGE-AWARE
# ==================================================
def find_pan_in_documents(pan: str):
    if not vector_db_exists():
        return {}

    db = Chroma(
        collection_name="pdf_content",
        persist_directory=VECTOR_DB_PATH,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"),
    )

    data = db.get(include=["documents", "metadatas"])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    pan = pan.upper()
    results = {}

    for text, meta in zip(docs, metas):
        if not meta or "source" not in meta:
            continue

        if pan in text.upper():
            file = meta["source"]
            page = meta.get("page", "Unknown")

            idx = text.upper().find(pan)
            start = max(0, idx - 40)
            end = min(len(text), idx + 40)
            snippet = text[start:end]

            snippet = re.sub(pan, f"**{pan}**", snippet, flags=re.IGNORECASE)

            if file not in results:
                results[file] = {"pages": set(), "snippets": []}

            results[file]["pages"].add(page)

            if len(results[file]["snippets"]) < 2:
                results[file]["snippets"].append(snippet.strip())

    return results

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
# UI & CHAT STYLING (UPDATED FOR YOUR IMAGE)
# ==================================================
image = Image.open("imgs/ChatPDF3.png")
st.set_page_config(page_title="PAN & DocChat AI", page_icon=image, layout="wide")

st.markdown(
    """
    <style>
    /* Chat layout styling */
    [data-testid="stChatMessage"] {
        display: flex;
        width: 100%;
        margin-bottom: 10px;
    }

    /* USER MESSAGE: Align Right */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
    }

    /* USER CONTENT BUBBLE */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:nth-child(2) {
        background-color: #DCF8C6 !important; /* Soft Green */
        color: #000000 !important;
        border-radius: 15px 15px 0px 15px;
        padding: 10px 15px;
        margin-right: 10px;
        max-width: 70%;
        flex-grow: 0;
    }

    /* ASSISTANT MESSAGE: Align Left (Default) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        flex-direction: row;
    }

    /* ASSISTANT CONTENT BUBBLE */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div:nth-child(2) {
        background-color: #F0F2F6 !important; /* Soft Grey */
        color: #000000 !important;
        border-radius: 15px 15px 15px 0px;
        padding: 10px 15px;
        margin-left: 10px;
        max-width: 70%;
        flex-grow: 0;
    }

    /* Hide the default background of chat containers to let bubbles show */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader("💬 PAN & DocChat AI")

_, indexed_files, chunk_count, last_indexed = get_db_metadata()
pdf_count = len(indexed_files)

st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.caption(f"📄 PDFs indexed: {pdf_count}")
c2.caption(f"🧩 Total chunks: {chunk_count}")
c3.caption(f"🕒 Last indexed: {last_indexed}" if last_indexed else "🕒 Last indexed: —")
st.markdown("---")

# ==================================================
# SIDEBAR
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

available_models = get_available_models()

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
    hashes, files, _, _ = get_db_metadata()

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
            with open(path, "wb") as f:
                f.write(data)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            )

            now = datetime.now().isoformat()
            pdf_doc = fitz.open(path)

            for page_no, page in enumerate(pdf_doc, start=1):
                text = page.get_text("text")
                for ch in splitter.split_text(text):
                    docs.append(
                        Document(
                            page_content=ch,
                            metadata={
                                "source": pdf.name,
                                "hash": h,
                                "indexed_at": now,
                                "page": page_no,
                                "names": list(set(found_names)),
                            },
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
# RAG SYSTEM
# ==================================================
rag = RAGSystem("pdf_content", VECTOR_DB_PATH, n_results)

# ==================================================
# CHAT INTERFACE
# ==================================================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("Ask me...")
if query:
    q = query.strip().upper()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ---------- PAN FLOW ----------
    if looks_like_pan(q):
        if not is_valid_pan(q):
            resp = "❌ Invalid PAN format. Expected **ABCDE1234F**."
        else:
            matches = find_pan_in_documents(q)
            if not matches:
                resp = f"❌ PAN **{q}** not found in indexed documents."
            else:
                lines = [f"✅ PAN **{q}** found in **{len(matches)} document(s)**:\n"]
                for file, info in matches.items():
                    pages = ", ".join(str(p) for p in sorted(info["pages"]))
                    lines.append(f"📄 **{file}**")
                    lines.append(f"📍 Pages: **{pages}**\n")
                resp = "\n".join(lines)

    # ---------- NORMAL RAG ----------
    else:
        with st.spinner("Thinking..."):
            ans, _, _, _, _ = rag.generate_response(query, selected_model)
            resp = remove_tags(ans)

    st.session_state.messages.append({"role": "assistant", "content": resp})
    with st.chat_message("assistant"):
        st.markdown(resp)