import os, re, io, hashlib
import shutil
import logging
import subprocess
from PIL import Image
import streamlit as st
from datetime import datetime
import fitz  # PyMuPDF
from rapidfuzz import process, fuzz
import json

# ==================================================
# 1. INITIALIZE SESSION STATE (MUST BE FIRST)
# ==================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_name_choices" not in st.session_state:
    st.session_state.pending_name_choices = None

# ==================================================
# HELPERS & EXTRACTION
# ==================================================
PAN_REGEX = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"

def looks_like_pan(text: str) -> bool:
    text = text.strip().upper()
    return bool(re.fullmatch(PAN_REGEX, text))

def get_pdf_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def vector_db_exists(path="./PDF_ChromaDB"):
    return os.path.exists(path) and any(os.scandir(path))

# ==================================================
# LANGCHAIN / VECTOR IMPORTS
# ==================================================
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rag import RAGSystem


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PAN_DEBUG")
    #logger.info(f"[INTENT] is_pan_search={intent.get('is_pan_search')} | person_name={intent.get('person_name')}")

# ==================================================
# CONFIG
# ==================================================
RESET_DB_PASSWORD = "admin123"
VECTOR_DB_PATH = "./PDF_ChromaDB"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100

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

def find_candidate_names(query_name: str):
    db = Chroma(
        collection_name="pdf_content",
        persist_directory=VECTOR_DB_PATH,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"),
    )

    data = db.get(include=["documents"])
    names = set()

    query_upper = query_name.upper()

    for text in data["documents"]:
        # Extract ALL possible full names first
        matches = re.findall(
             r"(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s(?:[A-Z][a-z]+|[A-Z]{2,})){1,3}",
            text
        )

        for m in matches:
        # Split multiline blocks into individual lines
            for line in m.splitlines():
                clean = line.strip().upper()

                # Must contain query token AND be name-like
                if (
                    query_upper in clean
                    and 2 <= len(clean.split()) <= 4
                    and clean.replace(" ", "").isalpha()
                ):
                    names.add(clean)


    return sorted(names)


def find_pan_for_person(person_name: str):
    db = Chroma(
        collection_name="pdf_content",
        persist_directory=VECTOR_DB_PATH,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"),
    )

    data = db.get(include=["documents", "metadatas"])
    results = {}
    name_upper = person_name.upper()

    for text, meta in zip(data["documents"], data["metadatas"]):
        if not meta:
            continue

        text_upper = text.upper()
        if name_upper not in text_upper:
            continue

        name_pos = text_upper.find(name_upper)

        pan_matches = [
            (m.group(), m.start())
            for m in re.finditer(PAN_REGEX, text_upper)
        ]

        if not pan_matches:
            continue

        nearest_pan = min(
            pan_matches,
            key=lambda p: abs(p[1] - name_pos)
        )[0]

        doc = meta.get("source", "Unknown")
        page = meta.get("page", "Unknown")

        results.setdefault(nearest_pan, {}).setdefault(doc, set()).add(page)

    return results
    logger.info(f"[INTENT] is_pan_search={intent.get('is_pan_search')} | person_name={intent.get('person_name')}")

# ==================================================
# UI
# ==================================================
image = Image.open("imgs/ChatPDF3.png")
st.set_page_config(page_title="PAN & DocChat AI", page_icon=image)
st.markdown("""
    <style>
    /* Chat message base */
    [data-testid="stChatMessage"] {
        display: flex;
        width: 100%;
        margin-bottom: 15px;
        background-color: transparent !important;
    }

    /* USER MESSAGE: Align Right */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
    }

    /* USER CONTENT BUBBLE: Green background, black text */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:nth-child(2) {
        background-color: #DCF8C6 !important;
        color: #000000 !important;
        border-radius: 15px 15px 0px 15px;
        padding: 12px 18px;
        margin-right: 12px;
        max-width: 75%;
        flex-grow: 0;
    }

    /* ASSISTANT MESSAGE: Align Left */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        flex-direction: row;
    }

    /* ASSISTANT CONTENT BUBBLE: Light grey background, black text for readability */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div:nth-child(2) {
        background-color: #F0F2F6 !important;
        color: #000000 !important;
        border-radius: 15px 15px 15px 0px;
        padding: 12px 18px;
        margin-left: 12px;
        max-width: 75%;
        flex-grow: 0;
    }

    /* Ensure Markdown text inside assistant bubbles is black */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown p {
        color: #000000 !important;
    }

    /* Target stats captions to be more visible */
    .stCaption {
        color: #A0A0A0 !important;
        font-weight: 500;
    }
    /* This targets the label specifically to remove the space and text */
        div[data-testid="stSelectbox"] label {
            display: none !important;
        }
    /* This reduces the gap between the subheader and the dropdown */
        div[data-testid="stSelectbox"] {
            margin-top: -15px;
        }
    </style>
    """, unsafe_allow_html=True)

st.subheader("💬 PAN & DocChat AI")

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

    # DB status
    if vector_db_exists():
        st.success("📚 Existing DB loaded")
    else:
        st.info("📭 No DB found")

    pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    hashes, files, _, _ = get_db_metadata()

    if pdfs and st.button("▶ Start Processing"):
        os.makedirs("./tmp", exist_ok=True)
        docs = []

        for pdf in pdfs:
            data = pdf.getbuffer().tobytes()
            h = get_pdf_hash(data)
            if h in hashes:
                continue

            path = f"./tmp/{pdf.name}"
            with open(path, "wb") as f:
                f.write(data)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP
            )
            pdf_doc = fitz.open(path)
            now = datetime.now().isoformat()

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
                                "page": page_no
                            }
                        )
                    )

        if docs:
            Chroma.from_documents(
                docs,
                OllamaEmbeddings(model="nomic-embed-text:latest"),
                collection_name="pdf_content",
                persist_directory=VECTOR_DB_PATH
            )
            st.success(f"Indexed {len(docs)} chunks")
            st.rerun()

    st.markdown("---")
    st.subheader("📂 Indexed Documents")

    if files:
        with st.container():
            st.selectbox(
                "Select a document",
                options=files,
                key="indexed_docs_select"
            )
    else:
        st.caption("No documents indexed yet")

    st.markdown("---")
    st.subheader("Reset DB")

    pwd = st.text_input("Reset DB Password", type="password", key="db_reset_pwd_input")
    if st.button("Reset Vector DB"):
        if pwd == RESET_DB_PASSWORD:
            RAGSystem("pdf_content").delete_collection()
            st.session_state.messages = []
            st.success("Database cleared.")
            st.rerun()
        else:
            st.error("Wrong password")

    st.markdown("---")
    st.subheader("Select SLM Model")
    selected_model = st.selectbox("Model", available_models)
    n_results = st.slider("Retrieved chunks", 1, 15, 5)

# ==================================================
# CHAT LOGIC
# ==================================================
rag = RAGSystem("pdf_content", VECTOR_DB_PATH, n_results)
logger.info(f"[PENDING STATE] {st.session_state.pending_name_choices}")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("Ask me...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 🔁 HANDLE PENDING NAME DISAMBIGUATION (FIXED)
    if st.session_state.pending_name_choices:
        choices = st.session_state.pending_name_choices
        raw = query.strip()

        # Case 1️⃣ : numeric choice (1, 2, ...)
        if raw.isdigit():
            idx = int(raw) - 1
            chosen = choices[idx] if 0 <= idx < len(choices) else None

        # Case 2️⃣ : full-name typed
        else:
            raw_upper = raw.upper()
            chosen = raw_upper if raw_upper in choices else None

        # ❌ Invalid input → STOP EVERYTHING
        if not chosen:
            resp = "❌ Invalid selection. Please type the number or full name."
            st.session_state.messages.append({"role": "assistant", "content": resp})
            with st.chat_message("assistant"):
                st.markdown(resp)
            st.stop()  # ⛔ REQUIRED

        # ✅ Valid choice
        st.session_state.pending_name_choices = None
        matches = find_pan_for_person(chosen)

        if not matches:
            resp = f"❌ No PAN found for **{chosen}**."
        else:
            lines = [f"✅ PAN details for **{chosen}**:\n"]
            for pan, docs in matches.items():
                lines.append(f"🆔 **PAN:** `{pan}`")
                for doc, pages in docs.items():
                    lines.append(
                        f"📄 **{doc}** (Pages: {', '.join(map(str, sorted(pages)))})"
                    )
                lines.append("")
            resp = "\n".join(lines)

        st.session_state.messages.append({"role": "assistant", "content": resp})
        with st.chat_message("assistant"):
            st.markdown(resp)

        st.stop()


    if looks_like_pan(query):
        db = Chroma(
            collection_name="pdf_content",
            persist_directory=VECTOR_DB_PATH,
            embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"),
        )
        data = db.get(include=["documents", "metadatas"])
        matches = {}

        for txt, meta in zip(data["documents"], data["metadatas"]):
            if query.upper() in txt.upper():
                matches.setdefault(meta["source"], set()).add(meta.get("page", "Unknown"))

        if not matches:
            resp = f"❌ PAN **{query.upper()}** not found."
        else:
            lines = [f"✅ PAN **{query.upper()}** found:"]
            for f, pages in matches.items():
                lines.append(f"📄 {f} (Pages: {', '.join(map(str, sorted(pages)))})")
            resp = "\n".join(lines)

        st.session_state.messages.append({"role": "assistant", "content": resp})
        with st.chat_message("assistant"):
            st.markdown(resp)
        st.stop()

    intent_prompt = f"""
User question: "{query}"

Is the user asking to FIND a PAN number for a person?

Return JSON only:
{{
  "is_pan_search": true/false,
  "person_name": "string or null"
}}
"""

    with st.spinner("Analyzing documents..."):
        raw, *_ = rag.generate_response(intent_prompt, selected_model)

    try:
        intent = json.loads(raw)
    except:
        intent = {"is_pan_search": False, "person_name": None}

    if intent["is_pan_search"] and intent["person_name"]:
        candidates = find_candidate_names(intent["person_name"])
        logger.info(f"[CANDIDATES] query_name={intent['person_name']} | candidates={candidates}")

        #if len(candidates) > 1:
        logger.info(
            f"[AMBIGUITY CHECK] "
            f"len={len(candidates)} | "
            f"query_upper={intent['person_name'].upper()} | "
            f"exact_match={intent['person_name'].upper() in candidates}"
        )

        if len(candidates) > 1 and query.upper() not in candidates:
            st.session_state.pending_name_choices = candidates
            logger.info(f"[PENDING SET] pending_name_choices={candidates}")

            resp = (
                "⚠️ Multiple people found:\n\n" +
                "\n".join(f"{i+1}. {n}" for i, n in enumerate(candidates)) +
                "\n\nPlease specify the full name."
            )
        else:
            person = candidates[0] if candidates else intent["person_name"]
            matches = find_pan_for_person(person)

            if not matches:
                resp = f"❌ No PAN found for **{person}**."
            else:
                lines = [f"✅ PAN details for **{person}**:\n"]
                for pan, docs in matches.items():
                    lines.append(f"🆔 **PAN:** `{pan}`")
                    for doc, pages in docs.items():
                        lines.append(
                            f"📄 **{doc}** (Pages: {', '.join(map(str, sorted(pages)))})"
                        )
                    lines.append("")
                resp = "\n".join(lines)

        st.session_state.messages.append({"role": "assistant", "content": resp})
        with st.chat_message("assistant"):
            st.markdown(resp)
        st.stop()
    # -------------------------------
# ✅ NON-PAN QUESTIONS → PURE RAG
# -------------------------------
    else:
        with st.spinner("Thinking..."):
            ans, _, _, _, _ = rag.generate_response(query, selected_model)
            resp = remove_tags(ans)

        st.session_state.messages.append(
            {"role": "assistant", "content": resp}
        )
        with st.chat_message("assistant"):
            st.markdown(resp)

        st.stop()


    ans, _, _, _, _ = rag.generate_response(query, selected_model)
    resp = remove_tags(ans)

    st.session_state.messages.append({"role": "assistant", "content": resp})
    with st.chat_message("assistant"):
        st.markdown(resp)



# import os, re, io, hashlib
# import shutil
# import logging
# import subprocess
# from PIL import Image
# import streamlit as st
# from datetime import datetime
# import fitz  # PyMuPDF
# from rapidfuzz import process, fuzz

# # ==================================================
# # 1. INITIALIZE SESSION STATE (MUST BE FIRST)
# # ==================================================
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "pending_name_choices" not in st.session_state:
#     st.session_state.pending_name_choices = None
# #if "db_reset_pwd_input" not in st.session_state:
#     #st.session_state.db_reset_pwd_input = ""

# # ==================================================
# # HELPERS & EXTRACTION
# # ==================================================
# def extract_pages(pdf_path):
#     doc = fitz.open(pdf_path)
#     pages = []
#     for page_number in range(len(doc)):
#         page = doc[page_number]
#         text = page.get_text("text")
#         pages.append({"page_number": page_number + 1, "text": text})
#     return pages

# PAN_REGEX = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"

# def looks_like_pan(text: str) -> bool:
#     text = text.strip().upper()
#     return bool(re.fullmatch(PAN_REGEX, text))

# def get_pdf_hash(file_bytes: bytes) -> str:
#     return hashlib.sha256(file_bytes).hexdigest()

# def vector_db_exists(path="./PDF_ChromaDB"):
#     return os.path.exists(path) and any(os.scandir(path))

# # ==================================================
# # LANGCHAIN / VECTOR IMPORTS
# # ==================================================
# from langchain_ollama import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_core.documents import Document
# from rag import RAGSystem

# # ==================================================
# # CONFIG
# # ==================================================
# RESET_DB_PASSWORD = "admin123"
# VECTOR_DB_PATH = "./PDF_ChromaDB"
# DEFAULT_CHUNK_SIZE = 512
# DEFAULT_CHUNK_OVERLAP = 100

# def get_db_metadata():
#     if not vector_db_exists():
#         return set(), [], 0, None
#     db = Chroma(
#         collection_name="pdf_content",
#         persist_directory=VECTOR_DB_PATH,
#         embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"),
#     )
#     metas = db.get(include=["metadatas"]).get("metadatas", [])
#     hashes, files, times = set(), set(), []
#     for m in metas:
#         if not m: continue
#         hashes.add(m.get("hash"))
#         files.add(m.get("source"))
#         if m.get("indexed_at"):
#             times.append(m["indexed_at"])
#     last = (datetime.fromisoformat(max(times)).strftime("%Y-%m-%d %H:%M:%S") if times else None)
#     return hashes, sorted(files), len(metas), last

# def remove_tags(text):
#     return re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()

# # ==================================================
# # UI & STYLING (FIXED COLORS)
# # ==================================================
# image = Image.open("imgs/ChatPDF3.png")
# st.set_page_config(page_title="PAN & DocChat AI", page_icon=image, layout="wide")

# st.markdown("""
#     <style>
#     /* Chat message base */
#     [data-testid="stChatMessage"] {
#         display: flex;
#         width: 100%;
#         margin-bottom: 15px;
#         background-color: transparent !important;
#     }

#     /* USER MESSAGE: Align Right */
#     [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
#         flex-direction: row-reverse;
#     }

#     /* USER CONTENT BUBBLE: Green background, black text */
#     [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:nth-child(2) {
#         background-color: #DCF8C6 !important;
#         color: #000000 !important;
#         border-radius: 15px 15px 0px 15px;
#         padding: 12px 18px;
#         margin-right: 12px;
#         max-width: 75%;
#         flex-grow: 0;
#     }

#     /* ASSISTANT MESSAGE: Align Left */
#     [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
#         flex-direction: row;
#     }

#     /* ASSISTANT CONTENT BUBBLE: Light grey background, black text for readability */
#     [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div:nth-child(2) {
#         background-color: #F0F2F6 !important;
#         color: #000000 !important;
#         border-radius: 15px 15px 15px 0px;
#         padding: 12px 18px;
#         margin-left: 12px;
#         max-width: 75%;
#         flex-grow: 0;
#     }

#     /* Ensure Markdown text inside assistant bubbles is black */
#     [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stMarkdown p {
#         color: #000000 !important;
#     }

#     /* Target stats captions to be more visible */
#     .stCaption {
#         color: #A0A0A0 !important;
#         font-weight: 500;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# st.subheader("💬 PAN & DocChat AI")
# _, indexed_files, chunk_count, last_indexed = get_db_metadata()
# st.markdown("---")
# c1, c2, c3 = st.columns(3)
# c1.caption(f"📄 PDFs indexed: {len(indexed_files)}")
# c2.caption(f"🧩 Total chunks: {chunk_count}")
# c3.caption(f"🕒 Last indexed: {last_indexed}" if last_indexed else "🕒 Last indexed: —")
# st.markdown("---")

# # ==================================================
# # SIDEBAR
# # ==================================================
# def get_available_models():
#     try:
#         r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
#         return [l.split()[0] for l in r.stdout.splitlines() if l and "embed" not in l.lower() and "NAME" not in l]
#     except: return []

# available_models = get_available_models()

# with st.sidebar:
#     st.image(image)
#     st.header("📄 Upload PDFs")
#     pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
#     hashes, files, _, _ = get_db_metadata()

#     if pdfs and st.button("▶ Start Processing"):
#         os.makedirs("./tmp", exist_ok=True)
#         docs = []
#         for pdf in pdfs:
#             data = pdf.getbuffer().tobytes()
#             h = get_pdf_hash(data)
#             if h in hashes: continue
#             path = f"./tmp/{pdf.name}"
#             with open(path, "wb") as f: f.write(data)
#             splitter = RecursiveCharacterTextSplitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
#             pdf_doc = fitz.open(path)
#             now = datetime.now().isoformat()
#             for page_no, page in enumerate(pdf_doc, start=1):
#                 text = page.get_text("text")
#                 for ch in splitter.split_text(text):
#                     docs.append(Document(page_content=ch, metadata={"source": pdf.name, "hash": h, "indexed_at": now, "page": page_no}))
#         if docs:
#             Chroma.from_documents(docs, OllamaEmbeddings(model="nomic-embed-text:latest"), collection_name="pdf_content", persist_directory=VECTOR_DB_PATH)
#             st.success(f"Indexed {len(docs)} chunks")
#             st.rerun()

#     st.markdown("---")
#     # Locate the "Reset Vector DB" logic in your sidebar
#     pwd = st.text_input("Reset DB Password", type="password", key="db_reset_pwd_input")

#     if st.button("Reset Vector DB"):
#         if pwd == RESET_DB_PASSWORD:
#             with st.spinner("Clearing Database..."):
#                 # 1. Perform the deletion
#                 RAGSystem("pdf_content").delete_collection()

#                 # 2. Clear the message history
#                 st.session_state.messages = []

#                 # 3. DO NOT manually set st.session_state.db_reset_pwd_input = ""
#                 # Instead, just clear the non-widget state and rerun.
#                 # The rerun will refresh the widget to its default empty state.

#                 st.success("📚 Database and chat history cleared!")
#                 st.rerun()
#         else:
#             st.error("Wrong password")

#     selected_model = st.selectbox("Model", available_models if available_models else ["None Found"])
#     n_results = st.slider("Retrieved chunks", 1, 15, 5)

# # ==================================================
# # CHAT LOGIC
# # ==================================================
# rag = RAGSystem("pdf_content", VECTOR_DB_PATH, n_results)

# # Display history
# for m in st.session_state.messages:
#     with st.chat_message(m["role"]):
#         st.markdown(m["content"])

# query = st.chat_input("Ask me...")
# if query:
#     st.session_state.messages.append({"role": "user", "content": query})
#     with st.chat_message("user"): st.markdown(query)

#     # 1. Raw PAN Search
#     if looks_like_pan(query):
#         db = Chroma(collection_name="pdf_content", persist_directory=VECTOR_DB_PATH, embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"))
#         data = db.get(include=["documents", "metadatas"])
#         matches = {}
#         for txt, meta in zip(data['documents'], data['metadatas']):
#             if query.upper() in txt.upper():
#                 source = meta['source']
#                 page = meta.get('page', 'Unknown')
#                 matches.setdefault(source, set()).add(page)

#         if not matches: resp = f"❌ PAN **{query.upper()}** not found."
#         else:
#             lines = [f"✅ PAN **{query.upper()}** found:"]
#             for f, pages in matches.items():
#                 lines.append(f"📄 {f} (Pages: {', '.join(map(str, sorted(pages)))})")
#             resp = "\n".join(lines)

#     # 2. Intelligent Search
#     # 2. Intelligent Search (REPLACE YOUR OLD ELSE BLOCK WITH THIS)
#     else:
#         with st.spinner("Thinking..."):
#             ans, context_docs, _, _, _ = rag.generate_response(query, selected_model)

#             clean_ans = remove_tags(ans)
#             found_details = []

#             if context_docs:
#                 for doc in context_docs:
#                     # 1. Get chunk content and metadata
#                     content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
#                     source = doc.metadata.get("source", "Unknown") if hasattr(doc, 'metadata') else "Unknown"
#                     page = doc.metadata.get("page", "?") if hasattr(doc, 'metadata') else "?"

#                     # 2. Extract PANs found in this specific chunk
#                     pans_in_chunk = re.findall(PAN_REGEX, content)

#                     for pan in pans_in_chunk:
#                         # Construct the specific string you requested
#                         # This links the PAN directly to its source file and page
#                         detail = f"🔹 **PAN {pan}** found in **{source}** (Page {page})"
#                         found_details.append(detail)

#             # 3. Format the final response
#             if found_details:
#                 # Deduplicate entries (in case multiple chunks from same page are found)
#                 unique_details = "\n".join(list(set(found_details)))

#                 # If the LLM just gave the PAN, we wrap it with the source info
#                 resp = f"The information found for your query is:\n\n{clean_ans}\n\n**Source Details:**\n{unique_details}"
#             else:
#                 resp = clean_ans

#     st.session_state.messages.append({"role": "assistant", "content": resp})
#     with st.chat_message("assistant"):
#         st.markdown(resp)

#     st.session_state.messages.append({"role": "assistant", "content": resp})
#     with st.chat_message("assistant"): st.markdown(resp)

