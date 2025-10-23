import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

from pdf2image import convert_from_path
import pytesseract
import tempfile

DATA_DIR = "./data/docs"
OUTPUT_DIR = "./vectorstore"

def extract_text_ocr(pdf_path):
    text = ""
    with tempfile.TemporaryDirectory() as path:
        images = convert_from_path(pdf_path, dpi=200, output_folder=path)
        for i, img in enumerate(images):
            text += f"\n--- Page {i+1} ---\n" + pytesseract.image_to_string(img)
    return text

def load_pdfs():
    docs = []
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".pdf"): continue
        pdf_path = os.path.join(DATA_DIR, file)
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            text = " ".join([p.page_content for p in pages])
            if len(text.strip()) < 50:  # too little text, use OCR
                text = extract_text_ocr(pdf_path)
            docs.append(Document(page_content=text, metadata={"source": file}))
        except Exception:
            text = extract_text_ocr(pdf_path)
            docs.append(Document(page_content=text, metadata={"source": file}))
    return docs

def build_vectorstore():
    docs = load_pdfs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedder)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vectorstore.save_local(OUTPUT_DIR)
    print(f"✅ Vectorstore saved. Total chunks: {len(chunks)}")

if __name__ == "__main__":
    build_vectorstore()
