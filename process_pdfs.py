import os
import pickle

from bidi.algorithm import get_display
from langchain.schema import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import word_tokenize
from chroma import save_to_chroma_db
from config import TOKENIZED_STORAGE_PATH

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

def process_PDFS_folder(PDF_FOLDER):
    documents = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"Processing {filename}")
            documents.append(get_pdf_file_content(pdf_path, filename))

    splits = text_splitter.split_documents(documents)
    save_to_chroma_db(splits)


def get_pdf_file_content(pdf_path, filename):
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    full_pdf = ""

    for doc in documents:
        full_pdf += get_display(doc.page_content)

    metadata = {
        "source": filename
    }

    return Document(page_content=full_pdf, metadata=metadata)

def tokenize_and_store(PDF_FOLDER, storage_path=TOKENIZED_STORAGE_PATH):
    all_tokenized_docs = {}

    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"Tokenizing {filename}")

            document = get_pdf_file_content(pdf_path, filename)
            tokenized_texts = word_tokenize(document.page_content)

            all_tokenized_docs[filename] = tokenized_texts

    with open(storage_path, "wb") as f:
        pickle.dump(all_tokenized_docs, f)

    print(f"Tokenized documents saved at {storage_path}")


def load_tokenized_documents(storage_path=TOKENIZED_STORAGE_PATH):
    if os.path.exists(storage_path):
        with open(storage_path, "rb") as f:
            all_tokenized_docs = pickle.load(f)
        print(f"Loaded precomputed tokenized data from {storage_path}")

        return all_tokenized_docs

    print("No precomputed data found, please run tokenize_and_store() first.")
    return []
