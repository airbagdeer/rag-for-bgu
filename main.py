import os
import pickle
from uuid import uuid4

from bidi.algorithm import get_display
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import word_tokenize
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from langchain_core.runnables import chain
from typing import List
from rank_bm25 import BM25Okapi
import streamlit as st
import heapq

# TODO for morning:
# 1. compare the performance of retrieval methods,
# maybe ask aviv if to write a report on that as well, I think yes, check how to compare them
# might be even better to right the report in a notebook
# 2. check more models, both for generation and embedding. specifically ones in hebrew. use the benchmark in the links
# 3. implement text preprocessing for hebrew text.
# 4. Evaluate the model and analyze responses.
# 5. improve streamlit so it saves conversation
# 6. improve rag so it has memory https://python.langchain.com/docs/tutorials/qa_chat_history/
# 7. start working on the report
# 8. add types
# 9. split functions to different folders
# 10.Change the prompt to hebrew

# END TODO:
# Move configs to configuration file
# remove all comments
# add readme.MD
# add requirement.txt
# check if runs on different PC
# deploy on internet?
# remove redundant prints

# nltk.download("punkt_tab")
template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Question: {question} Context: {context} Answer:"""

TOKENIZED_STORAGE_PATH = "tokenized_pdfs.pkl"

PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "BGU_shnatonim"
PDF_FOLDER = './shnatonim'

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

model = OllamaLLM(model="deepseek-r1:14b")
embeddings = OllamaEmbeddings(model="deepseek-r1:14b")

# model = OllamaLLM(model="mistral-small:24b")
# embeddings = OllamaEmbeddings(model="mistral-small:24b")

# model = OllamaLLM(model="deepseek-r1:32b")
# embeddings = OllamaEmbeddings(model="deepseek-r1:32b")
# model = OllamaLLM(model="deepseek-r1:70b")
# embeddings = OllamaEmbeddings(model="deepseek-r1:70b")


vector_store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings,
                      persist_directory=PERSIST_DIRECTORY)

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


def save_to_chroma_db(splits):
    print("Saving to chroma DB")
    uuids = [str(uuid4()) for _ in range(len(splits))]
    vector_store.add_documents(documents=splits, ids=uuids)
    print("Saved to chroma DB")


def get_all_documents(PDF_FOLDER):
    all_documents = []
    for filename in os.listdir(PDF_FOLDER):
        print(filename)
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"Getting {filename}")

            all_documents.extend(get_pdf_file_content(pdf_path))

    return all_documents


def dense_retrieval(query, top_k=1, filter_criteria=None):
    embedded_query = embeddings.embed_query(query)
    results = vector_store.similarity_search_by_vector_with_relevance_scores(embedding=embedded_query, k=top_k,
                                                                             filter=filter_criteria)
    return results


def tokenize_and_store(PDF_FOLDER, storage_path=TOKENIZED_STORAGE_PATH):
    all_tokenized_docs = {}

    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"Tokenizing {filename}")

            document = get_pdf_file_content(pdf_path)
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


def BM25_retrieval(query, tokenized_documents, top_k=1, k1=1.5, b=0.75):
    pdfs = list(tokenized_documents.keys())

    bm25 = BM25Okapi(list(tokenized_documents.values()), k1=k1, b=b)
    tokenized_query = word_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    results = {pdfs[i]: scores[i] for i in range(len(pdfs))}

    top_k_results = heapq.nlargest(top_k, results.items(), key=lambda x: x[1])
    top_k_results = {top_k_results[i][0]: top_k_results[i][1] for i in range(top_k)}

    return top_k_results


def hybrid_retrieval(query, tokenized_documents, tok_k_vector_similarity=5, top_k_BM25=3, k1=1.5, b=0.75):
    relevant_pdfs_by_lexical_similarity = BM25_retrieval(query, tokenized_documents, top_k_BM25, k1, b)

    print(relevant_pdfs_by_lexical_similarity)

    filter_criteria = {
        "source": {"$in": list(relevant_pdfs_by_lexical_similarity.keys())}
    }

    results = dense_retrieval(query, tok_k_vector_similarity, filter_criteria)

    return results


# print(hybrid_retrieval('מה זה הנדסת מחשבים?'))

# question = "מה זה היחידה להנדסה גרעינית ?"
# question = "מה זה הנדסת מחשבים?"
# retrieved_docs = hybrid_retrieval(question)
# docs_content = "\n\n".join(doc[0].page_content for doc in retrieved_docs)
# print(docs_content)
# promptTemplate = ChatPromptTemplate.from_template(template)
# prompt = promptTemplate.invoke({"question": question, "context": docs_content})
# answer = model.invoke(prompt)
# print(answer)


# print(dense_retrieval('מה זה הנדסת מחשבים?', 2)[0])
# print(dense_retrieval('מה זה הנדסת מחשבים?', 2)[1])
# print(BM25_retrieval(query='גרעינית', top_k=2))


# Preprocess pdfs and save to chromaDB:
# process_PDFS_folder(PDF_FOLDER)
# tokenize_and_store(PDF_FOLDER)


# To run streamlit: streamlit run .\main.py
tokenized_documents = load_tokenized_documents()
question = st.chat_input()

if question:
    st.chat_message("user").write(question)
    retrieved_docs = hybrid_retrieval(question, tokenized_documents)
    docs_content = "\n\n".join(doc[0].page_content for doc in retrieved_docs)
    promptTemplate = ChatPromptTemplate.from_template(template)
    prompt = promptTemplate.invoke({"question": question, "context": docs_content})
    answer = model.invoke(prompt)
    st.chat_message("assistant").write(answer)
