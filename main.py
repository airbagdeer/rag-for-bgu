import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from chroma import vector_store_manager
from config import template, model, MODEL, PDF_FOLDER, embedding_manager, REMOVE_THINK_TAGS
from process_pdfs import load_tokenized_documents, embedd_pdfs_and_save, tokenize_and_store
from retrieval_methods import hybrid_retrieval
import re

# To run streamlit: streamlit run .\main.py

# Use only if the pdfs aren't tokenized:
# tokenize_and_store(PDF_FOLDER=PDF_FOLDER)
tokenized_documents = load_tokenized_documents()

vector_store_manager.update_embedding_model('dicta-il/dictabert')

# Use only if you haven't embedded and saved to chroma
# embedd_pdfs_and_save(PDF_FOLDER=PDF_FOLDER)


def split_think(answer):
    match = re.search(r"<think>(.*?)</think>\s*(.*)", answer, re.DOTALL)
    if match:
        think_content = match.group(1).strip()
        actual_answer = match.group(2).strip()
        return think_content, actual_answer
    else:
        return None, answer


st.set_page_config(page_title="RAG for BGU", layout="centered")

st.sidebar.write("Generative Model:", MODEL)
st.sidebar.write("Embedding Model:", embedding_manager.get_embedding_model_name())

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).markdown(msg["content"])

question = st.chat_input("Type your question here...")

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)

    with st.spinner("Retrieving documents..."):
        retrieved_docs = hybrid_retrieval(question, tokenized_documents)
        docs_content = "\n\n".join(doc[0].page_content for doc in retrieved_docs)

    prompt_template = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.invoke({"question": question, "context": docs_content})

    with st.spinner("Generating answer..."):
        answer = model.invoke(prompt.messages[0].content)

    if REMOVE_THINK_TAGS:
        thinking_process, answer = split_think(answer)

    st.session_state["messages"].append({"role": "RAG", "content": answer})
    st.chat_message("RAG").markdown(answer)
