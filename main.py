import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from config import template, model
from process_pdfs import load_tokenized_documents
from retrieval_methods import hybrid_retrieval

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

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

# def get_all_documents(PDF_FOLDER):
#     all_documents = []
#     for filename in os.listdir(PDF_FOLDER):
#         print(filename)
#         if filename.endswith(".pdf"):
#             pdf_path = os.path.join(PDF_FOLDER, filename)
#             print(f"Getting {filename}")
#
#             all_documents.extend(get_pdf_file_content(pdf_path))
#
#     return all_documents



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
