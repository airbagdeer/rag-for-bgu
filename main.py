import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from chroma import vector_store_manager
from config import template, model, MODEL, PDF_FOLDER, embedding_manager
from process_pdfs import load_tokenized_documents, embedd_pdfs_and_save, tokenize_and_store
from retrieval_methods import hybrid_retrieval

# TODO for morning:
# 5. improve streamlit so it saves conversation
# 6. improve rag so it has memory https://python.langchain.com/docs/tutorials/qa_chat_history/
# 7. start working on the report
# 8. add types

# END TODO:
# remove all comments
# add readme.MD
# add requirement.txt
# check if runs on different PC
# deploy on internet?




# To run streamlit: streamlit run .\main.py


# Use only if the pdfs aren't tokenized:
# tokenize_and_store(PDF_FOLDER=PDF_FOLDER)
tokenized_documents = load_tokenized_documents()

vector_store_manager.update_embedding_model('dicta-il/dictabert')

# Use only if you haven't embedded and saved to chroma
# embedd_pdfs_and_save(PDF_FOLDER=PDF_FOLDER)


question = st.chat_input()
print('Using Model:', MODEL)
print('Using Embedding Model:', embedding_manager.get_embedding_model_name())
print('Persistent storage:', embedding_manager.get_persistent_directory())

if question:
    st.chat_message("user").write(question)
    retrieved_docs = hybrid_retrieval(question, tokenized_documents, tok_k_vector_similarity=2)
    docs_content = "\n\n".join(doc[0].page_content for doc in retrieved_docs)
    promptTemplate = ChatPromptTemplate.from_template(template)
    prompt = promptTemplate.invoke({"question": question, "context": docs_content})
    answer = model.invoke(prompt.messages[0].content)
    st.chat_message("assistant").write(answer)
