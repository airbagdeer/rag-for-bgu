from uuid import uuid4

from langchain_chroma import Chroma

from config import COLLECTION_NAME, embeddings, PERSIST_DIRECTORY

vector_store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings,
                      persist_directory=PERSIST_DIRECTORY)

def save_to_chroma_db(splits):
    print("Saving to chroma DB")
    uuids = [str(uuid4()) for _ in range(len(splits))]
    vector_store.add_documents(documents=splits, ids=uuids)
    print("Saved to chroma DB")
