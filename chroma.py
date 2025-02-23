from uuid import uuid4

from langchain_chroma import Chroma
from tqdm import tqdm

from config import COLLECTION_NAME, embedding_manager

COLLECTION_METADATA = {"hnsw:space": "cosine"}

class VectorStoreManager:
    def __init__(self, collection_name=COLLECTION_NAME, collection_metadata=COLLECTION_METADATA):
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata if collection_metadata is not None else {}
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self):
        storage_path = embedding_manager.get_persistent_directory()

        return Chroma(
            collection_name=self.collection_name,
            embedding_function=embedding_manager.get_embedding_model(),
            persist_directory=storage_path,
            collection_metadata=self.collection_metadata
        )

    def update_embedding_model(self, new_model_name):
        embedding_manager.set_embedding_model(new_model_name)
        self.vector_store = self._initialize_vector_store()

    def get_vector_store(self):
        return self.vector_store


vector_store_manager = VectorStoreManager()

def save_to_chroma_db(splits, batch_size=1):
    storage_path = embedding_manager.get_persistent_directory()

    print(f"Saving to ChromaDB at path: {storage_path} using model: {embedding_manager.get_embedding_model()}")

    total_docs = len(splits)
    uuids = [str(uuid4()) for _ in range(total_docs)]

    for i in tqdm(range(0, total_docs, batch_size), desc="Saving Progress", unit="batch"):
        batch_splits = splits[i:i + batch_size]
        batch_uuids = uuids[i:i + batch_size]
        vector_store_manager.get_vector_store().add_documents(documents=batch_splits, ids=batch_uuids)

    print("Successfully saved all documents to ChromaDB")

