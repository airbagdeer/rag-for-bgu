from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM

# template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Question: {question} Context: {context} Answer:"""
template = """מטרתך לעזור לענות על שאלות. תשתמש בהקשרים הבאים כדי לענות על השאלה. תשתמש רק בהקשרים הרלוונטיים אם יש כאלו. אם אתה לא יודע מה התשובה, פשוט תגיד אני לא יודע. תענה עד 3 משפטים מדוייקים.\n שאלה: {question},\n הקשרים: {context}"""

TOKENIZED_STORAGE_PATH = "tokenized_pdfs.pkl"

COLLECTION_NAME = "BGU_shnatonim"
PDF_FOLDER = './shnatonim'

# MODEL = "phi4:14b"
MODEL = "deepseek-r1:32b"


class EmbeddingManager:
    def __init__(self, model_name="dicta-il/dictabert", persist_directory = "./chroma_db"):
        self._model_name = model_name
        self._embedding_model = HuggingFaceEmbeddings(model_name=self._model_name)
        self._persist_directory = persist_directory

    def set_embedding_model(self, new_model_name):
        self._model_name = new_model_name
        self._embedding_model = HuggingFaceEmbeddings(model_name=self._model_name)
        self.set_persist_directory()

    def set_persist_directory(self):
        self._persist_directory = './chroma_db' + '_' + self._model_name.replace('/', '-')
    def get_embedding_model(self):
        return self._embedding_model

    def get_embedding_model_name(self):
        return self._model_name

    def get_persistent_directory(self):
        return self._persist_directory




embedding_manager = EmbeddingManager()

model = OllamaLLM(model=MODEL)