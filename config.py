from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Question: {question} Context: {context} Answer:"""

TOKENIZED_STORAGE_PATH = "tokenized_pdfs.pkl"

PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "BGU_shnatonim"
PDF_FOLDER = './shnatonim'

# model = OllamaLLM(model="mistral-small:24b")
# embeddings = OllamaEmbeddings(model="mistral-small:24b")

# model = OllamaLLM(model="deepseek-r1:32b")
# embeddings = OllamaEmbeddings(model="deepseek-r1:32b")
# model = OllamaLLM(model="deepseek-r1:70b")
# embeddings = OllamaEmbeddings(model="deepseek-r1:70b")


model = OllamaLLM(model="deepseek-r1:14b")
embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
