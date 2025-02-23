# Rag for BGU

## Files Explanation:
### Report Files:
retrieval_performence_report.ipynb - containing explanation and extensive analysis about the retrieving method, also contains some explanations about the parameters choices I made.
model_response_report.ipynb - containing some generated answers from the model

### Python files:
main.py - main runner file
config.py - containing all the configs for the project (like model)
chroma.py - containing all chroma utils
process_pdfs.py - containing all preprocessing utils needed for pdfs
retrieval_methods - containing the retrieval methods I used for the system

## To run:

First, install all the packages in requirements.tx

Install Ollama if you dont have it already.

Download deepseek:r1:32b parameters (The model I used) with the following command:
```bash
ollama pull deepseek-r1:32b
```

Download tokenizer (run in python code):
```
nltk.download("punkt_tab")
```

Preprocess pdfs if chroma directory doesnt exists (should exists if cloned from github) and save to chromaDB (import from process_pdfs.py and run in python file):
```
process_PDFS_folder(PDF_FOLDER)
```

Tokenize pdfs if chroma directory doesnt exists (should exists if cloned from github) (import from process_pdfs.py and run in python file):
```
tokenize_and_store(PDF_FOLDER)
```

To run streamlit:
```bash 
streamlit run .\main.py
```