# Rag for BGU

## Files Explanation:
### Report Files:
1. RAG_for_BGU_report.pdf - The report about the entire system.
1. retrieval_performence_report.ipynb - containing explanation and extensive analysis about the retrieving method, also contains some explanations about the parameters choices I made.
2. model_response_report.ipynb - containing some generated answers from the model

### Python files:
1. main.py - main runner file
2. config.py - containing all the configs for the project (like model)
3. chroma.py - containing all chroma utils
4. process_pdfs.py - containing all preprocessing utils needed for pdfs
5. retrieval_methods - containing the retrieval methods I used for the system

## To run:

First, install all the packages in requirements.txt:
```bash
pip install -r requirements.txt
```
Note - Some packages might require to download "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/  
Note - I used python 3.12 for my project

Install Ollama if you don't have it already. Then, download deepseek:r1:32b parameters (The model I used) with the following command:
```bash
ollama pull deepseek-r1:32b
```  
Note - The model weighs 20GB.  

Download tokenizer:
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

Or if it doesn't work just copy and run in a python file:
```
import nltk
nltk.download("punkt_tab")
```

Finally, run:
```bash 
streamlit run .\main.py
```   
Note - When running, you might sometimes get an irrelevant torch error, ignore it.

<br>

Additional Steps:

Preprocess pdfs if chroma directory doesnt exists (should exist if cloned from github) and save to chromaDB (import from process_pdfs.py and run in python file):
```
process_PDFS_folder(PDF_FOLDER)
```

Tokenize pdfs if chroma directory doesnt exists (should exist if cloned from github) (import from process_pdfs.py and run in python file):
```
tokenize_and_store(PDF_FOLDER)
```
