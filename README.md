# Rag for BGU

## Files Explanation:
### Report Files:
1. retrieval_performence_report.ipynb - containing explanation and extensive analysis about the retrieving method, also contains some explanations about the parameters choices I made.
2. model_response_report.ipynb - containing some generated answers from the model

### Python files:
1. main.py - main runner file
2. config.py - containing all the configs for the project (like model)
3. chroma.py - containing all chroma utils
4. process_pdfs.py - containing all preprocessing utils needed for pdfs
5. retrieval_methods - containing the retrieval methods I used for the system

## To run:

First, install all the packages in requirements.txt.   
Note - Some packages might require to download "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/

Install Ollama if you dont have it already. Then, download deepseek:r1:32b parameters (The model I used) with the following command:
```bash
ollama pull deepseek-r1:32b
```  
Note - The model weighs 20GB.  

Download tokenizer (run in a python file):
```
nltk.download("punkt_tab")
```
Finally, run:
```bash 
streamlit run .\main.py
```  
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
