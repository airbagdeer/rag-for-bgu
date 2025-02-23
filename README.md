check if needed:
nltk.download("punkt_tab")

Preprocess pdfs and save to chromaDB:
process_PDFS_folder(PDF_FOLDER)
tokenize_and_store(PDF_FOLDER)

To run streamlit:
```bash 
streamlit run .\main.py