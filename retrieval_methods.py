import heapq
from uuid import uuid4

from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from chroma import vector_store_manager
from config import embedding_manager
from process_pdfs import clean_hebrew_text


def dense_retrieval(query, top_k=5, filter_criteria=None):
    print("Started dense retrieval")
    embedded_query = embedding_manager.get_embedding_model().embed_query(query)
    results = vector_store_manager.get_vector_store().similarity_search_by_vector_with_relevance_scores(embedding=embedded_query, k=top_k,
                                                                             filter=filter_criteria)
    print("Finished dense retrieval")
    return results


def BM25_retrieval(query, tokenized_documents, top_k=3, k1=1.5, b=0.75, clean_text= True):
    print("Started BM25 retrieval")
    pdfs = list(tokenized_documents.keys())

    bm25 = BM25Okapi(list(tokenized_documents.values()), k1=k1, b=b)

    if(clean_text):
        query = clean_hebrew_text(query)

    tokenized_query = word_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    results = {pdfs[i]: scores[i] for i in range(len(pdfs))}

    top_k_results = heapq.nlargest(top_k, results.items(), key=lambda x: x[1])
    top_k_results = {top_k_results[i][0]: top_k_results[i][1] for i in range(top_k)}

    print("Finished BM25 retrieval")
    return top_k_results


def hybrid_retrieval(query, tokenized_documents, tok_k_vector_similarity=10, top_k_BM25=7, k1=1.5, b=0.75):
    relevant_pdfs_by_lexical_similarity = BM25_retrieval(query, tokenized_documents, top_k_BM25, k1, b)

    filter_criteria = {
        "source": {"$in": list(relevant_pdfs_by_lexical_similarity.keys())}
    }

    results = dense_retrieval(query, tok_k_vector_similarity, filter_criteria)

    return results
