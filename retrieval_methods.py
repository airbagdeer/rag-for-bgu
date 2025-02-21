import heapq

from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from config import embeddings
from chroma import vector_store

def dense_retrieval(query, top_k=1, filter_criteria=None):
    embedded_query = embeddings.embed_query(query)
    results = vector_store.similarity_search_by_vector_with_relevance_scores(embedding=embedded_query, k=top_k,
                                                                             filter=filter_criteria)
    return results


def BM25_retrieval(query, tokenized_documents, top_k=1, k1=1.5, b=0.75):
    pdfs = list(tokenized_documents.keys())

    bm25 = BM25Okapi(list(tokenized_documents.values()), k1=k1, b=b)
    tokenized_query = word_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    results = {pdfs[i]: scores[i] for i in range(len(pdfs))}

    top_k_results = heapq.nlargest(top_k, results.items(), key=lambda x: x[1])
    top_k_results = {top_k_results[i][0]: top_k_results[i][1] for i in range(top_k)}

    return top_k_results


def hybrid_retrieval(query, tokenized_documents, tok_k_vector_similarity=5, top_k_BM25=3, k1=1.5, b=0.75):
    relevant_pdfs_by_lexical_similarity = BM25_retrieval(query, tokenized_documents, top_k_BM25, k1, b)

    print(relevant_pdfs_by_lexical_similarity)

    filter_criteria = {
        "source": {"$in": list(relevant_pdfs_by_lexical_similarity.keys())}
    }

    results = dense_retrieval(query, tok_k_vector_similarity, filter_criteria)

    return results
