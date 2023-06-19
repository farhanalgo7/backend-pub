import os
import json
import datetime as dt
import logging

from haystack.utils import convert_files_to_docs

from models import load_document_store, load_preprocessor, load_retriever
from utils import check_type

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

def update_embeddings(type="news"):
    """Write new documents to DocumentStore and update embeddings.

    Args:
        type (str, optional): Type, either "news" or "tweets". Defaults to "news".
    """
    check_type(type)
    
    # Load DocumentStore and Retriever
    document_store = load_document_store(type=type)
    retriever = load_retriever(document_store, type=type)
    
    # Load Pre-Processor
    preprocessor = load_preprocessor()
    
    # Load text files from NEWS_DIR/TWEETS_DIR environment variable paths
    all_docs = convert_files_to_docs(dir_path=os.environ[f"{type.upper()}_DIR"])
    
    
    # Process docs and write to DocumentStore
    processed_docs = preprocessor.process(all_docs)
    print(f"\nProcessing {type} docs\nn_files_input: {len(all_docs)}\nn_docs_output: {len(processed_docs)}")
    
    document_store.write_documents(processed_docs, duplicate_documents='skip')
    
    # Remove documents older than threshold
    # print("*** Calling remove_older_docs()...")
    # document_store = remove_older_docs(document_store, remove_threshold, type=type)

    # Update embeddings
    document_store.update_embeddings(retriever, update_existing_embeddings=False)
    # document_store.update_embeddings(retriever)
    
    document_dir = os.environ[f"{type.upper()}_DIR"]        # NEWS_DIR or TWEETS_DIR
    document_store_dir = os.environ["DOCUMENT_STORE_DIR"]
    
    # Save DocumentStore at DOCUMENT_STORE_DIR/news_faiss_index.faiss or 
    # DOCUMENT_STORE_DIR/tweets_faiss_index.faiss depending on type
    docstore_path = os.path.join(document_store_dir, f"{type}_faiss_index.faiss")
    document_store.save(docstore_path)
    
    print(f"DocumentStore updated for {type.title()}")
    
    # Remove files that have already been written to DocumentStore
    documents_in_dir = os.listdir(document_dir)
    if len(documents_in_dir):
        for f in documents_in_dir:
            os.remove(os.path.join(document_dir, f))
