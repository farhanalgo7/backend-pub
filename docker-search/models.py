import os

from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, PreProcessor, FARMReader, TransformersQueryClassifier
from haystack.pipelines import DocumentSearchPipeline

from utils import check_type

from dotenv import load_dotenv
load_dotenv()

def load_document_store(type="news"):
    """Load the FAISS Document Store Index, for the given type of "news" or "tweets". The 
    DocumentStore index is assumed to be present at DOCUMENT_STORE_DIR environment variable 
    with a naming convention of f"{type}_faiss_index.faiss" 

    Args:
        type (str, optional): Type of DocumentStore, either "news" or "tweets". Defaults to "news".

    Returns:
        FAISSDocumentStore: The FAISS DocumentStore of the given type.
    """
    check_type(type)
    
    try:
        document_store = FAISSDocumentStore(
                faiss_index_path=os.path.join(os.environ["DOCUMENT_STORE_DIR"], f"{type}_faiss_index.faiss"),
                faiss_config_path=os.path.join(os.environ["DOCUMENT_STORE_DIR"], f"{type}_faiss_index.json")
            )
        return document_store
    
    except ValueError as ve:
        print("Cannot find the Document Store. Make sure the environment variables are correctly set and the document store is present.")
        print("Also, make sure that the document store correctly has the same number of embeddings as documents.")
        print("If the document stores and models do not exist, create them by running create_models.py")
        print(f"\nException Info: \n{ve}\n")
        
    except Exception as e:
        print(f"\nException Info: \n{e}\n")

def load_retriever(document_store, type="news"):
    """Load a DensePassageRetriever (DPR) for the given type of "news" or "tweets". The retriever
    models are assumed to be present at RETRIEVER_DIR, with the query and passage encoding
    models at QUERY_ENCODER_DIR and PASSAGE_ENCODER_DIR environment variables respectively.
    
    The current models used:
    query_embedding_model: soheeyang/rdr-question_encoder-single-nq-base
    passage_embedding_model: soheeyang/rdr-ctx_encoder-single-nq-base

    Args:
        document_store (FAISSDocumentStore): A FAISS DocumentStore corresponding to the type.   
        type (str, optional): The type of retriever, either "news" or "tweets". Defaults to "news".

    Returns:
        DensePassageRetriever: The DPR for the given DocumentStore
    """
    check_type(type)
    
    try:
        retriever = DensePassageRetriever(document_store=document_store,
                                        query_embedding_model=os.environ['QUERY_ENCODER_DIR'],
                                        passage_embedding_model=os.environ['PASSAGE_ENCODER_DIR'],
                                        max_seq_len_query=128,
                                        max_seq_len_passage=384,
                                        batch_size=16,
                                        use_gpu=True,
                                        embed_title=True,
                                        use_fast_tokenizers=True)
        return retriever
    
    except Exception as e:
        print("Make sure that the environment variables are properly set and the retriever is present at the correct path.")
        print("If the document stores and models do not exist, create them by running create_models.py")
        print(f"\nException Info: \n{e}\n")


def load_document_search_pipeline(type="news"):
    """Returns a DocumentSearchPipeline for the given type of 'news' or 'tweets'.
    It loads DocumentStore and Retriever first, then constructs pipeline.

    Args:
        type (str, optional): The type of retriever, either "news" or "tweets". Defaults to "news".

    Returns:
        DocumentSearchPipeline: Haystack's Document Search Pipeline that can used for querying
    """
    
    check_type(type)
    
    document_store = load_document_store(type)
    retriever = load_retriever(document_store, type)
    pipeline = DocumentSearchPipeline(retriever)
    
    return pipeline


def load_preprocessor():
    """Load Haystack PreProcessor to process documents.

    Returns:
        PreProcessor: A Haystack PreProcessor
    """
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=False
    )
    
    return preprocessor


def load_query_classifier():
    """Load a Question vs Statement classifier.

    Returns:
        TransformersQueryClassifier: A TransformersQueryClassifier for Question vs Statement classification.
    """
    try:
        classifier = TransformersQueryClassifier(os.environ['QUERY_CLASSIFIER_DIR'])
        return classifier

    except OSError as oe:
        print("Make sure the environment variables are correctly set and that the classifier is present at the correct path.")
        print("If the document stores and models do not exist, create them by running create_models.py")

        print(f"\nException Info: \n{oe}\n")
        
    except Exception as e:
        print(f"\nException Info: \n{e}\n")

def load_reader():
    """Load Haystack FARMReader to extract answers from the documents.

    Returns:
        FARMReader: Haystack FARMReader
    """
    try:
        reader = FARMReader(os.environ['READER_DIR'], top_k=3)
        return reader
    
    except Exception as e:
        print("Make sure the environment variables are correctly set and that the reader model is at the correct path.")
        print("If the document stores and models do not exist, create them by running create_models.py")
        print(f"\nException Info: \n{e}\n")