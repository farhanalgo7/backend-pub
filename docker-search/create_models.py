import os
import json

from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader, TransformersQueryClassifier

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dotenv import load_dotenv
load_dotenv()


def create_resources():
    """Create all resources required for running Semantic Search. This is meant to be executed only one time
    around when starting with a fresh retriever model, or in case some mistake results in previous data to be lost.
    This does not fetch the data, it only creates the resources. So, the fetch data program must be run
    independently from this.
    """
    document_store_dir = os.environ['DOCUMENT_STORE_DIR']
    retriever_dir = os.environ['RETRIEVER_DIR']
    mapping_dir = os.environ['MAPPING_DIR']
    
    # Create the directories if they don't exist
    if not os.path.exists(document_store_dir):
        os.makedirs(document_store_dir)
    
    if not os.path.exists(retriever_dir):
        os.makedirs(retriever_dir)
    
    if not os.path.exists(mapping_dir):
        os.makedirs(mapping_dir)
        
    for doctype in ["news", "tweets"]:
        # Create empty mapping files
        mapping_path = os.path.join(mapping_dir, f"{doctype}_mapping.json")
        if not os.path.exists(mapping_path):
            empty_dict = {}
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(empty_dict, f)
                
        # Create DocumentStores
        docstore_sql_url = os.path.join("sqlite:///", document_store_dir, f"{doctype}_faiss_document_store.db")
        docstore_faiss_index = os.path.join(document_store_dir, f"{doctype}_faiss_index.faiss")
        docstore_faiss_config = os.path.join(document_store_dir, f"{doctype}_faiss_index.json")
        
        document_store = FAISSDocumentStore(sql_url=docstore_sql_url)
        document_store.save(index_path=docstore_faiss_index, config_path=docstore_faiss_config)
    
    # Create and save Retriever
    retriever = DensePassageRetriever(
                                document_store=document_store,
                                query_embedding_model="soheeyang/rdr-question_encoder-single-nq-base",
                                passage_embedding_model="soheeyang/rdr-ctx_encoder-single-nq-base",
                                max_seq_len_query=128,
                                max_seq_len_passage=384,
                                batch_size=16,
                                use_gpu=True,
                                embed_title=True,
                                use_fast_tokenizers=True
                            )
    
    retriever.save(save_dir=retriever_dir)
    
    classifier_tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/question-vs-statement-classifier")
    classifier_model = AutoModelForSequenceClassification.from_pretrained("shahrukhx01/question-vs-statement-classifier")
    
    classifier_tokenizer.save_pretrained(os.environ['QUERY_CLASSIFIER_DIR'])
    classifier_model.save_pretrained(os.environ['QUERY_CLASSIFIER_DIR'])
    
    reader = FARMReader("deepset/minilm-uncased-squad2")
    reader.save(os.environ['READER_DIR'])
    
if __name__ == "__main__":
    create_resources()