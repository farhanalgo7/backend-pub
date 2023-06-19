import os
import json
import datetime as dt

from models import load_document_store
from utils import check_type, convert_datetime_to_date, get_today, outside_threshold

from dotenv import load_dotenv
load_dotenv()

def remove_older_docs(threshold, type="news"):
    """Remove documents older than the threshold date.

    Args:
        threshold (datetime.date): Threshold date
        type (str, optional): Type, either "news" or "tweets". Defaults to "news".
    """
    check_type(type)
    
    document_store = load_document_store(type=type)
    
    # Set Mapping file path
    mapping_file = os.path.join(os.environ['MAPPING_DIR'], f"{type}_mapping.json")

    with open(mapping_file, "r", encoding="utf-8") as fp:
        mapping = json.load(fp)    

    # Get list of UUIDs from mapping which are outside threshold
    remove_uids = [item for item in mapping if outside_threshold(convert_datetime_to_date(mapping[item][1]), threshold)]
    
    # Construct list of document names to remove
    remove_docs = [f"doc_{uid}.txt" for uid in remove_uids]
    
    # Get list of IDs in DocumentStore for the documents to remove
    remove_ids = [doc.id for doc in document_store.get_all_documents() if doc.meta['name'] in remove_docs]
    
    if remove_ids:
        old_doc_count = document_store.get_document_count()
        old_embed_count = document_store.get_embedding_count()
        print(f"Before: {old_doc_count} docs and {old_embed_count} embeddings")
        
        # Remove documents
        document_store.delete_documents(ids=remove_ids)
        
        # Remove entries from mapping file
        new_mapping = {uid: mapping[uid] for uid in mapping if uid not in remove_uids}
        
        with open(mapping_file, "w", encoding="utf-8") as fp:
            json.dump(new_mapping, fp, indent=2)

        print("Documents Deleted")
        new_doc_count = document_store.get_document_count()
        new_embed_count = document_store.get_embedding_count()
        print(f"After: {new_doc_count} docs and {new_embed_count} embeddings")
        
        assert old_doc_count >= new_doc_count, "Old and New doc count mismatch!"
        assert old_embed_count >= new_embed_count, "Old and New embeddings count mismatch!"
        
    else:
        print("No docs older than 30 days found in Document Store!")
    
    docstore_dir = os.environ["DOCUMENT_STORE_DIR"]
    docstore_path = os.path.join(docstore_dir, f"{type}_faiss_index.faiss")
    document_store.save(docstore_path)
    

if __name__ == "__main__":
    # Get today as datetime.date
    today = get_today()
    remove_threshold = today - dt.timedelta(days=30)
    
    for type in ["news", "tweets"]:
        print(f"Processing {type.title()} :")
        remove_older_docs(remove_threshold, type=type)
        print("\n----------------------------\n")
    