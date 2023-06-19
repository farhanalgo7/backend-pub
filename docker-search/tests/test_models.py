from haystack.document_stores.faiss import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader, TransformersQueryClassifier

from models import (
    load_document_store, 
    load_retriever,
    load_query_classifier,
    load_reader
)


class TestNewsDocumentStore:
    document_store = load_document_store(type="news")
    
    def test_load_document_store(self):
        assert isinstance(self.document_store, FAISSDocumentStore)
        
    def test_document_and_embedding_count(self):
        document_count = self.document_store.get_document_count()
        embedding_count = self.document_store.get_embedding_count()
        
        assert document_count == embedding_count
        
    def test_names_in_document_metadata(self):
        for document in self.document_store.get_all_documents():
            assert 'name' in document.meta
            assert document.meta['name'] is not "" and document.meta['name'] is not None


class TestTweetsDocumentStore:
    document_store = load_document_store(type="tweets")
    
    def test_load_document_store(self):
        assert isinstance(self.document_store, FAISSDocumentStore)
        
    def test_document_and_embedding_count(self):
        document_count = self.document_store.get_document_count()
        embedding_count = self.document_store.get_embedding_count()
        
        assert document_count == embedding_count
        
    def test_names_in_document_metadata(self):
        for document in self.document_store.get_all_documents():
            assert 'name' in document.meta
            assert document.meta['name'] is not "" and document.meta['name'] is not None

def test_retriever():
    document_store = load_document_store()
    retriever = load_retriever(document_store)

    assert isinstance(retriever, DensePassageRetriever)
    
    
class TestQueryClassifier:
    query_classifier = load_query_classifier()
    
    def test_load_query_classifier(self):
        assert isinstance(self.query_classifier, TransformersQueryClassifier)
    
    # Query Classifier returns ({}, 'output_1') or ({}, 'output_2')
    # Output 1 for Question, Output 2 for Statement
    
    def test_question_classification(self):
        response = self.query_classifier.run("Is this a question?")
        assert response[1] == 'output_1'
        
    def test_statement_classification(self):
        response = self.query_classifier.run("This is a statement.")
        assert response[1] == 'output_2'

def test_load_reader():
    reader = load_reader()
    assert isinstance(reader, FARMReader)