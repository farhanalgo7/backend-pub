# Background

This project is built with [Haystack](https://haystack.deepset.ai/).

## Keyword Search 

## Semantic Search

Semantic Search takes into account the semantics (meaning and intent) of a query, as compared to a keyword search which simply searches for the exact words in the query. This allows for better search results, in principle, which are more relevant to the actual query.

For this, we need to represent words as vectors, which can be mathematically compared. These vector representations of words are called **Word Embeddings**.

Implementing the Semantic Search System involves mainly two things:

### Dense Passage Retriever (DPR)

A Dense Passage Retriever consists of two models:

- Query Embedding Model: computes the embeddings for a query
- Passage Embedding Model: computes the embeddings for existing documents in the document store

### Document Store

We are using the FAISS (Facebook AI Similarity Search) Document Store. There are two components to it:

- sqlite database file: this contains the actual contents of the documents
- FAISS Index file: this contains the embeddings as computed by the Passage Embedding Model of the DPR

During an actual search, the DPR computes the embeddings of the query, and FAISS runs an algorithmic search of these against the existing passage embeddings, and returns the documents whose embeddings are the most similar.

## Question-Answering

QnA is accomplished by using a Reader model. We pass the top few results from the Semantic Search to the Reader, and the Reader just goes through these to find the most probable answer. The result is a short answer to the question, directly extracted from the top few documents from the search results.

## Query Classifier

We also use a Question vs Statement Classifier Model in between the Search and Reader. We invoke the QnA part only if the query is a question, as decided by the classifier.

---

## Model Evaluation and Selection

All the playground files can be found in this [Drive folder](https://drive.google.com/drive/folders/1ua1VEhVqN3kER8K6cLRIP6ieUnH0v3Ze?usp=sharing) (Needs an AlgoAnalytics email logged in to access). Refer to the ImprovedQnA Dataset folder for the evaluation notebooks.

The Stanford Question Answering Dataset (SQuAD) 2.0 is a a benchmark for QnA tasks. It consists of 150,000 questions over 500+ Wikipedia articles.

Many models are fine-tuned on SQuAD2.0 dataset for the purposes of QnA. To evaluate, I (Purva) created a custom dataset over a set of news documents.

### Custom Evaluation Dataset

This custom evaluation dataset has 258 questions over 171 documents. The annotation was performed with the [Haystack Annotation Tool](https://annotate.deepset.ai/).

Login Credentials:

- Email: pparmar@algoanalytics.com
- Password: `annotatingtools`

The project is named ImprovedNews in the Annotation Tool, because this improved upon an older existing evaluation dataset called News50, whose origins remain unclear.

### Evaluation Results

[Model Results](https://docs.google.com/spreadsheets/d/1lN2oe_W6x12f1ulYN74lxQiST5U8lp7xfQY34Eot0GU/edit?usp=sharing) sheet on Google Sheets lists all of the data that was generated from evaluation.

Sparse Retriever refers to keyword search ranking functions like BM25 and TF-IDF.

After some deliberations and considering the model performance and total runtime, the following were selected [the models are named in the HuggingFace naming convention]:

- Dense Passage Retriever
    - Query Embedding Model: `soheeyang/rdr-question_encoder-single-nq-base`
    - Passage Embedding Model: `soheeyang/rdr-ctx_encoder-single-nq-base`
- Reader: `deepset/minilm-uncased-squad2`
- Query Classifier: `shahrukhx01/question-vs-statement-classifier` (This was directly taken from the Haystack Documentation Page on [Query Classification](https://docs.haystack.deepset.ai/docs/query_classifier))

