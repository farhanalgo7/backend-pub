# Semantic Search and QnA

This is the Semantic Search + Question-Answering (QnA) application.

## Running the App

Create a virtual environment and activate it. Then, install the requirements.

`$ python -m pip install -r requirements.txt`

If running for the first time, see Usage section below. Otherwise, simply run the app.

`$ python app.py`

Now, you can open `locahost:8000` on browser. Check the docs on `localhost:8000/docs` to see the APIs and try them out.

The primary API is `/search/{search-string}`. So, for example, you can query `/search/TCS` and get a JSON response.

### Docker

With Docker, you can build and run the app as follows:

`$ docker build -t semanticqna:latest .`

`$ docker run -p 8000:8000 --name SemanticQnA semanticqna:latest`

The app should be running on Port 8000.

## Usage

### Create Models and Resources

- The `create_models.py` program creates a `data/` folder, containing document stores and models.
- If running for the first time, run `$ python create_models.py`

### Fetch data

- The FAISS Document Store needs to fetch data from CosmosDB and update retriever embeddings. The `fetch_from_DB.py` program accomplishes this. By default, it works on fetching data from yesterday onwards. Otherwise, it also accepts two command line arguments, `--from` and `--to` to mention the dates to fetch for in YYYY-MM-DD format.

- Run `$ python fetch_from_DB.py`

### Removing older documents

- We can remove documents older than 30 days by simply running the `remove_older_docs.py` program.
- `$ python remove_older_docs.py`

---

## Building Documentation

There are some documentation content pages in the `docs/` folder, which can be converted to a site with [MkDocs](https://www.mkdocs.org/). 

`python -m mkdocs build`

This will build the pages into a `site/` folder. However, the pages are in Markdown and can be perfectly viewed in any other Markdown viewer as well. GitHub displays them properly, for instance.