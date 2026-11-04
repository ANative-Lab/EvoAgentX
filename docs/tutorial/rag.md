# Build Your First RAG System with RAGEngine

In EvoAgentX, the `RAGEngine` is a powerful tool for building Retrieval-Augmented Generation (RAG) systems. It allows you to load documents, create searchable indices, and retrieve relevant information to answer questions. This tutorial is designed for beginners and will guide you through the essential steps to create and use a RAG system with `RAGEngine`:

1. **Setting Up RAGEngine**: Learn how to configure and initialize the engine.
2. **Indexing and Querying Documents**: Load a simple dataset and query it to find answers.
3. **Saving and Loading Indices**: Save your indexed data and reuse it later.

By the end of this tutorial, you’ll be able to set up a basic RAG system, index documents, query them, and persist your work for future use. We’ll use a sample from the HotPotQA dataset to make it practical and fun!

## 1. Setting Up RAGEngine

The first step is to set up your environment and initialize `RAGEngine`. This involves installing dependencies, configuring the storage backend, and setting up the embedding model.

### Install Dependencies
Ensure you have EvoAgentX installed. You’ll also need an OpenAI API key for embeddings. Run the following in your terminal:

```bash
pip install evoagentx llama_index pydantic python-dotenv
```

Create a `.env` file in your project directory and add your OpenAI API key:

```plaintext
OPENAI_API_KEY=your-openai-api-key
```

### Configure the Environment
Let’s write a Python script to set up `RAGEngine`. We’ll use SQLite for metadata storage and FAISS for vector embeddings, which are beginner-friendly options.

```python
import os
from dotenv import load_dotenv
from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, EmbeddingConfig, IndexConfig, RetrievalConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure storage (SQLite for metadata, FAISS for vectors)
store_config = StoreConfig(
    dbConfig=DBConfig(db_name="sqlite", path="./data/cache.db"),
    vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1536, index_type="flat_l2"),
    graphConfig=None,
    path="./data/indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# Configure RAGEngine
rag_config = RAGConfig(
    reader=ReaderConfig(recursive=False, exclude_hidden=True),
    chunker=ChunkerConfig(strategy="simple", chunk_size=512, chunk_overlap=50),
    embedding=EmbeddingConfig(provider="openai", model_name="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    index=IndexConfig(index_type="vector"),
    retrieval=RetrievalConfig(retrieval_type="vector", postprocessor_type="simple", top_k=3, similarity_cutoff=0.3)
)

# Initialize RAGEngine
rag_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)

print("RAGEngine is ready to go!")
```

### What’s Happening Here?
- **Environment Setup**: We load the OpenAI API key from the `.env` file.
- **Storage Configuration**: We set up SQLite to store metadata and FAISS to store vector embeddings. The `dimensions=1536` matches the OpenAI embedding model.
- **RAG Configuration**: We configure the pipeline:
  - `ReaderConfig`: Reads files (we’ll use a JSON file later).
  - `ChunkerConfig`: Splits documents into 512-character chunks with 50-character overlap.
  - `EmbeddingConfig`: Uses OpenAI’s `text-embedding-ada-002` for embeddings.
  - `IndexConfig`: Creates a vector index.
  - `RetrievalConfig`: Retrieves the top 3 most similar chunks with a similarity score above 0.3.
- **Initialization**: We create the `RAGEngine` instance, ready to process documents.

To use Milvus instead of FAISS, install the optional dependency and replace the vector store configuration:

```bash
pip install "evoagentx[milvus]"
```

```python
vectorConfig=VectorStoreConfig(
    vector_name="milvus",
    dimensions=1536,
    milvus_uri="./data/evoagentx_milvus.db",
    milvus_collection_name="rag_tutorial_vectors",
    milvus_metric_type="IP",
)
```

Use a Milvus Lite `.db` path for local development, or set `milvus_uri` and `milvus_token` for Milvus server or Zilliz Cloud.

For more details on configuration, see the [RAGEngine documentation](../modules/rag.md).

## 2. Indexing and Querying Documents

Now, let’s index a document from the HotPotQA dataset and query it. We’ll use the first HotPotQA example you provided, which contains information about Scott Derrickson and Ed Wood, and answer the question: “Were Scott Derrickson and Ed Wood of the same nationality?”

### Prepare the Dataset
Create a directory called `data` and save the HotPotQA example as `hotpotqa_sample.json`:

```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "answer": "yes",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "supporting_facts": [
    [
      "Scott Derrickson",
      0
    ],
    [
      "Ed Wood",
      0
    ]
  ],
  "context": [
    [
      "Ed Wood (film)",
      [
        "Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood.",
        " The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau.",
        " Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast."
      ]
    ],
    [
      "Scott Derrickson",
      [
        "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
        " He lives in Los Angeles, California.",
        " He is best known for directing horror films such as \"Sinister\", \"The Exorcism of Emily Rose\", and \"Deliver Us From Evil\", as well as the 2016 Marvel Cinematic Universe installment, \"Doctor Strange.\""
      ]
    ],
    [
      "Woodson, Arkansas",
      [
        "Woodson is a census-designated place (CDP) in Pulaski County, Arkansas, in the United States.",
        " Its population was 403 at the 2010 census.",
        " It is part of the Little Rock\u2013North Little Rock\u2013Conway Metropolitan Statistical Area.",
        " Woodson and its accompanying Woodson Lake and Wood Hollow are the namesake for Ed Wood Sr., a prominent plantation owner, trader, and businessman at the turn of the 20th century.",
        " Woodson is adjacent to the Wood Plantation, the largest of the plantations own by Ed Wood Sr."
      ]
    ],
    [
      "Tyler Bates",
      [
        "Tyler Bates (born June 5, 1965) is an American musician, music producer, and composer for films, television, and video games.",
        " Much of his work is in the action and horror film genres, with films like \"Dawn of the Dead, 300, Sucker Punch,\" and \"John Wick.\"",
        " He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn.",
        " With Gunn, he has scored every one of the director's films; including \"Guardians of the Galaxy\", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel.",
        " In addition, he is also the lead guitarist of the American rock band Marilyn Manson, and produced its albums \"The Pale Emperor\" and \"Heaven Upside Down\"."
      ]
    ],
    [
      "Ed Wood",
      [
        "Edward Davis Wood Jr. (October 10, 1924 \u2013 December 10, 1978) was an American filmmaker, actor, writer, producer, and director."
      ]
    ],
    [
      "Deliver Us from Evil (2014 film)",
      [
        "Deliver Us from Evil is a 2014 American supernatural horror film directed by Scott Derrickson and produced by Jerry Bruckheimer.",
        " The film is officially based on a 2001 non-fiction book entitled \"Beware the Night\" by Ralph Sarchie and Lisa Collier Cool, and its marketing campaign highlighted that it was \"inspired by actual accounts\".",
        " The film stars Eric Bana, \u00c9dgar Ram\u00edrez, Sean Harris, Olivia Munn, and Joel McHale in the main roles and was released on July 2, 2014."
      ]
    ],
    [
      "Adam Collis",
      [
        "Adam Collis is an American filmmaker and actor.",
        " He attended the Duke University from 1986 to 1990 and the University of California, Los Angeles from 2007 to 2010.",
        " He also studied cinema at the University of Southern California from 1991 to 1997.",
        " Collis first work was the assistant director for the Scott Derrickson's short \"Love in the Ruins\" (1995).",
        " In 1998, he played \"Crankshaft\" in Eric Koyanagi's \"Hundred Percent\"."
      ]
    ],
    [
      "Sinister (film)",
      [
        "Sinister is a 2012 supernatural horror film directed by Scott Derrickson and written by Derrickson and C. Robert Cargill.",
        " It stars Ethan Hawke as fictional true-crime writer Ellison Oswalt who discovers a box of home movies in his attic that puts his family in danger."
      ]
    ],
    [
      "Conrad Brooks",
      [
        "Conrad Brooks (born Conrad Biedrzycki on January 3, 1931 in Baltimore, Maryland) is an American actor.",
        " He moved to Hollywood, California in 1948 to pursue a career in acting.",
        " He got his start in movies appearing in Ed Wood films such as \"Plan 9 from Outer Space\", \"Glen or Glenda\", and \"Jail Bait.\"",
        " He took a break from acting during the 1960s and 1970s but due to the ongoing interest in the films of Ed Wood, he reemerged in the 1980s and has become a prolific actor.",
        " He also has since gone on to write, produce and direct several films."
      ]
    ],
    [
      "Doctor Strange (2016 film)",
      [
        "Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name, produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures.",
        " It is the fourteenth film of the Marvel Cinematic Universe (MCU).",
        " The film was directed by Scott Derrickson, who wrote it with Jon Spaihts and C. Robert Cargill, and stars Benedict Cumberbatch as Stephen Strange, along with Chiwetel Ejiofor, Rachel McAdams, Benedict Wong, Michael Stuhlbarg, Benjamin Bratt, Scott Adkins, Mads Mikkelsen, and Tilda Swinton.",
        " In \"Doctor Strange\", surgeon Strange learns the mystic arts after a career-ending car accident."
      ]
    ]
  ],
  "type": "comparison",
  "level": "hard"
}
```

### Index the Document
Let’s write a script to load the JSON file, index its content, and query it. Create a file called `rag_tutorial.py`:

```python
import os
import json

from dotenv import load_dotenv

from evoagentx.rag.schema import Query
from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, EmbeddingConfig, IndexConfig, RetrievalConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure storage (SQLite for metadata, FAISS for vectors)
store_config = StoreConfig(
    dbConfig=DBConfig(db_name="sqlite", path="./data/cache.db"),
    vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1536, index_type="flat_l2"),
    graphConfig=None,
    path="./data/indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# Configure RAGEngine
rag_config = RAGConfig(
    reader=ReaderConfig(recursive=False, exclude_hidden=True),
    chunker=ChunkerConfig(strategy="simple", chunk_size=512, chunk_overlap=50),
    embedding=EmbeddingConfig(provider="openai", model_name="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    index=IndexConfig(index_type="vector"),
    retrieval=RetrievalConfig(retrieval_type="vector", postprocessor_type="simple", top_k=3, similarity_cutoff=0.3)
)

# Initialize RAGEngine
rag_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)

print("RAGEngine is ready to go!")

# Step 1: Load and index the HotPotQA sample
with open("./data/hotpotqa_sample.json", "r", encoding="utf-8") as f:
    hotpotqa_data = json.load(f)

# Create a temporary text file with the context for indexing
context = hotpotqa_data["context"]
with open("./data/hotpotqa_context.txt", "w", encoding="utf-8") as f:
    for title, sentences in context:
        f.write(f"# {title}")
        for sentence in sentences:
            f.write(f"{sentence}\n")
        f.write(f"\n\n")

# Index the text file
corpus = rag_engine.read(
    file_paths="./data/hotpotqa_context.txt",
    filter_file_by_suffix=[".txt"],
    merge_by_file=True,
    show_progress=True,
    corpus_id="hotpotqa_corpus"
)
rag_engine.add(index_type="vector", nodes=corpus, corpus_id="hotpotqa_corpus")

print("Documents indexed successfully!")

# Step 2: Query the index
query = Query(query_str="Were Scott Derrickson and Ed Wood of the same nationality?", top_k=3)
result = rag_engine.query(query, corpus_id="hotpotqa_corpus")

# Print the retrieved chunks
print("\nRetrieved answers:")
for i, chunk in enumerate(result.corpus.chunks, 1):
    print(f"{i}. {chunk.text}")

# Clean up
rag_engine.clear(corpus_id="hotpotqa_corpus")
```

### Run the Script
Run the script:

```bash
python rag_tutorial.py
```

### What’s Happening Here?
- **Loading the Data**: We read the HotPotQA JSON file and extract its `context` field, which contains text about Scott Derrickson and Ed Wood.
- **Indexing**:
  - We create a temporary text file (`hotpotqa_context.txt`) with the context, formatted for readability.
  - The `read` method loads the text file into a `Corpus`, splitting it into chunks (based on the 512-character limit set earlier).
  - The `add` method indexes the chunks into a vector index using OpenAI embeddings.
- **Querying**:
  - We create a `Query` object with the question from the dataset.
  - The `query` method retrieves the top 3 chunks most similar to the question.
  - The results include sentences indicating both individuals are American (e.g., “Scott Derrickson… is an American director” and “Edward Davis Wood Jr… was an American filmmaker”).
- **Cleanup**: We clear the index to free up memory.

### Expected Output
You should see something like:

```
Documents indexed successfully!

Retrieved answers:
1. # Ed Wood (film)Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood.
 The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau. 
 Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast.


# Scott DerricksonScott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.
 He lives in Los Angeles, California.
 He is best known for directing horror films such as "Sinister", "The Exorcism of Emily Rose", and "Deliver Us From Evil", as well as the 2016 Marvel Cinematic Universe installment, "Doctor Strange."


# Woodson, ArkansasWoodson is a census-designated place (CDP) in Pulaski County, Arkansas, in the United States.
 Its population was 403 at the 2010 census.
 It is part of the Little Rock–North Little Rock–Conway Metropolitan Statistical Area.
 Woodson and its accompanying Woodson Lake and Wood Hollow are the namesake for Ed Wood Sr., a prominent plantation owner, trader, and businessman at the turn of the 20th century.
 Woodson is adjacent to the Wood Plantation, the largest of the plantations own by Ed Wood Sr.


# Tyler BatesTyler Bates (born June 5, 1965) is an American musician, music producer, and composer for films, television, and video games.
 Much of his work is in the action and horror film genres, with films like "Dawn of the Dead, 300, Sucker Punch," and "John Wick."
 He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn.
 With Gunn, he has scored every one of the director's films; including "Guardians of the Galaxy", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel.
2. With Gunn, he has scored every one of the director's films; including "Guardians of the Galaxy", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel.
 In addition, he is also the lead guitarist of the American rock band Marilyn Manson, and produced its albums "The Pale Emperor" and "Heaven Upside Down".  


# Ed WoodEdward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.


# Deliver Us from Evil (2014 film)Deliver Us from Evil is a 2014 American supernatural horror film directed by Scott Derrickson and produced by Jerry Bruckheimer.
 The film is officially based on a 2001 non-fiction book entitled "Beware the Night" by Ralph Sarchie and Lisa Collier Cool, and its marketing campaign highlighted that it was "inspired by actual accounts".
 The film stars Eric Bana, Édgar Ramírez, Sean Harris, Olivia Munn, and Joel McHale in the main roles and was released on July 2, 2014.


# Adam CollisAdam Collis is an American filmmaker and actor.
 He attended the Duke University from 1986 to 1990 and the University of California, Los Angeles from 2007 to 2010.
 He also studied cinema at the University of Southern California from 1991 to 1997.
 Collis first work was the assistant director for the Scott Derrickson's short "Love in the Ruins" (1995).
 In 1998, he played "Crankshaft" in Eric Koyanagi's "Hundred Percent".


# Sinister (film)Sinister is a 2012 supernatural horror film directed by Scott Derrickson and written by Derrickson and C. Robert Cargill.
 It stars Ethan Hawke as fictional true-crime writer Ellison Oswalt who discovers a box of home movies in his attic that puts his family in danger.
3. It stars Ethan Hawke as fictional true-crime writer Ellison Oswalt who discovers a box of home movies in his attic that puts his family in danger.       


# Conrad BrooksConrad Brooks (born Conrad Biedrzycki on January 3, 1931 in Baltimore, Maryland) is an American actor.
 He moved to Hollywood, California in 1948 to pursue a career in acting.
 He got his start in movies appearing in Ed Wood films such as "Plan 9 from Outer Space", "Glen or Glenda", and "Jail Bait."
 He took a break from acting during the 1960s and 1970s but due to the ongoing interest in the films of Ed Wood, he reemerged in the 1980s and has become a prolific actor.
 He also has since gone on to write, produce and direct several films.


# Doctor Strange (2016 film)Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name, produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures.
 It is the fourteenth film of the Marvel Cinematic Universe (MCU).
 The film was directed by Scott Derrickson, who wrote it with Jon Spaihts and C. Robert Cargill, and stars Benedict Cumberbatch as Stephen Strange, along with Chiwetel Ejiofor, Rachel McAdams, Benedict Wong, Michael Stuhlbarg, Benjamin Bratt, Scott Adkins, Mads Mikkelsen, and Tilda Swinton.
 In "Doctor Strange", surgeon Strange learns the mystic arts after a career-ending car accident.
```

The top1 in these chunks confirm that both Scott Derrickson and Ed Wood are American.

For more details on indexing and querying, see the [RAGEngine documentation](../modules/rag.md).

## 3. Saving and Loading Indices

Once you’ve indexed your documents, you can save the index to reuse it later without reprocessing the data. This is useful for large datasets or production environments.

### Save the Index
Modify the `rag_tutorial.py` script to save the index after indexing. Add the following line after `rag_engine.add`:

```python
# Save the index to disk
rag_engine.save(output_path="./data/indexing", corpus_id="hotpotqa_corpus", index_type="vector")
print("Index saved to ./data/indexing")
```

### Load the Index
To load the saved index and query it, create a new script called `rag_load.py`:

```python
import os
import json

from dotenv import load_dotenv

from evoagentx.rag.schema import Query
from evoagentx.rag.rag import RAGEngine
from evoagentx.rag.rag_config import RAGConfig, ReaderConfig, ChunkerConfig, EmbeddingConfig, IndexConfig, RetrievalConfig
from evoagentx.storages.base import StorageHandler
from evoagentx.storages.storages_config import StoreConfig, VectorStoreConfig, DBConfig

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure storage (SQLite for metadata, FAISS for vectors)
store_config = StoreConfig(
    dbConfig=DBConfig(db_name="sqlite", path="./data/cache.db"),
    vectorConfig=VectorStoreConfig(vector_name="faiss", dimensions=1536, index_type="flat_l2"),
    graphConfig=None,
    path="./data/indexing"
)
storage_handler = StorageHandler(storageConfig=store_config)

# Configure RAGEngine
rag_config = RAGConfig(
    reader=ReaderConfig(recursive=False, exclude_hidden=True),
    chunker=ChunkerConfig(strategy="simple", chunk_size=512, chunk_overlap=50),
    embedding=EmbeddingConfig(provider="openai", model_name="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    index=IndexConfig(index_type="vector"),
    retrieval=RetrievalConfig(retrieval_type="vector", postprocessor_type="simple", top_k=3, similarity_cutoff=0.3)
)

# Initialize RAGEngine
rag_engine = RAGEngine(config=rag_config, storage_handler=storage_handler)

print("RAGEngine is ready to go!")

# Load the saved index
rag_engine.load(source="./data/indexing", corpus_id="hotpotqa_corpus", index_type="vector")
print("Index loaded successfully!")

# Query the loaded index
query = Query(query_str="Were Scott Derrickson and Ed Wood of the same nationality?", top_k=3)
result = rag_engine.query(query, corpus_id="hotpotqa_corpus")

# Print the retrieved chunks
print("\nRetrieved answers:")
for i, chunk in enumerate(result.corpus.chunks, 1):
    print(f"{i}. {chunk.text}")

# Clean up
rag_engine.clear(corpus_id="hotpotqa_corpus")
```

### Run the Script
Run the script:

```bash
python rag_load.py
```

### What’s Happening Here?
- **Saving**: The `save` method stores the vector index and metadata to the `./data/indexing` directory.
- **Loading**: The `load` method reconstructs the index from the saved files, making it ready for querying without re-indexing.
- **Querying**: We run the same query as before, and the results should be identical.
- **Cleanup**: We clear the index to keep things tidy.

### Expected Output
The output should be the same as in Section 2, confirming that the loaded index works correctly.

For more details on saving and loading, see the [RAGEngine documentation](../modules/rag.md).

## Next Steps
Congratulations! You’ve built your first RAG system with `RAGEngine`. Here are some ideas to explore next:
- Try indexing a larger dataset or different file types (e.g., PDFs).
- Experiment with different chunk sizes or embedding models (e.g., Hugging Face).
- Integrate `RAGEngine` with an EvoAgentX agent to answer questions automatically.

For a complete example, refer to the [RAGEngine example](../../examples/rag.py).

Happy building with EvoAgentX!
