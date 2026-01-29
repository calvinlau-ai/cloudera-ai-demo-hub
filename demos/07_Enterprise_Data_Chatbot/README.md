# Enterprise Data Chatbot

* RAG architecture.
* Use Pinecone as the vector database.

## Milestone 1:
* Only use the code in the AMP code w/o any improvement.
* Retrieve *k* most related paragraphs, and generate the *k* responses one by one.
* Cli interface.
* Data: A random news data set.

## Milestone 2:
* Running as a CML application.
* Web UI with Gradio.
* Retrieve *k* most related paragraphs, and generate one most appropriate response based on these *k* paragraphs altogether.
* Data: Changed to a famous financial news data. https://huggingface.co/datasets/ashraq/financial-news-articles

## Milestone 3: Current Implementation
* Multilingual support (English, Chinese, and Indonesia language for this version).
* Rerank capabilities, which greatly improves the accuracy of the responses.
* Customised stop-criteria.
* Add support with the on-premise vector database Milvus.

## Milestone 4: WIP
* RAG Fusion -> Multi-Query Retrieval
