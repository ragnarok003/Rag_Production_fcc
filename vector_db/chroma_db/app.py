import chromadb
from rich import print
chroma_client =  chromadb.Client()
collection_name ="test_collection"

collection = chroma_client.get_or_create_collection(collection_name)

documents =[
    {"id":"doc1","text":"Hello, world!"},
    {"id":"doc2","text":"How are you today ?"},
    {"id":"doc3","text":"Good Bye see you later!"},
]

for doc in documents:
    collection.upsert(ids=doc['id'],documents=doc['text'])

query ="Hello, World!"

result= collection.query(query_texts=[query],n_results=3)

print(result)