from langchain_chroma import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from rich import print

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

documents = [
    Document(
        page_content="""Product SKU-7742X is our flagship router. It supports gigabit speeds and advanced QoS features.""",
        metadata={"type": "product"},
    ),
    Document(
        page_content="For network connectivity issues, first check the ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server rejected the connection. Check firewall settings.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="The authentication process requires valid credentials.Use OAuth2 for secure API access.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel at 192.168.1.1 to modify settings.",
        metadata={"type": "config"},
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have alt text and sufficient color contrast.",
        metadata={"type": "compliance"},
    ),
]

print(f"{len(documents)} documents")


vectorstore = Chroma.from_documents(
    documents=documents, embedding=embeddings, collection_name="Hybrid_test"
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Vector retriever ready")

bm25_retriever = BM25Retriever.from_documents(documents=documents, k=3)
print("BM25 retriever ready")

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
)
print("Hybrid retriever ready")


def test_query(query, name, retriever):
    """Test a query and show results"""
    results = retriever.invoke(query)
    print(f'\\n{name} - Query: "{query}"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + "..."
        print(f" {i + 1}. {preview}")
    return results


# Test queries designed to challenge vector search I
test_queries = [
    "SKU-7742X specifications",  # Exact product code
    "E_CONN_REFUSED error",  # Error code
    "How do I authenticate?",  # Semantic question
    "WCAG compliance",  # Acronym
    "router configuration",  # General semantic
]

for query in test_queries:
    print("=" * 120)
    # Vector only
    vector_results = test_query(query, "VECTOR", vector_retriever)
    # BM25 only
    bm25_results = test_query(query, "BM25", bm25_retriever)
    # Hybrid
    hybrid_results = test_query(query, "HYBRID", ensemble_retriever)
