from langchain_community.document_loaders import PyMuPDFLoader
from rich.markdown import Markdown
from rich.console import Console
from langchain_ollama import ChatOllama

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

csl = Console()


def load_pdf(path):
    llm = ChatOllama(model="gemma4:e2b", reasoning=True, temperature=0.0)
    loader = PyMuPDFLoader(file_path=path)
    documents = loader.load()
    print("DOCUMENTS")
    print("\n\n")
    csl.print(documents)
    print("\n\n")
    

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""
            You are an expert AI assistant.

            Instructions:
            1. Answer ONLY using the provided context.
            2. Cite sources for every sentence using format with [page, source] -> page number ,source from documents meta data.
            3. If unsure, say "I don't know".
            4. Do not make up information.
            """),

        HumanMessagePromptTemplate.from_template("""
            Context:
            {documents}

            Question:
            {question}

            Provide a clear and well-structured answer with citations.
            """),
    ])

    chain = prompt | llm

    response = chain.invoke({
        "documents": documents,
        "question": "What is this document about?"
    })

    csl.print(Markdown(response.content))
if __name__ == "__main__":
    load_pdf("./docs/langchain_demo.pdf")