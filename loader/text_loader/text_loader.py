import os
import tempfile
from langchain_community.document_loaders import TextLoader
from rich.markdown import Markdown
from rich.console import Console
from langchain_ollama import ChatOllama

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

csl = Console()


def load_text_file():
    llm = ChatOllama(model="gemma4:e2b", reasoning=True, temperature=0.0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(
            b"Hello, this is a sample text file.\nThis file is used to demonstrate the TextLoader."
        )
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        context = "\n".join([
            f"[{i}] {doc.page_content}"
            for i, doc in enumerate(documents)
        ])

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
                You are an expert AI assistant.

                Instructions:
                1. Answer ONLY using the provided context.
                2. Cite sources for every sentence using [n] format.
                3. If unsure, say "I don't know".
                4. Do not make up information.

                At the end, include a Sources section listing:
                [n] -> document name or chunk id
                """),

            HumanMessagePromptTemplate.from_template("""
                Context:
                {context}

                Question:
                {question}

                Provide a clear and well-structured answer with citations.
                """),
        ])

        chain = prompt | llm

        response = chain.invoke({
            "context": context,
            "question": "What is this document about?"
        })

        csl.print(Markdown(response.content))

    finally:
        os.remove(temp_file_path)


if __name__ == "__main__":
    load_text_file()