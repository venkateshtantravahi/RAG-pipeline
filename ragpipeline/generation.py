"""
Generation Module (RAG)
-----------------------
Connects retrieved context with a Generative LLM (Ollama).
Features:
- Streaming Output (Low Latency)
- Strict Prompt Engineering (Hallucination Control)
"""

from typing import Generator, List

from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from ragpipeline.config import LLM_MODEL_NAME, RAG_SYSTEM_PROMPT


class RAGGenerator:
    def __init__(self):
        """Initializes the connection to the local Ollama instance."""
        logger.info(f"Initializing Generator with model: {LLM_MODEL_NAME}...")

        """
        Temperature in LLM models is a parameter that controls the 
        randomness and creativity of the output by adjusting the 
        probability of the next word the model selects. 
        A low temperature makes the output more focused, 
        deterministic, and predictable, while a high temperature 
        increases randomness, leading to more creative and diverse, 
        but potentially less coherent, text.
        """
        # 'temperature=0.2' makes the model factual and deterministic.
        # We use ChatOllama for easy local inference.
        self.llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0.2, keep_alive="5m")

        # Create the chain: Prompt -> LLM -> String Parser
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

        # Chain combines steps into one executable object
        self.chain = self.prompt | self.llm | StrOutputParser()

        logger.success("Generator initialized.")

    def format_docs(self, docs: List[Document]) -> str:
        """
        Augmentation Step: Turns a list of documents into a single text
        block. Adds Source Metadata so the LLM knows where info came from.
        """
        formatted_text = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            clean_content = doc.page_content.replace("\n", " ").strip()

            entry = f'<doc id="{i + 1}" source="{source}">\n{clean_content}\n</doc>'
            formatted_text.append(entry)

        return "\n\n".join(formatted_text)

    def generate_stream(
        self, query: str, retrieved_docs: List[Document]
    ) -> Generator[str, None, None]:
        """
        Generates the answer token-by-token (Streaming).
        This is critical for perceived speed in the UI.
        """
        if not retrieved_docs:
            yield "I could not find any relevant documents to answer your query."
            return

        # Augment
        context_block = self.format_docs(retrieved_docs)

        logger.info(
            f"Generating answer for: '{query}' (Context Length: {len(context_block)} chars)"
        )

        # Generate
        stream = self.chain.stream({"context": context_block, "question": query})

        for chunk in stream:
            yield chunk


if __name__ == "__main__":
    # --- END-TO-END TEST (Retrieval + Generation) ----
    # Run: python -m ragpipeline.generation
    from ragpipeline.retrieval import Retriever

    try:
        retriever = Retriever()
        generator = RAGGenerator()

        test_query = "What is Scaled Dot-Product Attention?"

        print(f"\n Question: {test_query}\n")

        docs = retriever.search(test_query, k=3)

        if not docs:
            print("No documents found. Aborting generation.")
        else:
            docs = [doc for doc, score in docs]
            print(f"Found {len(docs)} relevant chunks. Generating answer...\n")
            print("-" * 50)

            full_response = ""
            for token in generator.generate_stream(test_query, docs):
                print(token, end="", flush=True)
                full_response += token
            print("\n" + "-" * 50)

    except Exception as e:
        logger.error(f"RAG Pipeline encountered an exception: {e}")
