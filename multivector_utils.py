import os
import pickle
from typing import List, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from CustomMVR import customMVR


def create_child_documents(parent_docs: List[Document], doc_ids: List[str], id_key: str) -> List[Document]:
    """
    Splits parent docs into child docs based on markdown headers and preserves metadata.
    """
    headers_to_split_on = [
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    child_text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

    sub_docs = []
    for i, doc in enumerate(parent_docs):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_text(doc.page_content) # Use split_text and pass the page_content
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
            # Add parent document metadata to the child document
            _doc.metadata.update(doc.metadata)
            _doc.metadata['parent_id'] = _id
        sub_docs.extend(_sub_docs)

    for item in sub_docs:
        item.metadata['source_type'] = 'Children'
    return sub_docs


def generate_summaries(parent_docs: List[Document], llm: ChatOpenAI, id_key: str, doc_ids: List[str]) -> List[Document]:
    """
    Generates summaries for the given parent documents using the provided LLM.
    """
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | llm
        | StrOutputParser()
    )

    summaries = chain.batch(parent_docs, {"max_concurrency": 5})
    summary_docs = [
        Document(page_content=s, metadata={"source_type": "summary", id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]
    return summary_docs


class HypotheticalQuestions(BaseModel):
    """Generate hypothetical questions."""
    questions: List[str] = Field(..., description="List of questions")


def generate_hypothetical_questions(parent_docs: List[Document], id_key: str, doc_ids: List[str]) -> List[Document]:
    """
    Generates hypothetical questions for each document and returns them as Document objects.
    """
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template(
            "Generate a list of exactly 3 hypothetical questions that the below document could be used to answer:\n\n{doc}"
        )
        | ChatOpenAI(max_retries=0, model="gpt-4o").with_structured_output(HypotheticalQuestions)
        | (lambda x: x.questions)
    )

    hypothetical_questions = chain.batch(parent_docs, {"max_concurrency": 5})
    question_docs = []
    for i, question_list in enumerate(hypothetical_questions):
        question_docs.extend(
            [Document(page_content=q, metadata={id_key: doc_ids[i]}) for q in question_list]
        )
    return question_docs


def create_MVR(parent_docs, doc_ids, vectorstore, filter_expression):
    """
    Create MultiVectorRetriever
    """
    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"

    # The Custom retriever (empty to start)
    retriever = customMVR(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,search_kwargs={"k": 5},
        filter_condition=filter_expression
    )
    retriever.docstore.mset(list(zip(doc_ids, parent_docs)))
    return retriever
