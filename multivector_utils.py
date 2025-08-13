import os
import uuid
import pickle
from typing import List, Any, Optional
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
import chunking_utils


def dump_pickle_file(retriever, filename: str):
    """
    Serialize and save the given data to a file using Pickle.
    Args:
        data: The data to be pickled.
        filename (str): The name of the file where the pickled data will be saved.
    Returns:
    A pickle file with object saved in it
    """
    with open(filename, 'wb') as handle:
        pickle.dump(retriever, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_file(filename: str):
    """
    De-Serialize file using Pickle.
    Args:
        filename (str): The name of the file where the pickled data will be saved.
    Returns:
    Pickle Object
    """
    with open(filename, "rb") as input_file:
        pickle_object = pickle.load(input_file)
    return pickle_object


def load_DI_output(di_results_filename: str):
    """
    Loads the DI output pickle file.
    """
    DI_OUTPUT_DIR = os.path.join(os.getcwd(), "DI_output")
    filepath = os.path.join(DI_OUTPUT_DIR, di_results_filename)
    with open(filepath, "rb") as f:
        di_results = pickle.load(f)
    return di_results


def create_parent_docs(content, llm):
    # Initialize the MarkdownHeaderTextSplitter with custom headers
    parent_headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2")
    ]

    # Create splitter with current configuration
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=parent_headers_to_split_on,
        strip_headers=True
    )

    # Split the document
    split_docs = splitter.split_text(content)
    return split_docs


def generate_final_parents(md_result, full_text_with_images, source_file_name, llm):
    """
    Generate parent document chunks and metadata.
    """
    parent_docs = create_parent_docs(full_text_with_images, llm)

    parent_docs = chunking_utils.assign_page_numbers_to_parent_docs(md_result, parent_docs)

    for doc in parent_docs:
        chunking_utils.create_custom_metadata_for_all_sources(doc)
    
    parent_docs = chunking_utils.add_source_metadata(parent_docs, source_file_name)
    
    final_parents = chunking_utils.append_custom_metadata(parent_docs)
    
    # Create doc ids for parent docs
    doc_ids = [str(uuid.uuid4()) for _ in final_parents]

    # store final_parents & doc_ids in dictionary
    parent_dict = {"doc_ids":doc_ids, "parent_docs":final_parents}

    #save to pickle
    dump_pickle_file(parent_dict, source_file_name+".pkl")
    return final_parents, doc_ids



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


def generate_hypothetical_questions(parent_docs: List[Document], llm: ChatOpenAI, id_key: str, doc_ids: List[str]) -> List[Document]:
    """
    Generates hypothetical questions for each document and returns them as Document objects.
    """
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template(
            "Generate a list of exactly 3 hypothetical questions that the below document could be used to answer:\n\n{doc}"
        )
        | llm.with_structured_output(HypotheticalQuestions)
        | (lambda x: x.questions)
    )

    hypothetical_questions = chain.batch(parent_docs, {"max_concurrency": 5})
    question_docs = []
    for i, question_list in enumerate(hypothetical_questions):
        question_docs.extend(
            [Document(page_content=q, metadata={id_key: doc_ids[i]}) for q in question_list]
        )
    return question_docs


def create_chroma_vectordb(embeddings, vector_db_name, sub_docs, summary_docs, question_docs):
    persist_directory = os.path.join(os.getcwd(), vector_db_name)

    # The vectorstore to use to index the child chunks (creating an empty vectorstore for now)
    vector_store = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    vector_store.add_documents(sub_docs)
    vector_store.add_documents(summary_docs)
    vector_store.add_documents(question_docs)
    return

def create_MVR(parent_docs, doc_ids, vectorstore):
    """
    Create MultiVectorRetriever
    """
    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"

    # The Custom retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,search_kwargs={"k": 5},

    )
    retriever.docstore.mset(list(zip(doc_ids, parent_docs)))
    return retriever


def create_ensemble_retriever_with_bm25(parent_docs, mvr_retriever):
    bm25_retriever = BM25Retriever.from_documents(parent_docs)
    # initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, mvr_retriever], weights=[0.5, 0.5]
    )
    return ensemble_retriever


def create_retriever_pipeline(
    di_results_filename: str,
    source_file_name: str,
    vector_db_name: str,
    embeddings_model: Optional[OpenAIEmbeddings] = None,
    llm_model: Optional[ChatOpenAI] = None,
    vectorstore_exists=False
):
    """
    Full pipeline to create and return a MultiVectorRetriever.
    Steps:
    1. Load DI output
    2. Generate parent docs (via generate_final_parents)
    3. Load parent docs & doc_ids from pickle
    4. Generate child docs, summaries, and hypothetical questions
    5. Create Chroma vector DB
    6. Create MultiVectorRetriever
    """
    persist_dir = os.path.join(os.getcwd(), vector_db_name)

    if not vectorstore_exists:
      # --- Step 1: Load DI output ---
      di_output = load_DI_output(di_results_filename)
      md_result = di_output["md_result"]
      md_result_with_images = di_output['result_with_image_descp']

      # --- Step 2: Generate parent docs ---
      parent_docs, doc_ids = generate_final_parents(
          md_result=md_result,
          full_text_with_images=md_result_with_images,
          source_file_name=source_file_name,
          llm=llm_model
      )

      # --- Step 3: Create child docs ---
      sub_docs = create_child_documents(parent_docs, doc_ids, id_key="doc_id")

      # --- Step 4: Generate summaries ---
      summary_docs = generate_summaries(parent_docs, llm_model, id_key="doc_id", doc_ids=doc_ids)

      # --- Step 5: Generate hypothetical questions ---
      question_docs = generate_hypothetical_questions(parent_docs, llm_model, id_key="doc_id", doc_ids=doc_ids)

      # --- Step 6: Create Chroma DB ---
      vector_store = Chroma(
          collection_name="documents",
          embedding_function=embeddings_model,
          persist_directory=persist_dir
      )
      vector_store.add_documents(sub_docs)
      vector_store.add_documents(summary_docs)
      vector_store.add_documents(question_docs)
    else:
      # --- Step 7: Load Parent docs & Doc ids ---
      parent_dict = load_pickle_file(os.getcwd() + "/parent_pickles/" + source_file_name + "_parents.pkl")
      parent_docs = parent_dict["parent_docs"]
      doc_ids = parent_dict["doc_ids"]

      # --- Step 8: Define exsiting vectorstore ---
      vector_store = Chroma(collection_name="documents",
      embedding_function=embeddings_model,
      persist_directory=persist_dir
      )

    # --- Step 9: Create MultiVectorRetriever ---
    mvr_retriever = create_MVR(parent_docs, doc_ids, vector_store)

    # ---Step 10: Create EnsembleRetriever ---
    ensemble_retriever = create_ensemble_retriever_with_bm25(parent_docs, mvr_retriever)

    return ensemble_retriever
