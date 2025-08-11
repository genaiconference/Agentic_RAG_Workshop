from pydantic import model_validator
from langchain_core.documents import Document
from langchain_core.stores import ByteStore, BaseStore
from langchain.storage._lc_store import create_kv_docstore

from langchain.storage import InMemoryByteStore
import yaml
import pickle
from langchain_core.vectorstores import VectorStore
from typing import Any, Dict, List, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
)


class customMVR(BaseRetriever):
    vectorstore: VectorStore
    byte_store: Optional[ByteStore] = None
    docstore: BaseStore[str, Document]
    id_key: str = "doc_id"
    filter_condition: str

    @model_validator(mode="before")
    @classmethod
    def shim_docstore(cls, values: Dict) -> Any:
        byte_store = values.get("byte_store")
        docstore = values.get("docstore")
        if byte_store is not None:
            docstore = create_kv_docstore(byte_store)
        elif docstore is None:
            raise Exception("You must pass a `byte_store` parameter.")
        values["docstore"] = docstore
        return values

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        This method will return child documents
        """
        if self.filter_condition:
            sub_docs = self.vectorstore.semantic_hybrid_search(
                query=query,
                filters=self.filter_condition
            )
        else:
            sub_docs = self.vectorstore.semantic_hybrid_search(query=query)

        # sub_docs = self._get_relevant_documents(query) # - child docs

        ids = [doc.metadata['doc_id'] for doc in sub_docs]

        return self.docstore.mget(ids)

