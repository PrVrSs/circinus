from pathlib import Path

import chromadb
from attrs import define
from chromadb.config import Settings
from llama_index import VectorStoreIndex
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores.types import VectorStore as _VectorStore


BASE = Path(__file__).parent


@define
class ContextFilter:
    doc_type: str
    docs_ids: list[str] | None


def _chromadb_doc_id_metadata_filter(context_filter: ContextFilter | None) -> dict | None:
    return {'doc_type': context_filter.doc_type}


class VectorStore:
    vector_store: _VectorStore

    def __init__(self) -> None:
        chroma_settings = Settings(anonymized_telemetry=False)
        chroma_client = chromadb.PersistentClient(
            path=str((BASE / 'chroma_db').absolute()),
            settings=chroma_settings,
        )
        chroma_collection = chroma_client.get_or_create_collection('make_this_parameterizable_per_api_call')
        self.vector_store = ChromaVectorStore(chroma_client=chroma_client, chroma_collection=chroma_collection)

    @staticmethod
    def get_retriever(
        index: VectorStoreIndex,
        context_filter: ContextFilter | None = None,
        similarity_top_k: int = 2,
    ) -> VectorIndexRetriever:
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            vector_store_kwargs={
                'where': _chromadb_doc_id_metadata_filter(context_filter)
            },
        )
