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
    docs_ids: list[str] | None


def _chromadb_doc_id_metadata_filter(context_filter: ContextFilter | None) -> dict | None:
    if context_filter is None or context_filter.docs_ids is None:
        return {}
    if len(context_filter.docs_ids) < 1:
        return {'doc_id': '-'}

    doc_filter_items = []
    if len(context_filter.docs_ids) > 1:
        doc_filter = {'$or': doc_filter_items}
        for doc_id in context_filter.docs_ids:
            doc_filter_items.append({'doc_id': doc_id})
    else:
        doc_filter = {'doc_id': context_filter.docs_ids[0]}

    return doc_filter


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
