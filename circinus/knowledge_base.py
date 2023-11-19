import tempfile
from pathlib import Path
from typing import Any, AnyStr, Literal

from attrs import define, field
from llama_index import (
    Document,
    ServiceContext,
    StorageContext,
    StringIterableReader,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.base import BaseEmbedding
from llama_index.llms.base import LLM
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.readers.file.base import DEFAULT_FILE_READER_CLS
from llama_index.schema import NodeWithScore
from llama_index.storage.docstore import BaseDocumentStore
from llama_index.storage.index_store.types import BaseIndexStore

from circinus.components import (
    doc_store_component,
    embedding_component,
    index_store_component,
    llm_component,
)
from circinus.vector_store import ContextFilter, VectorStore


Base = (Path(__file__).parent.parent / 'data').resolve(strict=True)


@define
class DocumentDTO:
    object: Literal['ingest.document']
    doc_id: str
    doc_metadata: dict[str, Any]

    @staticmethod
    def curate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Remove unwanted metadata keys."""
        metadata.pop('doc_id', None)
        metadata.pop('window', None)
        metadata.pop('original_text', None)
        return metadata


@define
class ChunkDTO:
    object: Literal['context.chunk']
    score: float
    document: DocumentDTO
    text: str
    previous_texts: list[str] | None = field(default=None)
    next_texts: list[str] | None = field(default=None)

    @classmethod
    def from_node(cls: type['ChunkDTO'], node: NodeWithScore) -> 'ChunkDTO':
        doc_id = node.node.ref_doc_id if node.node.ref_doc_id is not None else '-'
        return cls(
            object='context.chunk',
            score=node.score or 0.0,
            document=DocumentDTO(
                object='ingest.document',
                doc_id=doc_id,
                doc_metadata=node.metadata,
            ),
            text=node.get_content(),
        )


def read_data(data):
    string_reader = StringIterableReader()
    if isinstance(data, Path):
        return string_reader.load_data([data.read_text()])
    if isinstance(data, bytes):
        return string_reader.load_data([data.decode('utf-8')])
    if isinstance(data, str):
        return string_reader.load_data([data])

    raise ValueError(f'Unsupported data type {type(data)}')


def read_data_by_reader(reader, data):
    if isinstance(data, Path):
        return reader.load_data(data)

    with tempfile.NamedTemporaryFile() as tmp:
        path_to_tmp = Path(tmp.name)
        if isinstance(data, bytes):
            path_to_tmp.write_bytes(data)
        else:
            path_to_tmp.write_text(str(data), encoding='utf-8')

        return reader.load_data(path_to_tmp)


def to_documents(extension, data) -> list[Document]:
    reader_cls = DEFAULT_FILE_READER_CLS.get(extension)
    if reader_cls is None:
        return read_data(data)

    return read_data_by_reader(reader=reader_cls(), data=data)


class KnowledgeBase:
    def __init__(
        self,
        llm: LLM,
        vector_store_component: VectorStore,
        embedding: BaseEmbedding,
        index_store: BaseIndexStore,
        document_store: BaseDocumentStore,
    ) -> None:
        self.llm = llm
        self.vector_store_component = vector_store_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=document_store,
            index_store=index_store,
        )
        self.service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embedding,
            node_parser=SentenceWindowNodeParser.from_defaults(),
        )

    def add(self, file_name: str, file_data: AnyStr | Path) -> list[DocumentDTO]:
        documents = to_documents(extension=Path(file_name).suffix, data=file_data)
        for document in documents:
            document.metadata['file_name'] = file_name

        return self._save_docs(documents)

    def _save_docs(self, documents: list[Document]) -> list[DocumentDTO]:
        for document in documents:
            document.metadata['doc_id'] = document.doc_id
            document.excluded_embed_metadata_keys = ['doc_id']
            document.excluded_llm_metadata_keys = ['file_name', 'doc_id', 'page_label']

        try:
            index = load_index_from_storage(
                storage_context=self.storage_context,
                service_context=self.service_context,
                store_nodes_override=True,
                show_progress=True,
            )
            for doc in documents:
                index.insert(doc)
        except ValueError:
            VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                service_context=self.service_context,
                store_nodes_override=True,
                show_progress=True,
            )

        self.storage_context.persist(persist_dir=str(Base))

        return [
            DocumentDTO(
                object='ingest.document',
                doc_id=document.doc_id,
                doc_metadata=DocumentDTO.curate_metadata(document.metadata),
            )
            for document in documents
        ]

    def _get_sibling_nodes_text(
        self,
        node_with_score: NodeWithScore,
        related_number: int,
        forward: bool = True,
    ) -> list[str]:
        explored_nodes_texts = []
        current_node = node_with_score.node
        for _ in range(related_number):
            if forward is True:
                explored_node_info = current_node.next_node
            else:
                explored_node_info = current_node.prev_node

            if explored_node_info is None:
                break

            explored_node = self.storage_context.docstore.get_node(explored_node_info.node_id)
            explored_nodes_texts.append(explored_node.get_content())
            current_node = explored_node

        return explored_nodes_texts

    def retrieve(
        self,
        text: str,
        context_filter: ContextFilter | None = None,
        limit: int = 10,
        prev_next_chunks: int = 0,
    ) -> list[ChunkDTO]:
        index = VectorStoreIndex.from_vector_store(
            self.vector_store_component.vector_store,
            storage_context=self.storage_context,
            service_context=self.service_context,
            show_progress=True,
        )
        vector_index_retriever = self.vector_store_component.get_retriever(
            index=index,
            context_filter=context_filter,
            similarity_top_k=limit,
        )
        nodes = vector_index_retriever.retrieve(text)
        nodes.sort(key=lambda n: n.score or 0.0, reverse=True)

        retrieved_nodes = []
        for node in nodes:
            chunk = ChunkDTO.from_node(node)
            chunk.previous_texts = self._get_sibling_nodes_text(node, prev_next_chunks, False)
            chunk.next_texts = self._get_sibling_nodes_text(node, prev_next_chunks)
            retrieved_nodes.append(chunk)

        return retrieved_nodes


def init_knowledge_base(config) -> KnowledgeBase:
    return KnowledgeBase(
        llm=llm_component(config=config),
        vector_store_component=VectorStore(),
        embedding=embedding_component(config=config),
        document_store=doc_store_component(),
        index_store=index_store_component(),
    )
