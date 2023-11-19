from functools import partial
from pathlib import Path

from llama_index import MockEmbedding, OpenAIEmbedding
from llama_index.embeddings.base import BaseEmbedding
from llama_index.llms import MockLLM, OpenAI
from llama_index.llms.base import LLM
from llama_index.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.storage.index_store.types import BaseIndexStore

from circinus.logger import logger


Base = (Path(__file__).parent.parent / 'data').resolve(strict=True)


def _openai_llm(config):
    return OpenAI(api_key=config.openai_settings)


def _mock_llm() -> LLM:
    return MockLLM()


def _openai_embedding(config):
    return OpenAIEmbedding(api_key=config.openai_settings)


def _mock_embedding():
    return MockEmbedding(384)


def llm_component(config) -> LLM:
    return {
        'openai': partial(_openai_llm, config),
        'mock': _mock_llm,
    }[config.mode]()


def embedding_component(config) -> BaseEmbedding:
    return {
        'openai': partial(_openai_embedding, config),
        'mock': _mock_embedding,
    }[config.mode]()


def index_store_component() -> BaseIndexStore:
    try:
        return SimpleIndexStore.from_persist_dir(persist_dir=str(Base))
    except FileNotFoundError:
        logger.debug('Local index store not found, creating a new one')
        return SimpleIndexStore()


def doc_store_component() -> BaseDocumentStore:
    try:
        return SimpleDocumentStore.from_persist_dir(persist_dir=str(Base))
    except FileNotFoundError:
        logger.debug('Local document store not found, creating a new one')
        return SimpleDocumentStore()
