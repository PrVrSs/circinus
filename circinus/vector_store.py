import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api.types import QueryResult
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from circinus.logger import logger


def tmp_db() -> Path:
    return Path(tempfile.gettempdir()) / 'circinus' / datetime.now().strftime('%Y%m%d-%H%M%S') / 'chromadb.db'


def split_text_to_chunks(text: str) -> list[str]:
    # TODO
    return [text]


class VectorStore:

    def __init__(self, db_path: Optional[str] = None):
        self._client = chromadb.PersistentClient(path=db_path or str(tmp_db()))

    def add_text(
        self,
        text: str,
        name: str = 'code-snippets',
        model_name: str = 'all-MiniLM-L6-v2',
        batch: int = 10000,
    ) -> None:
        collection = self._client.get_or_create_collection(
            name=name,
            embedding_function=SentenceTransformerEmbeddingFunction(model_name=model_name),
            metadata={"style": "style1"},
        )

        chunks = split_text_to_chunks(text)
        for i in range(0, len(chunks), min(batch, len(chunks))):
            end = i + min(batch, len(chunks) - i)
            collection.upsert(
                documents=chunks[i:end],
                ids=[
                    uuid.uuid4().hex
                    for _ in range(i, end)
                ],
            )

    def query(
        self,
        query: list[str],
        model_name: str = 'all-MiniLM-L6-v2',
        name: str = 'code-snippets',
        search_string: str = '',
        n_results: int = 5,
    ) -> QueryResult:
        collection = self._client.get_collection(name=name)

        return collection.query(
            query_embeddings=SentenceTransformerEmbeddingFunction(model_name=model_name)(query),
            n_results=n_results,
            where_document={'$contains': search_string} if search_string else None,
        )


def main():
    store = VectorStore()
    store.add_text(text='qwerty asd ggg gg')
    store.add_text(text='asd ggg gg qwerty')
    print(store.query(query=['qwerty']))


if __name__ == '__main__':
    main()
