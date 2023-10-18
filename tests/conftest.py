import pytest

from circinus.vector_store import VectorStore


@pytest.fixture
def empty_vector_store():
    yield VectorStore()


@pytest.fixture
def vector_store(empty_vector_store):
    def inner(texts):
        for text in texts:
            empty_vector_store.add_text(text=text)

        return empty_vector_store
    yield inner
