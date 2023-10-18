import pytest


@pytest.mark.parametrize('vector_store_data, query, expected', [
    (['qwerty asd', 'asd qwerty'], 'qwerty', 'qwerty'),
])
def test_vector_store(vector_store, vector_store_data, query, expected):
    vector_sore = vector_store(vector_store_data)

    assert any([
        expected in document[0]
        for document in vector_sore.query(query=[query]).get('documents', [])
    ]) is True
