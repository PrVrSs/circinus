from circinus.utils import extract_code

import pytest


CODE = """
def func(a):
    return a
"""

TEXT = """
TEXT
...
```language
{code}
```
...
TEXT
"""


@pytest.mark.parametrize('test_input, expected', [
    ('', None),
    (TEXT.format(code=CODE), CODE),
])
def test_extract_code(test_input, expected):
    assert extract_code(test_input) == expected
