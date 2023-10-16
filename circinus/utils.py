import re
from typing import Optional


CODE_BLOCK = re.compile(
    r'```(?P<language>\w*)\n(?P<code>.*?)\n```',
    flags=re.DOTALL | re.MULTILINE,
)


def extract_code(text: str) -> Optional[str]:
    match = CODE_BLOCK.search(text)
    return match and match.group('code')
