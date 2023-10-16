from pathlib import Path

from .fuzzer import UserInput, candidate_prompt, fuzzing_loop
from .llm import GPT


class Agent:
    def __init__(self):
        self._llm = GPT()

    def fuzz(self, documentation: str, specification: str, code: str):
        prompts = candidate_prompt(
            llm=self._llm,
            user_input=UserInput(
                documentation=Path(documentation).resolve(strict=True).read_text(encoding='utf-8'),
                specification=Path(specification).resolve(strict=True).read_text(encoding='utf-8'),
                code=Path(code).resolve(strict=True).read_text(encoding='utf-8')
            ),
            num_samples=2,
        )

        for prompt in prompts:
            yield from fuzzing_loop(llm=self._llm, prompt=prompt)
