from pathlib import Path

from attrs import define

from circinus.fuzzer import UserInput, candidate_prompt, fuzzing_loop
from circinus.knowledge_base import ChunkDTO, init_knowledge_base
from circinus.llm import GPT


@define(hash=True)
class Source:
    file: str
    page: str
    text: str

    @staticmethod
    def curate_sources(sources: list[ChunkDTO]) -> set['Source']:
        curated_sources = set()

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata

            file_name = doc_metadata.get('file_name', '-') if doc_metadata else '-'
            page_label = doc_metadata.get('page_label', '-') if doc_metadata else '-'

            source = Source(file=file_name, page=page_label, text=chunk.text)
            curated_sources.add(source)

        return curated_sources

    def __str__(self):
        return '\n'.join([
            f'{self.file=}',
            f'{self.page=}',
            f'{self.text=}'
        ])


class Agent:
    def __init__(self, config):
        self._llm = GPT()
        self.knowledge_base = init_knowledge_base(config)

    def search_in_docs(self):
        pass

    def add_docs(self, file_name: str, file_data: str | bytes) -> None:
        self.knowledge_base.add(file_name, file_data)

    def query_docs(self, message: str):
        return Source.curate_sources(
            sources=self.knowledge_base.retrieve(
                text=message,
                limit=4,
                prev_next_chunks=0,
            ),
        )

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
