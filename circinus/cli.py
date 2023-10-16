from pathlib import Path

import click

from circinus.fuzzer import UserInput, candidate_prompt, fuzzing_loop
from circinus.llm import GPT


@click.command()
@click.option(
    '-d', '--documentation',
    type=click.Path(dir_okay=False, resolve_path=True),
    help='',
)
@click.option(
    '-sp', '--specification',
    type=click.Path(dir_okay=False, resolve_path=True),
    help='',
)
@click.option(
    '-c', '--code',
    type=click.Path(dir_okay=False, resolve_path=True),
    help='',
)
@click.option(
    '-s', '--samples',
    type=int,
    default=2,
    help='',
)
@click.option(
    '-o', '--output',
    type=click.Path(file_okay=False, resolve_path=True),
    help='Output directory.'
)
def cli(documentation: str, specification: str, code: str, samples: int, output: str) -> None:
    llm = GPT()

    prompts = candidate_prompt(
        llm=llm,
        user_input=UserInput(
            documentation=Path(documentation).read_text(encoding='utf-8'),
            specification=Path(specification).read_text(encoding='utf-8'),
            code=Path(code).read_text(encoding='utf-8')
        ),
        num_samples=samples,
    )

    for index_x, prompt in enumerate(prompts):
        snippets = fuzzing_loop(llm=llm, prompt=prompt)
        for index_y, snippet in enumerate(snippets):
            if snippet:
                (Path(output) / f'{index_x}-{index_y}.txt').write_text(data=snippet, encoding='utf-8')
