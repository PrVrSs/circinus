from pathlib import Path

from circinus.agent import Agent
from circinus.settings import load_config


docs = Path(__file__).parent / 'blob_documentation.txt'
specification = Path(__file__).parent / 'blob.webidl'
code = Path(__file__).parent / 'blob.js'

agent = Agent(config=load_config('conf.toml'))

agent.add_docs(
    file_name=docs.name,
    file_data=docs.read_text(encoding='utf-8'),
    type_='documentation',
)
agent.add_docs(
    file_name=specification.name,
    file_data=specification.read_text(encoding='utf-8'),
    type_='specification',
)
agent.add_docs(
    file_name=code.name,
    file_data=code.read_text(encoding='utf-8'),
    type_='code',
)


samples = agent.fuzzing(
    target='Blob object',
    generation_prompt='Please create a program which uses Blob class according to the following description'
)

for sample in samples:
    print(sample)
