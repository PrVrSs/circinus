import random
import time
from itertools import chain
from typing import Optional

from attrs import define

from .llm import GPT
from .utils import extract_code


circinus_random = random.Random(time.process_time())

SUMMARIZE_PROMPT = """
Please summarize the above information in a concise manner to describe the usage and functionality of the target."""
GENERATION_PROMPT = 'Please create a program which uses Blob class according to the following description'
GENERATION_MUTATE_PROMPT = 'Program:\n {generation}.\n Please create a mutated program that modifiers this program.'
GENERATION_SEMANTIC_EQUIP = 'Program:\n {generation}.\n Please create a semantically equivalent program to this program'


@define
class UserInput:
    documentation: Optional[str]
    specification: Optional[str]
    code: Optional[str]

    def __str__(self):
        return '\n'.join(filter(None, [self.documentation, self.specification, self.code]))


def candidate_prompt(llm: GPT, user_input: UserInput, num_samples: int = 3) -> list[str]:
    llm.temperature = 0
    greedy_prompt = llm.ask('\n'.join([SUMMARIZE_PROMPT, str(user_input)]))

    llm.temperature = 1
    diverse_prompt = [
        llm.ask('\n'.join([SUMMARIZE_PROMPT, str(user_input)]))
        for _ in range(num_samples)
    ]

    return list(chain([greedy_prompt], diverse_prompt))


def fuzzing_loop(llm: GPT, prompt: str) -> list[str]:
    result = []
    llm.temperature = 1

    program = llm.ask('\n'.join([GENERATION_PROMPT, prompt]))
    result.append(code_snippet := extract_code(program))

    for _ in range(2):
        instruction = circinus_random.choice([GENERATION_MUTATE_PROMPT, GENERATION_SEMANTIC_EQUIP])
        program = llm.ask(instruction.format(generation=code_snippet))
        result.append(code_snippet := extract_code(program))

    return result
