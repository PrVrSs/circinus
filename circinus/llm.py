import openai

from .settings import settings


TOKEN_MAX = {
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-0301': 4096,
    'gpt-3.5-turbo-0613': 4096,
    'gpt-3.5-turbo-16k': 16384,
    'gpt-3.5-turbo-16k-0613': 16384,
    'gpt-4-0314': 8192,
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-4-32k-0314': 32768,
    'gpt-4-0613': 8192,
    'text-embedding-ada-002': 8192,
}


class GPT:

    system_prompt = 'You are a helpful assistant.'

    def __init__(self):
        openai.api_key = settings.openai_api_key

        self.llm = openai
        self.model = settings.openai_api_model
        self.auto_max_tokens = False
        self.temperature = 0

    def _system_msg(self, msg: str) -> dict[str, str]:
        return {'role': 'system', 'content': msg}

    def _user_msg(self, msg: str) -> dict[str, str]:
        return {'role': 'user', 'content': msg}

    def _default_system_msg(self):
        return self._system_msg(self.system_prompt)

    def ask(self, msg: str) -> str:
        message = [self._default_system_msg(), self._user_msg(msg)]
        rsp = self.completion(message)

        return self.get_choice_text(rsp)

    def get_choice_text(self, rsp: dict) -> str:
        return rsp.get('choices')[0]['message']['content']

    def completion(self, messages: list[dict]) -> dict:
        return self._chat_completion(messages)

    def _chat_completion(self, messages: list[dict]) -> dict:
        return self.llm.ChatCompletion.create(**self._cons_kwargs(messages))

    def _cons_kwargs(self, messages: list[dict]) -> dict:
        return {
            'messages': messages,
            'max_tokens': settings.max_tokens,
            'n': 1,
            'stop': None,
            'temperature': self.temperature,
            'timeout': 3,
            'model': self.model
        }
