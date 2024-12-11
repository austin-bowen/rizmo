from collections.abc import Callable
from datetime import datetime
from typing import Literal

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage


class Chat:
    def __init__(
            self,
            client: OpenAI,
            model: str,
            system_prompt_builder: 'SystemPromptBuilder',
            **kwargs,
    ):
        self.client = client
        self.model = model
        self.system_prompt_builder = system_prompt_builder
        self.kwargs = kwargs

        self.messages = []

    def add(self, role: Literal['user', 'assistant'], content: str) -> None:
        self.messages.append(dict(role=role, content=content))

    def get_response(self, user_content: str = None) -> ChatCompletionMessage:
        if user_content:
            self.add('user', user_content)

        system_prompt = self.system_prompt_builder()

        messages = [
                       dict(role='system', content=system_prompt),
                   ] + self.messages

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            **self.kwargs,
        )
        message = response.choices[0].message

        self.messages.append(message)

        return message


SystemPromptBuilder = Callable[[], str]


def with_datetime(prompt: str) -> str:
    now = datetime.now()
    date = now.strftime('%A, %B %d, %Y')
    time = now.strftime('%I:%M %p')
    return prompt.format(date=date, time=time)
