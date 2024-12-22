import asyncio
from collections.abc import Callable
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

SystemPromptBuilder = Callable[[], str]


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

    def add_user_message(self, content: str) -> None:
        self.messages.append(dict(role='user', content=content))

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(dict(role='assistant', content=content))

    def add_tool_message(self, tool_call_id: str, result: Any) -> None:
        self.messages.append(dict(
            role='tool',
            content=result,
            tool_call_id=tool_call_id,
        ))

    async def get_response(self) -> ChatCompletionMessage:
        system_prompt = self.system_prompt_builder()

        messages = [dict(role='system', content=system_prompt)]
        messages += self.messages

        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            messages=messages,
            model=self.model,
            **self.kwargs,
        )

        message = response.choices[0].message

        self.messages.append(message)

        return message
