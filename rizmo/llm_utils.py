import asyncio
import json
from abc import ABC
from collections import deque
from collections.abc import Callable
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import Function

SystemPromptBuilder = Callable[[], str]


class Chat:
    def __init__(
            self,
            client: OpenAI,
            model: str,
            system_prompt_builder: SystemPromptBuilder,
            tool_handler: 'ToolHandler' = None,
            **kwargs,
    ):
        self.client = client
        self.model = model
        self.system_prompt_builder = system_prompt_builder
        self.tool_handler = tool_handler
        self.kwargs = kwargs

        self.messages = deque()

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
        response = await self._get_one_response()
        while True:
            print('Assistant:', response)

            if response.content:
                return response

            for tool_call in response.tool_calls:
                result = await self.tool_handler.handle(tool_call.function)
                print(f'Tool call: {tool_call.function.name} -> {result}')
                self.add_tool_message(tool_call.id, result)

            response = await self._get_one_response()

    async def _get_one_response(self) -> ChatCompletionMessage:
        system_message = dict(
            role='system',
            content=self.system_prompt_builder(),
        )

        self.messages.appendleft(system_message)
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=self.messages,
                model=self.model,
                tools=self.tool_handler.tools if self.tool_handler else None,
                **self.kwargs,
            )
        finally:
            self.messages.popleft()

        message = response.choices[0].message
        self.messages.append(message)

        return message


class ToolHandler(ABC):
    tools = (
        # Template
        # dict(
        #     type='function',
        #     function=dict(
        #         name='name',
        #         description='Description of function.',
        #         parameters=dict(
        #             type='object',
        #             properties=dict(
        #                 arg=dict(
        #                     type='string',
        #                     description='Description of arg.',
        #                 ),
        #             ),
        #             required=['arg'],
        #             additionalProperties=False,
        #         ),
        #     ),
        # ),
    )

    async def handle(self, func_spec: Function) -> str:
        func = getattr(self, func_spec.name)
        kwargs = json.loads(func_spec.arguments)

        try:
            result = await func(**kwargs)
        except Exception as e:
            result = dict(error=repr(e))

        return json.dumps(result)
