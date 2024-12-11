import asyncio
import json
import subprocess
from argparse import Namespace
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable, Literal

from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever
from easymesh.node.node import TopicSender
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import Function

from rizmo.config import config
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.signal import graceful_shutdown_on_sigterm
from rizmo.weather import WeatherProvider

NAME = 'rizmo'
ALT_NAMES = (
    'prisma',
    'prismo',
    'prizma',
    'prizmo',
    'resma',
    'resmo',
    'risma',
    'rismo',
    'rizma',
    'rizzmo',
    'rosma',
    'rosmo',
)

MAIN_SYSTEM_MESSAGE = '''
You are a robot named Rizmo. Keep your responses short. Write as if you were
speaking out loud.
'''.strip()

CONVO_DETECTOR_SYSTEM_MESSAGE = '''
You are a robot named Rizmo. You will receive transcripts of audio, and you must
determine if the transcripts are part of a conversation with you, or not.

The transcripts are not perfect, and might contain false positives, especially
for short phrases, which may appear as phrases like "Thank you", or "Shh",
or ".", which are probably not part of a conversation with you.

If you think the transcript is part of a conversation with you, respond with "yes";
otherwise, respond with "no". 
'''


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)
    say_topic = node.get_topic_sender('say')

    api_key = open('.openai_api_key', 'r').read().strip()
    client = OpenAI(api_key=api_key)

    chat = Chat(
        client,
        model='gpt-4o-mini',
        store=False,
        system_message=MAIN_SYSTEM_MESSAGE,
        tools=ToolHandler.TOOLS,
    )

    convo_detector = ConvoDetector(
        Chat(
            client,
            model='gpt-4o-mini',
            store=False,
            system_message=CONVO_DETECTOR_SYSTEM_MESSAGE,
        ),
    )

    @dataclass
    class State:
        last_datetime: datetime = datetime.now()

    state = State()

    tool_handler = ToolHandler(
        say_topic,
        weather_provider=WeatherProvider.build(config.weather_location),
    )

    async def handle_transcript(topic, transcript: str) -> None:
        now = datetime.now()
        try:
            dt = now - state.last_datetime
            print(f'{now} ({dt}):')
            print('Person:', transcript)

            transcript = preprocess(transcript)

            if not talking_to_me(transcript):
                print('[Not talking to me]')
                return

            # if not convo_detector.is_conversation(transcript):
            #     print('[Not talking to me]')
            #     return

            response = chat.get_response(transcript)
            while True:
                print('Rizmo:', response)

                if response.content:
                    break

                for tool_call in response.tool_calls:
                    result = await tool_handler.handle(tool_call.function)
                    print(f'Tool call: {tool_call.function.name} -> {result}')

                    chat.messages.append(dict(
                        role='tool',
                        content=result,
                        tool_call_id=tool_call.id,
                    ))

                response = chat.get_response()

            await say_topic.send(response.content)
        finally:
            state.last_datetime = now

    await node.listen('transcript', handle_transcript)
    await forever()


class ToolHandler:
    TOOLS = (
        # Template
        # dict(
        #     type='function',
        #     function=dict(
        #         name='name',
        #         parameters=dict(
        #             type='object',
        #             properties=dict(
        #                 arg=dict(type='string'),
        #             ),
        #             required=['arg'],
        #             additionalProperties=False,
        #         ),
        #     ),
        # ),
        dict(
            type='function',
            function=dict(
                name='get_current_date',
            ),
        ),
        dict(
            type='function',
            function=dict(
                name='get_current_time',
            ),
        ),
        dict(
            type='function',
            function=dict(
                name='get_weather',
            ),
        ),
        dict(
            type='function',
            function=dict(
                name='reboot',
            ),
        ),
        dict(
            type='function',
            function=dict(
                name='shutdown',
            ),
        ),
    )

    def __init__(
            self,
            say_topic: TopicSender,
            weather_provider: WeatherProvider,
    ):
        self.say_topic = say_topic
        self.weather_provider = weather_provider

    async def handle(self, func_spec: Function) -> str:
        func = getattr(self, func_spec.name)
        kwargs = json.loads(func_spec.arguments)
        result = await func(**kwargs)
        return json.dumps(result)

    async def get_current_date(self) -> str:
        current_date = datetime.now()
        return current_date.strftime('%A, %B %d, %Y')

    async def get_current_time(self) -> str:
        current_time = datetime.now()
        return current_time.strftime('%I:%M %p')

    async def get_weather(self) -> dict:
        weather = await self.weather_provider.get_weather()
        return asdict(weather)

    async def reboot(self) -> dict:
        await self.say_topic.send('Be right back, rebooting.')
        await asyncio.sleep(3)

        return await self._run_cmd(
            'sudo', '--non-interactive', 'reboot'
        )

    async def shutdown(self) -> dict:
        await self.say_topic.send('Shutting down.')
        await asyncio.sleep(3)

        return await self._run_cmd(
            'sudo', '--non-interactive', 'shutdown', '-h', 'now'
        )

    async def _run_cmd(self, *args) -> dict:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return dict(
            command=' '.join(args),
            exit_code=result.returncode,
            stdout=result.stdout.decode(),
            stderr=result.stderr.decode(),
        )


class Chat:
    def __init__(
            self,
            client: OpenAI,
            model: str,
            system_message: str,
            **kwargs,
    ):
        self.client = client
        self.model = model
        self.kwargs = kwargs

        self.messages = [
            dict(role='system', content=system_message),
        ]

    def add(self, role: Literal['user', 'assistant'], content: str) -> None:
        self.messages.append(dict(role=role, content=content))

    def get_response(self, user_content: str = None) -> ChatCompletionMessage:
        if user_content:
            self.add('user', user_content)

        response = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            **self.kwargs,
        )
        message = response.choices[0].message

        self.messages.append(message)

        return message


class ConvoDetector:
    def __init__(self, chat: Chat):
        self.chat = chat

    def is_conversation(self, transcript: str) -> bool:
        response = self.chat.get_response(transcript).content

        if response == 'yes':
            return True
        elif response == 'no':
            return False
        else:
            print(f'ERROR: Unexpected response from convo detector: {response!r}')
            return False


def preprocess(transcript: str) -> str:
    t = transcript.lower()

    for alt_name in ALT_NAMES:
        t = t.replace(alt_name, NAME)

    return t


def talking_to_me(transcript: str) -> bool:
    t = transcript.replace(',', '')

    return transcript.startswith(NAME) or any_phrase_in(t, (
        f'hey {NAME}',
        f'hi {NAME}',
        f'hello {NAME}',
        f'okay {NAME}',
        f'ok {NAME}',
    ))


def any_phrase_in(transcript: str, phrases: Iterable[str]) -> bool:
    return any(phrase in transcript for phrase in phrases)


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
