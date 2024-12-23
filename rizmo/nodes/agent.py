import asyncio
import json
import re
import subprocess
from argparse import Namespace
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable, Optional

import psutil
from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever
from easymesh.node.node import MeshNode
from openai import OpenAI
from openai.types.chat.chat_completion_message_tool_call import Function

from rizmo import secrets
from rizmo.config import config
from rizmo.llm_utils import Chat
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import MotorSystemCommand
from rizmo.nodes.topics import Topic
from rizmo.nodes.messages_py36 import Detections
from rizmo.signal import graceful_shutdown_on_sigterm
from rizmo.weather import WeatherProvider

NAME = 'Rizmo'
ALT_NAMES = (
    'prisma',
    'prismo',
    'prizma',
    'prizmo',
    'resma',
    'resmo',
    'rezma',
    'rezmo',
    'risma',
    'rismo',
    'rizma',
    'rizno',
    'rizzmo',
    'rosma',
    'rosmo',
)

ALT_NAME_PATTERN = re.compile(
    rf'\b({"|".join(ALT_NAMES)})\b',
    flags=re.IGNORECASE,
)

WAKE_PATTERN = re.compile(
    # Start of sentence
    r'(^|.*[.?!] )'
    # Optional "hey", "hi", etc.
    r'((hey|hi|hello|okay|ok),? )?'
    # Name
    rf'{NAME}',
    flags=re.IGNORECASE,
)

MAIN_SYSTEM_PROMPT = '''
You are a robot named Rizmo.

You are stationary, but you have a head that looks around, and a camera to see.
You have a microphone and a speaker.

You are hearing a live transcript of audio. Sometimes the transcript may contain
errors, especially for short phrases; you can ignore these if they don't appear
to be part of the conversation.

Whatever you say will be read out loud, so write as if you were speaking.
Keep your responses short. You do not need to reply to all messages;
if a message does not need a reply, simply say "<NO REPLY>".

Here are some phrases you should listen for and how to respond to them:
- "rest in a deep and dreamless slumber": call the "shutdown" function.
- "stop/cease all motor functions": call the "motor_system" function with the "enabled" argument set to "false".
- "bring yourself back online": call the "motor_system" function with the "enabled" argument set to "true".

Context:
- Current date: {date}
- Current time: {time}
- Objects seen (count): {objects}.
'''.strip()


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)
    say_topic = node.get_topic_sender(Topic.SAY)

    client = OpenAI(api_key=secrets.OPENAI_API_KEY)

    system_prompt_builder = SystemPromptBuilder(MAIN_SYSTEM_PROMPT)

    chat = Chat(
        client,
        model='gpt-4o-mini',
        system_prompt_builder=system_prompt_builder,
        store=False,
        tools=ToolHandler.TOOLS,
    )

    @dataclass
    class State:
        in_conversation: bool = False
        last_datetime: datetime = datetime.now()

    state = State()

    tool_handler = ToolHandler(
        node,
        weather_provider=WeatherProvider.build(config.weather_location),
    )

    async def handle_transcript(topic, transcript: str) -> None:
        now = datetime.now()
        try:
            dt = now - state.last_datetime
            print(f'{now} ({dt}):')
            print('Person:', transcript)

            transcript = preprocess(transcript)

            if talking_to_me(transcript):
                state.in_conversation = True

            if not state.in_conversation:
                print('[Not talking to me]')
                return

            if any_phrase_in(transcript, (
                    'pause conversation',
                    'pause convo',
            )):
                print('[Conversation paused]')
                state.in_conversation = False
                await say_topic.send('Okay.')

                chat.add_user_message(transcript)
                chat.add_assistant_message('Okay.')

                return

            chat.add_user_message(transcript)
            response = await chat.get_response()
            while True:
                print('Rizmo:', response)

                if response.content:
                    break

                for tool_call in response.tool_calls:
                    result = await tool_handler.handle(tool_call.function)
                    print(f'Tool call: {tool_call.function.name} -> {result}')
                    chat.add_tool_message(tool_call.id, result)

                response = await chat.get_response()

            if response.content.strip() != '<NO REPLY>':
                await say_topic.send(response.content)
        finally:
            state.last_datetime = now

    async def handle_objects_detected(topic, objects: Detections) -> None:
        system_prompt_builder.objects = objects

    await node.listen(Topic.TRANSCRIPT, handle_transcript)
    await node.listen(Topic.OBJECTS_DETECTED, handle_objects_detected)

    try:
        await forever()
    finally:
        client.close()


class SystemPromptBuilder:
    objects: Optional[Detections]

    def __init__(self, system_prompt_template: str):
        self.system_prompt_template = system_prompt_template

        self.objects = None

    def __call__(self) -> str:
        template_vars = self._get_template_vars()
        prompt = self.system_prompt_template.format(**template_vars)
        print('System prompt:', prompt)
        return prompt

    def _get_template_vars(self) -> dict:
        return {
            **self._get_datetime(),
            **self._get_objects(),
        }

    def _get_datetime(self) -> dict:
        now = datetime.now()
        date = now.strftime('%A, %B %d, %Y')
        time = now.strftime('%I:%M %p')
        return dict(date=date, time=time)

    def _get_objects(self) -> dict:
        objects = self.objects

        if objects is not None:
            obj_counts = Counter(obj.label for obj in objects.objects)
            labels = sorted(obj_counts.keys())
            obj_strs = (f'{label} ({obj_counts[label]})' for label in labels)
            objects = ', '.join(obj_strs)
        else:
            objects = 'None'

        return dict(objects=objects)


class ToolHandler:
    TOOLS = (
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
        dict(
            type='function',
            description='Gets system CPU, memory, and disk usage, and CPU temperature in Celsius.',
            function=dict(
                name='get_system_status',
            ),
        ),
        dict(
            type='function',
            function=dict(
                name='get_weather',
                description=
                'Gets the weather for today, tomorrow, and the week, '
                'as well as the current moon phase.',
            ),
        ),
        dict(
            type='function',
            function=dict(
                name='motor_system',
                description='Controls the motor system.',
                parameters=dict(
                    type='object',
                    properties=dict(
                        enabled=dict(
                            type='boolean',
                            description='Whether to enable or disable the motor system.',
                        ),
                    ),
                    required=['enabled'],
                    additionalProperties=False,
                ),
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
            node: MeshNode,
            weather_provider: WeatherProvider,
    ):
        self.say_topic = node.get_topic_sender(Topic.SAY)
        self.motor_system_topic = node.get_topic_sender(Topic.MOTOR_SYSTEM)
        self.weather_provider = weather_provider

    async def handle(self, func_spec: Function) -> str:
        func = getattr(self, func_spec.name)
        kwargs = json.loads(func_spec.arguments)
        result = await func(**kwargs)
        return json.dumps(result)

    async def get_system_status(self) -> dict:
        cpu_usage_percent = psutil.cpu_percent(interval=0.5)
        memory_usage = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')

        temps = psutil.sensors_temperatures()
        cpu_temp_c = (
            temps['thermal-fan-est'][0].current
            if 'thermal-fan-est' in temps else 'unknown'
        )

        return dict(
            cpu_usage_percent=cpu_usage_percent,
            memory_usage_percent=memory_usage.percent,
            disk_usage_percent=disk_usage.percent,
            cpu_temp_c=cpu_temp_c,
        )

    async def get_weather(self) -> dict:
        weather = await self.weather_provider.get_weather()
        return asdict(weather)

    async def motor_system(self, enabled: bool) -> None:
        await self.motor_system_topic.send(MotorSystemCommand(enabled=enabled))

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


def preprocess(transcript: str) -> str:
    return ALT_NAME_PATTERN.sub(NAME, transcript)


def talking_to_me(transcript: str) -> bool:
    return WAKE_PATTERN.match(transcript) is not None


def any_phrase_in(transcript: str, phrases: Iterable[str]) -> bool:
    transcript = transcript.lower()
    return any(phrase in transcript for phrase in phrases)


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
