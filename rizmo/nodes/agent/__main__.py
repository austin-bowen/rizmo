import asyncio
import re
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever
from openai import OpenAI

from rizmo import secrets
from rizmo.config import config
from rizmo.llm_utils import Chat
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.agent.reminders import ReminderSystem
from rizmo.nodes.agent.system_prompt import SystemPromptBuilder
from rizmo.nodes.agent.tools import RizmoToolHandler
from rizmo.nodes.messages_py36 import Detections
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm
from rizmo.weather import WeatherProvider

NAME = 'Rizmo'

ALT_NAME_PATTERN = re.compile(
    rf'\b(p?r[eio][sz]+[mn][ao])\b',
    flags=re.IGNORECASE,
)
"""Matches names like "Rizmo", "Rizma", "Prizmo", etc."""

WAKE_PATTERN = re.compile(
    # Start of line or sentence
    r'(^|.*[.?!] )'
    # Optional "hey", "hi", etc.
    r'((hey|hi|hello|okay|ok),? )?'
    # Name
    rf'{NAME}',
    flags=re.IGNORECASE,
)


async def main(args: Namespace) -> None:
    @dataclass
    class State:
        in_conversation: bool = False
        last_datetime: datetime = datetime.now()

    state = State()

    client = OpenAI(api_key=secrets.OPENAI_API_KEY)
    system_prompt_builder = SystemPromptBuilder()

    weather_provider = WeatherProvider.build(config.weather_location)
    reminder_system = ReminderSystem(config.reminders_file_path)

    node = await build_mesh_node_from_args(args=args)
    say_topic = node.get_topic_sender(Topic.SAY)

    tool_handler = RizmoToolHandler(
        say_topic=say_topic,
        motor_system_topic=node.get_topic_sender(Topic.MOTOR_SYSTEM),
        weather_provider=weather_provider,
        reminder_system=reminder_system,
    )

    chat = Chat(
        client,
        model='gpt-4o-mini',
        system_prompt_builder=system_prompt_builder,
        tool_handler=tool_handler,
        store=False,
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
                    'pause the conversation',
                    'pause the convo',
            )):
                print('[Conversation paused]')
                state.in_conversation = False
                await say_topic.send('Okay.')

                chat.add_user_message(transcript)
                chat.add_assistant_message('Okay.')

                return

            chat.add_user_message(transcript)
            response = await chat.get_response()
            response = response.content.strip()

            if response != '<NO REPLY>':
                await say_topic.send(response)
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
