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
from rizmo.nodes.agent.system_prompt import SystemPromptBuilder
from rizmo.nodes.agent.tools import get_tool_handler
from rizmo.nodes.agent.value_store import ValueStore
from rizmo.nodes.messages_py36 import Detections
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm

NAME = 'Rizmo'

ALT_NAME_PATTERN = re.compile(
    rf'\b((pe)?r[eio][sz]+[mn][ao])\b',
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
        last_reply_datetime: datetime = datetime.now()

    state = State()

    client = OpenAI(api_key=secrets.OPENAI_API_KEY)

    memory_store = ValueStore(config.memory_file_path)
    system_prompt_builder = SystemPromptBuilder(memory_store)

    node = await build_mesh_node_from_args(args=args)
    say_topic = node.get_topic_sender(Topic.SAY)

    tool_handler = get_tool_handler(node, memory_store)

    chat = Chat(
        client,
        model=args.model,
        system_prompt_builder=system_prompt_builder,
        tool_handler=tool_handler,
        store=False,
    )

    async def handle_transcript(topic, transcript: str) -> None:
        now = datetime.now()
        transcript = preprocess(transcript)
        print(f'{now}:')
        print('Person:', transcript)

        if talking_to_me(transcript):
            state.in_conversation = True
        else:
            seconds_since_last_reply = (now - state.last_reply_datetime).total_seconds()
            if seconds_since_last_reply >= args.pause_convo_after:
                state.in_conversation = False

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
            await say('Okay.')

            chat.add_user_message(transcript)
            chat.add_assistant_message('Okay.')

            return

        chat.add_user_message(transcript)
        response = await chat.get_response()
        response = response.content.strip()

        if response != '<NO REPLY>':
            await say(response)

    async def say(text: str) -> None:
        await say_topic.send(text)
        state.last_reply_datetime = datetime.now()

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

    parser.add_argument(
        '--model',
        default='gpt-4o-mini',
        help='The OpenAI model to use. Default: %(default)s',
    )

    parser.add_argument(
        '--pause-convo-after',
        type=int,
        default=60,
        help='Pause conversation after this many seconds without a reply. Default: %(default)s',
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
