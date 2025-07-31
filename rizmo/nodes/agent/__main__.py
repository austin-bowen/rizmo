import asyncio
import logging
import re
from argparse import Namespace
from asyncio import Queue
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal, Optional

from openai import OpenAI
from rosy import build_node_from_args

from rizmo import secrets
from rizmo.conference_speaker import ConferenceSpeaker
from rizmo.config import config
from rizmo.llm_utils import Chat
from rizmo.location import get_location_provider
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.agent.system_prompt import SystemPromptBuilder
from rizmo.nodes.agent.tools.timer import Timer, timedelta_to_hms_text
from rizmo.nodes.agent.tools.tools import get_tool_handler
from rizmo.nodes.agent.value_store import ValueStore
from rizmo.nodes.messages import FaceDetections
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

RESPONSE_REPLACEMENT_PATTERNS = [
    (
        re.compile(rf'\b{re.escape(old)}\b', flags=re.IGNORECASE),
        new,
    ) for old, new in [
        ('fr', 'for real'),
        ('sus', 'suss'),
    ]
]


async def main(args: Namespace) -> None:
    logging.basicConfig(level=args.log)

    @dataclass
    class State:
        in_conversation: bool = False
        last_reply_datetime: datetime = datetime.now()
        messages: Queue = Queue()

    state = State()

    client = OpenAI(api_key=secrets.OPENAI_API_KEY)

    location_provider = get_location_provider()
    memory_store = ValueStore(config.memory_file_path)
    speaker = ConferenceSpeaker.build()
    system_prompt_builder = SystemPromptBuilder(
        location_provider,
        memory_store,
        speaker,
    )

    node = await build_node_from_args(args=args)
    say_topic = node.get_topic(Topic.SAY)

    async def timer_complete_callback(timer: Timer) -> None:
        message_ = f'{timer} is done'

        overdone_ = timer.time_overdone
        if overdone_ is not None and overdone_.total_seconds() >= 60:
            overdue = timedelta_to_hms_text(overdone_, plural=True)
            message_ += f' (overdue by {overdue})'

        await state.messages.put(Message('system', message_))

    tool_handler = get_tool_handler(
        node,
        memory_store,
        timer_complete_callback,
        speaker,
    )

    chat = Chat(
        client,
        model=args.model,
        system_prompt_builder=system_prompt_builder,
        tool_handler=tool_handler,
        store=False,
        temperature=args.temperature,
    )

    async def handle_transcript(topic, transcript_: str) -> None:
        transcript_ = preprocess(transcript_)
        await state.messages.put(Message('user', transcript_))

    async def handle_objects_detected(topic, objects: Detections) -> None:
        system_prompt_builder.objects = objects

    async def handle_faces_recognized(topic, faces: FaceDetections) -> None:
        system_prompt_builder.faces = faces

    await node.listen(Topic.TRANSCRIPT, handle_transcript)
    await node.listen(Topic.OBJECTS_DETECTED, handle_objects_detected)
    await node.listen(Topic.FACES_RECOGNIZED, handle_faces_recognized)
    await say_topic.wait_for_listener()

    async def say(text: str) -> None:
        await say_topic.send(text)
        state.last_reply_datetime = datetime.now()

    async def say_response() -> None:
        async for response in chat.get_responses():
            print('Rizmo:', response)
            response = postprocess_response(response.content)
            if response and response != '<NO REPLY>':
                await say(response)
                state.in_conversation = True

    await say_response()

    while True:
        message = await state.messages.get()
        print(f'Message: {message!r}')

        if message.type == 'user':
            transcript = message.text

            now = datetime.now()

            if talking_to_me(transcript):
                state.in_conversation = True
            else:
                seconds_since_last_reply = (now - state.last_reply_datetime).total_seconds()
                if seconds_since_last_reply >= args.pause_convo_after:
                    state.in_conversation = False

            if not state.in_conversation:
                print('[Not talking to me]')
                continue

            if any_phrase_in(transcript, (
                    'pause conversation',
                    'pause convo',
                    'pause the conversation',
                    'pause the convo',
            )):
                print('[Conversation paused]')
                await say('Okay.')
                state.in_conversation = False

                chat.add_user_message(message.format())
                chat.add_assistant_message('Okay.')

                continue
        elif message.type == 'system':
            pass
        else:
            raise ValueError(f'Unknown message type: {message.type!r}')

        chat.add_user_message(message.format())
        await say_response()


@dataclass
class Message:
    type: Literal['user', 'system']
    text: str

    def format(self) -> str:
        return f'<{self.type}>{self.text}</{self.type}>'


def preprocess(transcript: str) -> str:
    return ALT_NAME_PATTERN.sub(NAME, transcript)


def postprocess_response(response: Optional[str]) -> str:
    response = response.strip() if response else ''

    for old_pattern, new in RESPONSE_REPLACEMENT_PATTERNS:
        response = old_pattern.sub(new, response)

    return response


def talking_to_me(transcript: str) -> bool:
    return WAKE_PATTERN.match(transcript) is not None


def any_phrase_in(transcript: str, phrases: Iterable[str]) -> bool:
    transcript = transcript.lower()
    return any(phrase in transcript for phrase in phrases)


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)

    parser.add_argument(
        '--model',
        default='gpt-4.1',
        help='The OpenAI model to use. Default: %(default)s',
    )

    parser.add_argument(
        '--pause-convo-after',
        type=int,
        default=60,
        help='Pause conversation after this many seconds without a reply. Default: %(default)s',
    )

    parser.add_argument(
        '--temperature',
        type=float,
        help='The temperature to use for the model.',
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
