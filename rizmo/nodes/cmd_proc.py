import asyncio
import subprocess
from argparse import Namespace
from collections.abc import Iterable
from datetime import datetime

from easymesh import build_mesh_node_from_args
from easymesh.argparse import get_node_arg_parser
from easymesh.asyncio import forever

from rizmo.config import config

NAME = 'rizmo'


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    async def handle_transcript(topic, transcript: str) -> None:
        print()
        print(f'Transcript: {transcript}')

        transcript = preprocess(transcript)

        if not talking_to_me(transcript):
            print('[Not talking to me]')
            return

        if any_phrase_in(transcript, (
                'shut down now',
                'shutdown now',
                'rest in a deep and dreamless slumber',
        )):
            await shutdown()
        elif any_phrase_in(transcript, (
                'what is today',
                "what is today's date",
                "what's today",
                'what day is it',
        )):
            await say_date()
        elif any_phrase_in(transcript, (
                'what time is it',
                "what's the time",
        )):
            await say_time()
        else:
            print('[Unrecognized command]')

    async def shutdown() -> None:
        await say('Shutting down.')
        await asyncio.sleep(3)
        subprocess.run(['sudo', 'poweroff'])

    async def say_date() -> None:
        current_date = datetime.now()
        current_date = current_date.strftime('%A, %B %d')

        await say(f'It is {current_date}.')

    async def say_time() -> None:
        current_time = datetime.now()
        current_time = current_time.strftime('%I:%M %p')

        await say(f'It is {current_time}.')

    async def say(message: str) -> None:
        print(repr(message))
        await node.send('say', message)

    await node.listen('transcript', handle_transcript)
    await forever()


def preprocess(transcript: str) -> str:
    t = transcript.lower()

    for alt_name in [
        'resma',
        'resmo',
        'prizmo',
        'risma',
        'rismo',
        'rizma',
        'rizzmo',
        'rosmo',
    ]:
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
    parser = get_node_arg_parser(
        default_node_name='cmd_proc',
        default_coordinator=config.coordinator,
    )

    return parser.parse_args()


if __name__ == '__main__':
    asyncio.run(main(parse_args()))
