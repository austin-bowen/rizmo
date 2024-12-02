import asyncio
import subprocess
from argparse import Namespace
from collections.abc import Iterable
from datetime import datetime

from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever

from rizmo.config import config
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.signal import graceful_shutdown_on_sigterm
from rizmo.weather import WeatherProvider

NAME = 'rizmo'
ALT_NAMES = (
    'resma',
    'resmo',
    'prisma',
    'prismo',
    'prizma',
    'prizmo',
    'risma',
    'rismo',
    'rizma',
    'rizzmo',
    'rosmo',
)


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
                "what's today's date",
                "what's today",
                'what day is it',
        )):
            await say_date()
        elif any_phrase_in(transcript, (
                'what time is it',
                "what's the time",
        )):
            await say_time()
        elif any_phrase_in(transcript, (
                "what is the weather",
                "what is the forecast",
                "what's the weather",
                "what's the forecast",
        )):
            await say_weather(transcript)
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

    async def say_weather(transcript: str) -> None:
        await say('Checking.')

        today = tomorrow = this_week = False

        if 'today' in transcript:
            today = True
        if 'tomorrow' in transcript:
            tomorrow = True
        if 'this week' in transcript:
            today = tomorrow = this_week = True
        if not (today or tomorrow or this_week):
            today = tomorrow = True

        weather = await weather_provider.get_weather()

        if today:
            await say(weather.today)
        if tomorrow:
            await say(weather.tomorrow)
        if this_week:
            await say(weather.this_week)

    async def say(message: str) -> None:
        print(repr(message))
        await node.send('say', message)

    await node.listen('transcript', handle_transcript)

    async with WeatherProvider.build(config.weather_location) as weather_provider:
        await forever()


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
    parser = get_rizmo_node_arg_parser('cmd_proc')
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
