import asyncio
import random
from argparse import Namespace

import easymesh
from easymesh.asyncio import forever
from voicebox import ParallelVoicebox, Voicebox, reliable_tts
from voicebox.effects import Flanger, Tail
from voicebox.sinks import SoundDevice
from voicebox.tts import AmazonPolly, ESpeakNG

from rizmo.aws import get_polly_client
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    node = await easymesh.build_mesh_node_from_args(args=args)

    with build_voicebox(args.tts) as voicebox:
        voicebox.say(random.choice([
            'Hello, world!',
            'I am awake.',
            'I am online.',
            "It's good to be back!",
        ]))

        async def handle_say(topic, message: str) -> None:
            print(repr(message))
            voicebox.say(message)

        await node.listen('say', handle_say)
        await forever()


def build_voicebox(tts: str) -> Voicebox:
    ttss = []
    if tts == 'kevin':
        ttss.append(AmazonPolly(
            client=get_polly_client(),
            voice_id='Kevin',
            engine='neural',
            language_code='en-US',
            sample_rate=16_000,
        ))
    elif tts != 'espeak':
        raise ValueError(f'Unknown TTS engine: {tts!r}')

    ttss.append(ESpeakNG())

    return ParallelVoicebox(
        tts=reliable_tts(ttss=ttss),
        effects=[
            Tail(0.5),
            Flanger(),
        ],
        sink=SoundDevice(latency=0.2),
    )


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)

    parser.add_argument(
        '--tts',
        default='kevin',
        choices=('espeak', 'kevin'),
        help='The text-to-speech engine to use. Default: %(default)s',
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
