import asyncio
import random
from argparse import Namespace

import easymesh
from easymesh.asyncio import forever
from voicebox import ParallelVoicebox, Voicebox, reliable_tts
from voicebox.effects import Flanger, Normalize
from voicebox.sinks import SoundDevice
from voicebox.tts import ESpeakNG

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    node = await easymesh.build_mesh_node_from_args(args=args)

    with build_voicebox() as voicebox:
        voicebox.say(random.choice([
            'Hello, world!',
            'I am awake.',
            'I am online.',
        ]))

        async def handle_say(topic, message: str) -> None:
            print(repr(message))
            voicebox.say(message)

        await node.listen('say', handle_say)
        await forever()


def build_voicebox() -> Voicebox:
    return ParallelVoicebox(
        tts=reliable_tts(ttss=[ESpeakNG()]),
        effects=[
            Flanger(),
            Normalize(),
        ],
        sink=SoundDevice(latency=0.2),
    )


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
