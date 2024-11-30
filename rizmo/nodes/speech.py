import asyncio
import random

import easymesh
from easymesh.asyncio import forever
from voicebox import ParallelVoicebox, Voicebox, reliable_tts
from voicebox.effects import Flanger, Glitch, Normalize
from voicebox.sinks import SoundDevice
from voicebox.tts import ESpeakNG


async def main():
    node = await easymesh.build_mesh_node('speech')

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
            Glitch(),
            Normalize(),
        ],
        sink=SoundDevice(latency=0.2),
    )


if __name__ == '__main__':
    asyncio.run(main())
