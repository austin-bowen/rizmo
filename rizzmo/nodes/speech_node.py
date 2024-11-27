import asyncio

import easymesh
from easymesh.asyncio import forever
from voicebox import ParallelVoicebox, Voicebox
from voicebox.effects import Flanger, Glitch, Normalize
from voicebox.tts import ESpeakNG


async def main():
    node = await easymesh.build_mesh_node('speech_node')

    with build_voicebox() as voicebox:
        async def handle_say(topic, message: str) -> None:
            voicebox.say(message)

        await node.listen('say', handle_say)
        await forever()


def build_voicebox() -> Voicebox:
    return ParallelVoicebox(
        tts=ESpeakNG(),
        effects=[
            Flanger(),
            Glitch(),
            Normalize(),
        ]
    )


if __name__ == '__main__':
    asyncio.run(main())
