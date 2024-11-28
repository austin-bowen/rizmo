import asyncio

import numpy as np
from easymesh import build_mesh_node
from easymesh.asyncio import forever

from rizzmo.config import config

block_symbols = '▁▂▃▄▅▆▇█'


async def handle_audio(topic, data) -> None:
    audio, timestamp = data

    power = np.abs(audio.data).max()
    power = min(power, 1. - 1e-3)
    power = int(power * len(block_symbols))
    power = block_symbols[power]
    print(power, end='', flush=True)


async def main():
    node = await build_mesh_node(
        name='audio_viz',
        coordinator_host=config.coordinator_host,
    )

    await node.listen('audio', handle_audio)
    await forever()


if __name__ == '__main__':
    asyncio.run(main())
