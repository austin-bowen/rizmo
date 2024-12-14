"""
Setup:
- `pip install funasr modelscope torchaudio`
"""

import asyncio
from argparse import Namespace
from dataclasses import dataclass, field
from functools import wraps

from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever
from easymesh.node.node import MeshNode
from easymesh.types import Data, Topic
from funasr import AutoModel

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.signal import graceful_shutdown_on_sigterm


def depends_on_listener(node: MeshNode, downstream_topic: Topic):
    def decorator(func):
        @wraps(func)
        async def wrapper(topic_: Topic, data_: Data) -> None:
            if not await node.topic_has_listeners(downstream_topic):
                await node.stop_listening(topic_)
                await node.wait_for_listener(downstream_topic)
                await node.listen(topic_, wrapper)
                return

            return await func(topic_, data_)

        return wrapper

    return decorator


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    voice_detected_topic = node.get_topic_sender('voice_detected')

    vad_model = AutoModel(
        model='fsmn-vad',
        device='cuda',
    )

    @dataclass
    class State:
        model_cache: dict = field(default_factory=dict)
        voice_detected: bool = False

    state = State()

    @voice_detected_topic.depends_on_listener()
    async def handle_audio(topic, data):
        audio, timestamp = data
        block_size = audio.data.shape[0]
        indata = audio.data.squeeze()
        chunk_size_ms = int(round(1000 * block_size / audio.sample_rate))

        res = vad_model.generate(
            input=indata,
            cache=state.model_cache,
            is_final=False,
            chunk_size=chunk_size_ms,
        )

        voice_detected = state.voice_detected
        for detection in res[0]['value']:
            if detection[0] != -1:
                voice_detected = True
            elif detection[1] != -1:
                voice_detected = False

        if voice_detected != state.voice_detected:
            print('Voice detected:', state.voice_detected)
            state.voice_detected = voice_detected

        await voice_detected_topic.send((audio, timestamp, voice_detected))

    await node.listen('audio', handle_audio)

    try:
        await forever()
    finally:
        pass


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
