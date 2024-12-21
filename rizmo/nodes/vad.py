"""
Setup:
- `pip install funasr modelscope torchaudio`
"""

import asyncio
from argparse import Namespace
from dataclasses import dataclass, field

import numpy as np
from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever
from funasr import AutoModel
from voicebox.audio import Audio as VoiceboxAudio
from voicebox.effects import Filter

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import Topic
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    voice_detected_topic = node.get_topic_sender(Topic.VOICE_DETECTED)

    vad_model = AutoModel(
        model='fsmn-vad',
        device='cuda',
    )

    @dataclass
    class State:
        model_cache: dict = field(default_factory=dict)
        voice_detected: bool = False

    state = State()

    motor_noise_filter = Filter.build('lowpass', freq=3000, order=6)

    def filter_motor_noise(signal_: np.ndarray, sample_rate: int) -> np.ndarray:
        return motor_noise_filter(VoiceboxAudio(signal_, sample_rate)).signal

    @voice_detected_topic.depends_on_listener()
    async def handle_audio(topic, data):
        audio, timestamp = data
        block_size = audio.signal.shape[0]
        indata = audio.signal.squeeze()
        chunk_size_ms = int(round(1000 * block_size / audio.sample_rate))

        indata = filter_motor_noise(indata, audio.sample_rate)

        res = await asyncio.to_thread(
            vad_model.generate,
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
            print('Voice detected:', voice_detected)
            state.voice_detected = voice_detected

            # Elements of cache['stats'] seem to grow without bound;
            # clear it so it will be re-initialized and memory freed.
            if not voice_detected:
                state.model_cache.clear()

        await voice_detected_topic.send((audio, timestamp, voice_detected))

    await node.listen(Topic.AUDIO, handle_audio)
    await forever()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
