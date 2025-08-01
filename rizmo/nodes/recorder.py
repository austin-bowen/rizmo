import asyncio
import logging
import wave
from argparse import Namespace

import numpy as np
from rosy import build_node_from_args
from rosy.utils import require
from voicebox.audio import Audio

from rizmo.config import config
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    logging.basicConfig(level=args.log)

    node = await build_node_from_args(args=args)

    async def handle_audio(topic, data) -> None:
        audio: Audio = data[0]
        await save_audio(audio)

    async def handle_voice_detected(topic, data) -> None:
        audio, _, voice_detected = data
        if voice_detected:
            await save_audio(audio)

    async def save_audio(audio: Audio) -> None:
        require(
            audio.sample_rate == args.sample_rate,
            f'Expected sample rate {args.sample_rate}, got {audio.sample_rate}',
        )

        print('.', end='', flush=True)

        wav_data = audio.signal * 32767
        wav_data = wav_data.astype(np.int16)
        wav_file.writeframes(wav_data.tobytes())

    print(f'Writing audio to {str(args.file_name)!r}...')
    with wave.open(args.file_name, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(args.sample_rate)

        if args.voice_only:
            await node.listen(Topic.VOICE_DETECTED, handle_voice_detected)
        else:
            await node.listen(Topic.AUDIO, handle_audio)

        await node.forever()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)

    parser.add_argument(
        'file_name',
        help='The file name to save the audio to. '
             'It is a wave file, so it should end in ".wav".',
    )

    parser.add_argument(
        '--sample-rate', '-s',
        type=int,
        default=config.mic_sample_rate,
        help='The expected sample rate of the audio. Default: %(default)s',
    )

    parser.add_argument(
        '--voice-only', '-v',
        action='store_true',
        help='Only record audio when voice is detected.',
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
