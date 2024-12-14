import asyncio
import wave
from argparse import Namespace

import numpy as np
from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever
from easymesh.utils import require

from rizmo.config import config
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import Audio
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    async def handle_audio(topic, data) -> None:
        audio: Audio = data[0]

        require(
            audio.sample_rate == args.sample_rate,
            f'Expected sample rate {args.sample_rate}, got {audio.sample_rate}',
        )

        print('.', end='', flush=True)

        wav_data = audio.data * 32767
        wav_data = wav_data.astype(np.int16)
        wav_file.writeframes(wav_data.tobytes())

    print(f'Writing audio to {str(args.file_name)!r}...')
    with wave.open(args.file_name, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(args.sample_rate)

        await node.listen('audio', handle_audio)
        await forever()


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

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
