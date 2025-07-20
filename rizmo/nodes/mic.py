import asyncio
from abc import abstractmethod
from argparse import Namespace
from collections.abc import Callable
from threading import Thread
from typing import Optional

import numpy as np
import sounddevice as sd
from easymesh import build_node_from_args
from easymesh.asyncio import forever
from voicebox.audio import Audio

from rizmo.config import config
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm

WEBCAM_MIC = 'USB CAMERA: Audio'
CONFERENCE_MIC = 'eMeet M0: USB Audio'
DEFAULT_MIC = CONFERENCE_MIC


class GainControl:
    @abstractmethod
    def get_gain(self, indata: np.ndarray, sample_rate: int) -> float:
        ...


class FixedGain(GainControl):
    def __init__(self, gain: float = 1.):
        self.gain = gain

    def get_gain(self, indata: np.ndarray, sample_rate: int) -> float:
        return self.gain


class LimitedMaxPowerGainControl(GainControl):
    """
    Increases the gain exponentially until the max power of the signal
    reaches the target power.
    """

    def __init__(
            self,
            target_power: float = 1.,
            max_gain: float = 100.,
            alpha: float = 2.,
    ):
        self.target_power = target_power
        self.max_gain = max_gain
        self.alpha = alpha

        self._dt = 0.

    def get_gain(self, indata: np.ndarray, sample_rate: int) -> float:
        samples = indata.shape[0]
        self._dt += samples / sample_rate

        gain = 2 ** (self.alpha * self._dt) - 1
        gain = min(gain, self.max_gain)

        power = np.abs(indata).max()

        gained_power = gain * power
        if gained_power > self.target_power:
            gain = self.target_power / (power + 1e-6)
            self._dt = np.log2(gain + 1) / self.alpha

        return gain


class Microphone(Thread):
    def __init__(
            self,
            block_handler,
            sample_rate: int,
            block_size: int,
            device: int = None,
            channels: int = 1,
            name: str = 'Microphone',
            daemon: bool = True,
            **kwargs,
    ):
        super().__init__(name=name, daemon=daemon, **kwargs)

        self.block_handler = block_handler
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self.channels = channels

    def run(self) -> None:
        def input_stream_callback(indata: np.ndarray, frames: int, timestamp, status):
            if status:
                print(f"Warning: {status}", flush=True)

            self.block_handler(indata, frames, timestamp, status)

        with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=self.device,
                channels=self.channels,
                callback=input_stream_callback,
        ):
            while True:
                sd.sleep(1000)


class Gate:
    def __init__(
            self,
            threshold: float,
            attack: float,
            signal_transform: Callable[[np.ndarray, int], np.ndarray],
            debug: bool = False,
    ):
        self.threshold = threshold
        self.attack = attack
        self.signal_transform = signal_transform
        self.debug = debug

        self._gate_closed_time = 0.

    def __call__(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        input_power = np.abs(signal).max()

        transformed_signal = self.signal_transform(signal, sample_rate)

        if input_power >= self.threshold:
            if self.debug:
                print(f'Gate open; threshold={self.threshold}')

            self._gate_closed_time = 0.
        else:
            samples = signal.shape[0]
            self._gate_closed_time += samples / sample_rate

            gain = 1. / (2 ** (self._gate_closed_time * self.attack))

            if self.debug:
                print(f'Gate closed; threshold={self.threshold}; gain={gain}')

            transformed_signal *= gain

        return transformed_signal


async def main(
        args: Namespace,
        channels: int = 1,
):
    node = await build_node_from_args(args=args)
    audio_topic = node.get_topic(Topic.AUDIO)

    loop = asyncio.get_event_loop()

    gain_control = LimitedMaxPowerGainControl()
    if args.gain == 'auto':
        gain_control = LimitedMaxPowerGainControl()
    else:
        gain = float(args.gain)
        gain_control = FixedGain(gain)

    limiter = lambda signal: np.clip(signal, -1, 1)

    def signal_transform(signal, sample_rate_):
        return limiter(signal * gain_control.get_gain(signal, args.sample_rate))

    gate = Gate(
        threshold=0.,  # Disabled
        # threshold=0.001,  # Built-in webcam mic
        # threshold=0.02,  # USB webcam mic
        attack=16.,
        signal_transform=signal_transform,
    )

    def mic_callback(indata: np.ndarray, frames: int, timestamp, status):
        indata = gate(indata, args.sample_rate)
        audio = Audio(indata, args.sample_rate)

        asyncio.run_coroutine_threadsafe(
            audio_topic.send((audio, timestamp.inputBufferAdcTime)),
            loop,
        ).result()

    if args.device is None:
        device = None
    else:
        while not (device := next(
                (
                        i for i, d in enumerate(sd.query_devices())
                        if d['name'].startswith(args.device)
                ),
                None
        )):
            print('No mic device found.')
            await asyncio.sleep(5)

    mic = Microphone(
        mic_callback,
        args.sample_rate,
        block_size=args.block_size,
        device=device,
        channels=channels,
    )
    mic.start()

    await forever()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)

    def device_arg(arg_: str) -> Optional[str]:
        return None if arg_.lower() == 'none' else arg_

    parser.add_argument(
        '--device', '-d',
        default=DEFAULT_MIC,
        type=device_arg,
        help='The name prefix of the microphone device. '
             '"None" will use system mic. Default: "%(default)s"',
    )

    parser.add_argument(
        '--sample-rate', '-s',
        type=int,
        default=config.mic_sample_rate,
        help='The sample rate of the microphone. Default: %(default)s',
    )

    parser.add_argument(
        '--block-size', '-b',
        type=int,
        default=config.mic_block_size,
        help='The block size of the microphone samples. Default: %(default)s',
    )

    parser.add_argument(
        '--gain', '-g',
        default='auto',
        help='The gain control to use. Can be a number for constant gain, '
             'or "auto" for automatic gain adjustment. Default: %(default)s',
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
