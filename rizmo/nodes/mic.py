import asyncio
from abc import abstractmethod
from argparse import Namespace
from collections.abc import Callable
from threading import Thread
from typing import Optional

import numpy as np
import sounddevice as sd
from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import Audio

WEBCAM_MIC = 'USB CAMERA: Audio'
CONFERENCE_MIC = 'eMeet M0: USB Audio'
DEFAULT_MIC = CONFERENCE_MIC

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BLOCK_SIZE = 4096


class GainControl:
    @abstractmethod
    def get_gain(self, indata: np.ndarray, sample_rate: int) -> float:
        ...


class FixedGain(GainControl):
    def __init__(self, gain: float = 1.):
        self.gain = gain

    def get_gain(self, indata: np.ndarray, sample_rate: int) -> float:
        return self.gain


class LimitPowerGainControl(GainControl):
    def __init__(self, target_power: float = 0.5, keep_s: float = 2.):
        self.keep_s = keep_s
        self.target_power = target_power

        self._raw_powers = []

    def get_gain(self, indata: np.ndarray, sample_rate: int) -> float:
        # power = np.abs(indata).mean()
        power = np.abs(indata).max()

        self._raw_powers.append(power)
        self._raw_powers = self._raw_powers[-round(sample_rate * self.keep_s / len(indata)):]

        # gain = self.target_power / (max(self._raw_powers) + 1e-6)
        gain = self.target_power / (power + 1e-6)

        return gain


class AveragePowerGainControl(GainControl):
    def __init__(self, target_power: float = .8):
        self.target_power = target_power

        self._gain = 1.

    def get_gain(self, indata: np.ndarray, sample_rate: int) -> float:
        power = np.abs(indata).max()

        gained_power = self._gain * power

        if gained_power < self.target_power:
            samples = indata.shape[0]
            self._gain *= 1 + 2 * samples / sample_rate
        else:
            self._gain = self.target_power / (power + 1e-6)

        return self._gain


class AveragePowerGainControl2(GainControl):
    def __init__(
            self,
            target_power: float = 1.,
            max_gain: float = 100.,
    ):
        self.target_power = target_power
        self.max_gain = max_gain

        self._dt = 0.

    def get_gain(self, indata: np.ndarray, sample_rate: int) -> float:
        samples = indata.shape[0]
        self._dt += samples / sample_rate

        alpha = 2.
        gain = 2 ** (alpha * self._dt) - 1
        gain = min(gain, self.max_gain)

        power = np.abs(indata).max()

        gained_power = gain * power

        if gained_power > self.target_power:
            gain = self.target_power / (power + 1e-6)
            self._dt = np.log2(gain + 1) / alpha

        print(f'{self._dt:.2f}\t{gain}')

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

            # indata = indata.astype(np.float16)

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
    ):
        self.threshold = threshold
        self.attack = attack
        self.signal_transform = signal_transform

        self._gate_closed_time = 0.

    def __call__(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        input_power = np.abs(signal).max()

        # alpha = 0.1 * signal.shape[0] / sample_rate
        # self.threshold = alpha * input_power * 1. + (1 - alpha) * self.threshold

        transformed_signal = self.signal_transform(signal, sample_rate)

        if input_power >= self.threshold:
            print(f'Gate open; threshold={self.threshold}')
            self._gate_closed_time = 0.
        else:
            samples = signal.shape[0]
            self._gate_closed_time += samples / sample_rate

            gain = 1. / (2 ** (self._gate_closed_time * self.attack))
            print(f'Gate closed; threshold={self.threshold}; gain={gain}')

            transformed_signal *= gain

        return transformed_signal


async def main(
        args: Namespace,
        channels: int = 1,
):
    node = await build_mesh_node_from_args(args=args)
    audio_topic = node.get_topic_sender('audio')

    loop = asyncio.get_event_loop()

    # gain_control=LimitPowerGainControl(target_power=1., keep_s=2.)
    # gain_control=FixedGain(1.)
    gain_control = AveragePowerGainControl2()

    limiter = lambda signal: np.clip(signal, -1, 1)

    # limiter = np.tanh

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

        # power = np.abs(indata).max()
        # power = min(power, 1. - 1e-6)
        # power = int(power * len(block_symbols))
        # power = block_symbols[power]
        # print(power, end='', flush=True)

        audio = Audio(indata, args.sample_rate)

        asyncio.run_coroutine_threadsafe(
            audio_topic.send((audio, timestamp.inputBufferAdcTime)),
            loop,
        ).result()

    device = next(
        i for i, d in enumerate(sd.query_devices())
        if d['name'].startswith(args.device)
    )

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
    parser = get_rizmo_node_arg_parser('mic')

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
        default=DEFAULT_SAMPLE_RATE,
        help='The sample rate of the microphone. Default: %(default)s',
    )

    parser.add_argument(
        '--block-size', '-b',
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help='The block size of the microphone samples. Default: %(default)s',
    )

    return parser.parse_args()


if __name__ == '__main__':
    asyncio.run(main(parse_args()))
