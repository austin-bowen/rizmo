"""
Automatic Speech Recognition (ASR) node.

Followed instructions here:
- https://huggingface.co/learn/audio-course/en/chapter5/asr_models
- https://huggingface.co/openai/whisper-large-v3-turbo
"""

import asyncio
from argparse import Namespace
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread

import numpy as np
import torch
from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from voicebox.audio import Audio

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm

PRE_BUFFER_DURATION_S = 1.
"""How much previous audio to prepend to the audio buffer when voice is detected."""


class ASR(Thread):
    def __init__(
            self,
            pipe,
            handle_transcript: Callable[[str], None],
            name='ASR',
            daemon=True,
            **kwargs,
    ):
        super().__init__(name=name, daemon=daemon, **kwargs)

        self.pipe = pipe
        self.handle_transcript = handle_transcript
        self.queue = Queue()

    def run(self) -> None:
        while (audio := self.queue.get()) is not None:
            try:
                self._handle_audio(audio)
            finally:
                self.queue.task_done()

    def _handle_audio(self, audio: Audio) -> None:
        sample = dict(
            array=audio.signal.copy(),
            sampling_rate=audio.sample_rate,
        )

        result = self.pipe(
            sample,
            generate_kwargs={'language': 'english'},
        )

        transcript = result['text'].strip()

        self.handle_transcript(transcript)

    def stop(self):
        self.queue.put(None)


class MaxDurationAudioBuffer:
    def __init__(self, max_duration: float):
        self.max_duration = max_duration
        self._buffer = deque()

    def __iter__(self):
        return iter(self._buffer)

    def append(self, audio: Audio) -> None:
        self._buffer.append(audio)

        duration = sum(a.len_seconds for a in self._buffer)
        while duration > self.max_duration:
            removed_audio = self._buffer.popleft()
            duration -= removed_audio.len_seconds


def build_asr_thread(handle_transcript: Callable[[str], None]):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = 'openai/whisper-large-v3-turbo'

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        'automatic-speech-recognition',
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        torch_dtype=torch_dtype,
        chunk_length_s=30,
    )

    asr = ASR(pipe, handle_transcript)
    asr.start()
    return asr


async def main(args: Namespace):
    node = await build_mesh_node_from_args(args=args)
    transcript_topic = node.get_topic_sender(Topic.TRANSCRIPT)

    loop = asyncio.get_event_loop()

    def handle_transcript(transcript: str) -> None:
        print(repr(transcript))

        asyncio.run_coroutine_threadsafe(
            transcript_topic.send(transcript),
            loop,
        ).result()

    asr = build_asr_thread(handle_transcript)

    @dataclass
    class State:
        most_recent_audios: list[Audio] = field(
            default_factory=lambda: MaxDurationAudioBuffer(max_duration=PRE_BUFFER_DURATION_S),
        )
        audio_buffer: list[Audio] = field(default_factory=list)
        self_speaking: bool = False

    state = State()

    async def handle_voice_detected(topic, data):
        audio, _, voice_detected = data

        state.most_recent_audios.append(audio)
        audio_buffer = state.audio_buffer

        # Do not transcribe while speaking
        if state.self_speaking:
            if audio_buffer:
                print('[Speaking; ASR disabled]')
                audio_buffer.clear()

            return

        if voice_detected:
            if not audio_buffer:
                print('Transcribing... ')
                audio_buffer.extend(state.most_recent_audios)
            else:
                audio_buffer.append(audio)
            return

        if audio_buffer:
            full_audio_data = np.concatenate([a.signal for a in audio_buffer]).flatten()
            sample_rate = audio_buffer[0].sample_rate
            audio_buffer.clear()

            full_audio = Audio(full_audio_data, sample_rate)

            if full_audio.len_seconds >= args.min_duration:
                asr.queue.put(full_audio)
            else:
                print('[Too short]')

    async def handle_speaking(topic, speaking: bool) -> None:
        state.self_speaking = speaking

    await node.listen(Topic.VOICE_DETECTED, handle_voice_detected)
    await node.listen(Topic.SPEAKING, handle_speaking)

    try:
        await forever()
    except KeyboardInterrupt:
        pass
    finally:
        asr.stop()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)

    parser.add_argument(
        '--min-duration',
        type=float,
        default=1.5,
        help='Minimum duration of audio, in seconds, to transcribe. Default: %(default)s',
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
