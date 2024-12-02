"""
Followed instructions here:
- https://huggingface.co/learn/audio-course/en/chapter5/asr_models
- https://huggingface.co/openai/whisper-large-v3-turbo
"""

import asyncio
from argparse import Namespace
from collections import deque
from collections.abc import Callable
from queue import Queue
from threading import Thread

import numpy as np
import torch
from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import Audio
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
            array=audio.data.copy(),
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

        duration = sum(a.duration for a in self._buffer)
        while duration > self.max_duration:
            removed_audio = self._buffer.popleft()
            duration -= removed_audio.duration


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
    transcript_topic = node.get_topic_sender('transcript')

    loop = asyncio.get_event_loop()

    def handle_transcript(transcript: str) -> None:
        print(repr(transcript))

        asyncio.run_coroutine_threadsafe(
            transcript_topic.send(transcript),
            loop,
        ).result()

    asr = build_asr_thread(handle_transcript)

    most_recent_audios = MaxDurationAudioBuffer(max_duration=PRE_BUFFER_DURATION_S)
    audio_buffer = []

    async def handle_voice_detected(topic, data):
        audio, timestamp, voice_detected = data

        most_recent_audios.append(audio)

        if voice_detected:
            if not audio_buffer:
                print('Transcribing... ')
                audio_buffer.extend(most_recent_audios)
            else:
                audio_buffer.append(audio)
            return

        if audio_buffer:
            full_audio_data = np.concatenate([a.data for a in audio_buffer]).flatten()
            sample_rate = audio_buffer[0].sample_rate
            audio_buffer.clear()

            full_audio = Audio(full_audio_data, sample_rate)
            asr.queue.put(full_audio)

    await node.listen('voice_detected', handle_voice_detected)

    try:
        await forever()
    except KeyboardInterrupt:
        pass
    finally:
        asr.stop()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser('asr')
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
