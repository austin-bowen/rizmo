import asyncio
from argparse import Namespace

import easymesh
from easymesh.asyncio import forever
from voicebox import ParallelVoicebox, Voicebox, reliable_tts
from voicebox.audio import Audio
from voicebox.effects import Flanger, Tail
from voicebox.sinks import Sink, SoundDevice
from voicebox.tts import AmazonPolly, ESpeakNG, PrerecordedTTS, TTS
from voicebox.voiceboxes.splitter import PunktSentenceSplitter

from rizmo.aws import get_polly_client
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    node = await easymesh.build_mesh_node_from_args(args=args)
    loop = asyncio.get_event_loop()

    def handle_speech_start() -> None:
        send_speaking_message(True)

    def handle_speech_end() -> None:
        send_speaking_message(False)

    def send_speaking_message(speaking: bool) -> None:
        asyncio.run_coroutine_threadsafe(
            node.send(Topic.SPEAKING, speaking),
            loop,
        ).result()

    with build_voicebox(
            args.tts,
            handle_speech_start,
            handle_speech_end,
    ) as voicebox:
        async def handle_say(topic, message: str) -> None:
            print(repr(message))
            voicebox.say(message)

        await node.listen(Topic.SAY, handle_say)
        await forever()


def build_voicebox(
        tts_name: str,
        handle_speech_start,
        handle_speech_end,
) -> Voicebox:
    PunktSentenceSplitter.download_resources()

    return ParallelVoicebox(
        tts=_build_tts(tts_name),
        effects=[
            Tail(0.5),
            Flanger(),
        ],
        sink=SinkWithCallbacks(
            SoundDevice(latency=0.5),
            handle_speech_start,
            handle_speech_end,
        ),
        text_splitter=PunktSentenceSplitter(),
    )


def _build_tts(tts_name: str) -> TTS:
    ttss = []
    if tts_name == 'kevin':
        ttss.append(AmazonPolly(
            client=get_polly_client(timeout=5),
            voice_id='Kevin',
            engine='neural',
            language_code='en-US',
            sample_rate=16_000,
        ))
    elif tts_name != 'espeak':
        raise ValueError(f'Unknown TTS engine: {tts_name!r}')

    ttss.append(ESpeakNG())

    tts = reliable_tts(ttss=ttss)

    try:
        print('Pre-loading TTS messages...')
        tts = PrerecordedTTS.from_tts(
            tts,
            texts=[
                'Network reconnected!',
                'Network disconnected; attempting to reconnect...',
            ],
        )
    except Exception as e:
        print(f'Failed to pre-load TTS messages: {e!r}')

    return tts


class SinkWithCallbacks(Sink):
    def __init__(self, sink: Sink, on_speech_start, on_speech_end) -> None:
        self.sink = sink
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

    def play(self, audio: Audio) -> None:
        self.on_speech_start()
        try:
            self.sink.play(audio)
        finally:
            self.on_speech_end()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)

    parser.add_argument(
        '--tts',
        default='kevin',
        choices=('espeak', 'kevin'),
        help='The text-to-speech engine to use. Default: %(default)s',
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
