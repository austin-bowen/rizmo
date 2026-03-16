import asyncio
import logging
from argparse import Namespace

import rosy
from voicebox import ParallelVoicebox, Voicebox, reliable_tts
from voicebox.audio import Audio
from voicebox.effects import Flanger, Tail
from voicebox.sinks import Sink, SoundDevice
from voicebox.tts import AmazonPolly, ESpeakNG, ElevenLabsTTS, PrerecordedTTS, TTS
from voicebox.voiceboxes.splitter import PunktSentenceSplitter

from rizmo import secrets
from rizmo.aws import get_polly_client
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    logging.basicConfig(level=args.log)

    async with await rosy.build_node_from_args(args=args) as node:
        await _main(args, node)


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)

    parser.add_argument(
        '--voice',
        default='grete',
        choices=('blondie', 'espeak', 'grete', 'kevin'),
        help='The voice to use. Default: %(default)s',
    )

    return parser.parse_args()


async def _main(args: Namespace, node) -> None:
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
            args.voice,
            handle_speech_start,
            handle_speech_end,
    ) as voicebox:
        async def handle_say(topic, message: str) -> None:
            print(repr(message))
            voicebox.say(message)

        await node.listen(Topic.SAY, handle_say)

        print(f'[{node}] Ready')
        await node.forever()


def build_voicebox(
        voice_name: str,
        handle_speech_start,
        handle_speech_end,
) -> Voicebox:
    PunktSentenceSplitter.download_resources()

    return ParallelVoicebox(
        tts=_build_tts(voice_name),
        effects=[
            Tail(0.5),
            Flanger(),
        ],
        sink=SinkWithCallbacks(
            SoundDevice(latency=0.5),
            handle_speech_start,
            handle_speech_end,
        ),
        # text_splitter=PunktSentenceSplitter(),
    )


def _build_tts(voice_name: str) -> TTS:
    ttss = []
    if voice_name == 'blondie':
        ttss.append(_build_elevenlabs_tts(
            voice_id='XXphLKNRxvJ1Qa95KBhX',
            convert_kwargs=dict(
                # https://elevenlabs.io/docs/overview/models#models-overview
                model_id='eleven_flash_v2_5'
            )
        ))
    elif voice_name == 'grete':
        ttss.append(_build_elevenlabs_tts(
            voice_id='ONFS8Q3TuiPLQCXXa4dy',
            convert_kwargs=dict(
                # https://elevenlabs.io/docs/overview/models#models-overview
                model_id='eleven_flash_v2_5',
                voice_settings=dict(
                    speed=1.1,
                )
            )
        ))
    elif voice_name == 'kevin':
        ttss.append(AmazonPolly(
            client=get_polly_client(timeout=5),
            voice_id='Kevin',
            engine='neural',
            language_code='en-US',
            sample_rate=16_000,
        ))
    elif voice_name != 'espeak':
        raise ValueError(f'Unknown TTS engine: {voice_name!r}')

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


def _build_elevenlabs_tts(
        voice_id: str,
        convert_kwargs: dict,
) -> ElevenLabsTTS:
    return ElevenLabsTTS(
        voice_id=voice_id,
        api_key=secrets.ELEVENLABS_API_KEY,
        sample_rate=16_000,
        convert_kwargs=convert_kwargs,
    )


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


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
