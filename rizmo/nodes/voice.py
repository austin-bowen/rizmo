import asyncio
import random
from argparse import Namespace

import easymesh
import nltk
from easymesh.asyncio import forever
from nltk.tokenize import PunktTokenizer
from voicebox import ParallelVoicebox, Voicebox, reliable_tts
from voicebox.audio import Audio
from voicebox.effects import Flanger, Tail
from voicebox.sinks import Sink, SoundDevice
from voicebox.tts import AmazonPolly, ESpeakNG
from voicebox.voiceboxes.splitter import NltkTokenizerSplitter

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
        voicebox.say(random.choice([
            'Hello, world!',
            'I am awake.',
            'I am online.',
            "It's good to be back!",
        ]))

        async def handle_say(topic, message: str) -> None:
            print(repr(message))
            voicebox.say(message)

        await node.listen(Topic.SAY, handle_say)
        await forever()


def build_voicebox(
        tts: str,
        handle_speech_start,
        handle_speech_end,
) -> Voicebox:
    ttss = []
    if tts == 'kevin':
        ttss.append(AmazonPolly(
            client=get_polly_client(),
            voice_id='Kevin',
            engine='neural',
            language_code='en-US',
            sample_rate=16_000,
        ))
    elif tts != 'espeak':
        raise ValueError(f'Unknown TTS engine: {tts!r}')

    ttss.append(ESpeakNG())

    sink = SinkWithCallbacks(
        SoundDevice(latency=0.2),
        handle_speech_start,
        handle_speech_end,
    )

    PunktSentenceSplitter.download_resources()

    return ParallelVoicebox(
        tts=reliable_tts(ttss=ttss),
        effects=[
            Tail(0.5),
            Flanger(),
        ],
        sink=sink,
        text_splitter=PunktSentenceSplitter(),
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


# TODO Move this fix into voicebox
class PunktSentenceSplitter(NltkTokenizerSplitter):
    """
    Uses the `Punkt <https://www.nltk.org/api/nltk.tokenize.punkt.html>`_
    sentence tokenizer from `NLTK <https://www.nltk.org>`_ to split text into
    sentences more intelligently than a simple pattern-based splitter. It can
    handle instances of mid-sentence punctuation very well; e.g. "Mr. Jones went
    to see Dr. Sherman" would be correctly "split" into only one sentence.

    This requires that the Punkt NLTK resources be located on disk,
    e.g. by downloading via one of these methods:

        >>> PunktSentenceSplitter.download_resources()

    or

        >>> import nltk; nltk.download('punkt_tab')

    or

        $ python -m nltk.downloader punkt_tab

    See here for all NLTK Data installation methods:
    https://www.nltk.org/data.html
    """

    def __init__(self, language: str = 'english'):
        tokenizer = PunktTokenizer(language)
        super().__init__(tokenizer)

    @staticmethod
    def download_resources(**kwargs):
        """Download the Punkt NLTK resources."""
        nltk.download('punkt_tab', **kwargs)  # pragma: no cover


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
