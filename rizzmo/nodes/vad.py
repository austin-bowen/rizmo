"""
Setup:
- `pip install funasr modelscope torchaudio`
"""

import asyncio
from functools import wraps

from easymesh import build_mesh_node
from easymesh.asyncio import forever
from easymesh.node.node import MeshNode
from easymesh.types import Data, Topic
from funasr import AutoModel

from rizzmo.config import config


def depends_on_listener(node: MeshNode, downstream_topic: Topic):
    def decorator(func):
        @wraps(func)
        async def wrapper(topic_: Topic, data_: Data) -> None:
            if not await node.topic_has_listeners(downstream_topic):
                await node.stop_listening(topic_)
                await node.wait_for_listener(downstream_topic)
                await node.listen(topic_, wrapper)
                return

            return await func(topic_, data_)

        return wrapper

    return decorator


async def main():
    node = await build_mesh_node(
        name='vad',
        coordinator_host=config.coordinator.host,
        coordinator_port=config.coordinator.port,
    )

    voice_detected_topic = node.get_topic_sender('voice_detected')

    vad_model = AutoModel(
        model='fsmn-vad',
        device='cuda',
    )

    # wav_file = wave.open('/home/austin/Downloads/recording.vad.wav', 'wb')
    # wav_file.setnchannels(1)
    # wav_file.setsampwidth(2)
    # wav_file.setframerate(16000)

    voice_detected = False
    model_cache = {}

    # @depends_on_listener(node, voice_detected_topic.topic)
    async def handle_audio(topic, data):
        audio, timestamp = data
        block_size = audio.data.shape[0]
        indata = audio.data.squeeze()
        chunk_size_ms = int(round(1000 * block_size / audio.sample_rate))

        res = vad_model.generate(
            input=indata,
            cache=model_cache,
            is_final=False,
            chunk_size=chunk_size_ms,
        )

        nonlocal voice_detected
        for detection in res[0]['value']:
            if detection[0] != -1:
                voice_detected = True
            elif detection[1] != -1:
                voice_detected = False
        print('Voice detected:', voice_detected)

        await voice_detected_topic.send((audio, timestamp, voice_detected))

        # if voice_detected:
        #     wav_data = indata * 32767
        #     wav_data = wav_data.astype(np.int16)
        #     wav_file.writeframes(wav_data.tobytes())

    await node.listen('audio', handle_audio)

    try:
        await forever()
    finally:
        wav_file.close()


if __name__ == '__main__':
    asyncio.run(main())
