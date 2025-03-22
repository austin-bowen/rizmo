from rizmo.conference_speaker import ConferenceSpeaker
from rizmo.llm_utils import Tool


class VolumeTool(Tool):
    def __init__(self, speaker: ConferenceSpeaker):
        self.speaker = speaker

    @property
    def schema(self) -> dict:
        return dict(
            type='function',
            function=dict(
                name='volume',
                description='Set speaker volume.',
                parameters=dict(
                    type='object',
                    properties=dict(
                        volume=dict(
                            type='number',
                            description='Desired speaker volume in range 0-10.',
                        ),
                    ),
                    required=['volume'],
                    additionalProperties=False,
                ),
            ),
        )

    async def call(self, volume: int) -> str:
        volume *= 10
        volume = min(max(0, volume), 100)
        volume = int(volume)

        try:
            self.speaker.speaker.setvolume(volume)
        except Exception as e:
            return repr(e)
        else:
            return 'ok'
