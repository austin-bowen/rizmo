from rizmo.llm_utils import Tool
from rizmo.nodes.messages import MotorSystemCommand


class MotorSystemTool(Tool):
    def __init__(self, motor_system_topic):
        self.motor_system_topic = motor_system_topic

    @property
    def schema(self) -> dict:
        return dict(
            type='function',
            function=dict(
                name='motor_system',
                description='Controls the motor system.',
                parameters=dict(
                    type='object',
                    properties=dict(
                        enabled=dict(
                            type='boolean',
                            description='Whether to enable or disable the motor system.',
                        ),
                    ),
                    required=['enabled'],
                    additionalProperties=False,
                ),
            ),
        )

    async def call(self, enabled: bool) -> None:
        await self.motor_system_topic.send(MotorSystemCommand(enabled=enabled))
