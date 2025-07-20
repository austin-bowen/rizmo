from easymesh.node.node import TopicProxy

from rizmo.llm_utils import Tool


class FacesTool(Tool):
    def __init__(self, face_cmd_topic: TopicProxy):
        self.face_cmd_topic = face_cmd_topic

    @property
    def schema(self) -> dict:
        return dict(
            type='function',
            function=dict(
                name='faces',
                description='Manage stored faces.',
                parameters=dict(
                    type='object',
                    properties=dict(
                        action=dict(
                            type='string',
                            description='The action to perform.',
                            enum=['add'],
                        ),
                        name=dict(
                            type='string',
                            description='The name of the face to add.',
                        ),
                    ),
                    required=['action', 'name'],
                    additionalProperties=False,
                ),
            ),
        )

    async def call(self, action: str, name: str) -> str:
        if action == 'add':
            command = dict(action=action, name=name)
            await self.face_cmd_topic.send(command)
            return 'Done'
        else:
            raise ValueError(f'Invalid action: {action}')
