from rosy.node.node import ServiceProxy

from rizmo.llm_utils import Tool


class FacesTool(Tool):
    def __init__(self, face_cmd_service: ServiceProxy):
        self.face_cmd_service = face_cmd_service

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
                            enum=['add', 'list'],
                        ),
                        name=dict(
                            type='string',
                            description='For "add" command: The name of the face to add.',
                        ),
                    ),
                    required=['action'],
                    additionalProperties=False,
                ),
            ),
        )

    async def call(self, action: str, name: str = None) -> str:
        if action == 'add':
            return await self.face_cmd_service(action, name=name)
        elif action == 'list':
            return await self.face_cmd_service(action)
        else:
            raise ValueError(f'Invalid action: {action}')
